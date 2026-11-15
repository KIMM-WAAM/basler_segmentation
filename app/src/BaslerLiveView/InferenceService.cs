using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.ExecutionProvider.DirectML;
using YoloDotNet.Models;

namespace BaslerLiveView;

/// <summary>One crop handed to the model: BGRA pixels, where it sits in the full sensor
/// frame (null = not cropped; only used to also print the wire tip in sensor pixels), and
/// an optional window name that labels its detections ("wire L").</summary>
public readonly record struct InferenceInput(int Width, int Height, byte[] Bgra, (int X, int Y)? Origin,
    string? Name = null);

/// <summary>Annotated crop, same size and order as its <see cref="InferenceInput"/>.</summary>
public readonly record struct InferenceOutput(int Width, int Height, byte[] Bgra);

/// <summary>
/// Runs YOLO on incoming BGRA camera frames using YoloDotNet on the DirectML (GPU)
/// execution provider. The task follows the model file: an instance-segmentation model
/// draws masks, an oriented-bounding-box (OBB) model draws rotated boxes — both come out
/// of training/ (train.seg_all.yaml / train.obb_all.yaml), so switching is just pointing
/// config.xml &lt;Inference&gt;&lt;Model&gt; at the other .onnx.
///
/// Inference is far slower than the grab rate, so frames are processed on a private
/// worker thread with a "latest frame wins" policy: <see cref="Submit(IReadOnlyList{InferenceInput})"/>
/// only ever keeps the newest frame and drops any backlog, so the grab loop and UI never block
/// waiting on the GPU. Each finished frame is published via <see cref="FrameProcessed"/>.
///
/// A frame may consist of several crops (config.xml <c>&lt;Rois&gt;</c> — e.g. BestFA, one wire
/// per 640×640 window): each crop goes through the model on its own, in order, and the frame is
/// published once all of them are done. A single crop is just a one-element frame.
/// </summary>
public sealed class InferenceService : IDisposable
{
    private readonly Yolo _yolo;
    private readonly SegmentationDrawingOptions _drawing;
    private readonly DetectionDrawingOptions _obbDrawing;
    private readonly Thread _worker;
    private readonly AutoResetEvent _frameSignal = new(false);
    private readonly object _lock = new();

    // Per crop: drop detections scoring below _confidence, then keep the _maxDetections
    // most confident (config.xml <Inference><MaxDetections> / <Confidence>). The kept ones
    // are listed left to right (box center x), so with several wires each keeps its slot
    // in the info text instead of swapping whenever their scores cross.
    private readonly int _maxDetections;
    private readonly double _confidence;

    /// <summary>Per-crop state, one per position in the submitted frame. The pending copy is
    /// guarded by _lock; the bitmap and output buffers belong to the worker thread.</summary>
    private sealed class Slot
    {
        public byte[] Pending = Array.Empty<byte>();
        public int Width;
        public int Height;
        public (int X, int Y)? Origin;
        public string? Name;

        // Reused across frames: a crop is ~6.5 MB (1280×1280 BGRA), and allocating that per
        // frame lands on the Large Object Heap, which is only reclaimed by full (gen2) GCs —
        // at tens of frames per second those show up as stutters.
        public SKBitmap? Bitmap;
        public readonly byte[][] Outputs = { Array.Empty<byte>(), Array.Empty<byte>() };
    }

    // Latest submitted frame, owned by this service (copied out of the reused camera buffer).
    private readonly List<Slot> _slots = new();
    private int _pendingCount;
    private bool _hasPending;
    private volatile bool _running = true;
    private int _nextOutput;

    /// <summary>Raised per processed frame with one annotated crop per submitted crop, in the
    /// same order. Each crop's two output buffers alternate, so a buffer stays valid only until
    /// the next event — the receiver must drop its references to the previous ones by the time
    /// the handler returns (MainWindow swaps them under its overlay lock).</summary>
    public event Action<IReadOnlyList<InferenceOutput>>? FrameProcessed;

    /// <summary>Detection text of the latest processed frame (class · score · wire tip),
    /// one item per line. Set on the worker thread right before <see cref="FrameProcessed"/>.</summary>
    public string Info { get; private set; } = "";

    /// <summary>Raised for inference failures on the worker thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    /// <param name="modelPath">Path to the ONNX model (segmentation or OBB).</param>
    /// <param name="maxDetections">Detections kept per frame, highest score first
    /// (= wires in view).</param>
    /// <param name="confidence">Minimum detection score, 0–1.</param>
    /// <param name="gpuId">DirectML device id (0 = default GPU, -1 = CPU fallback).</param>
    public InferenceService(string modelPath, int maxDetections = 1, double confidence = 0.24, int gpuId = 0)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model not found: {modelPath}", modelPath);

        _maxDetections = Math.Max(1, maxDetections);
        _confidence = confidence;

        // The execution backend is decided purely by gpuId in the provider:
        // >= 0 uses DirectML on that GPU device, < 0 runs on CPU.
        Backend = gpuId >= 0 ? $"GPU (DirectML dev {gpuId})" : "CPU";

        _yolo = new Yolo(new YoloOptions
        {
            ExecutionProvider = new DirectMLExecutionProvider(modelPath, gpuId),
            // Stretch to the model's input size; matches the segmentation demo defaults.
            ImageResize = ImageResize.Stretched,
            SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None),
        });

        // YoloDotNet reads the task from the ONNX metadata but only exposes it through
        // ModelInfo ("<ModelType> (yolo vNN)"), so parse the leading enum name back.
        ModelTask = Enum.TryParse(_yolo.ModelInfo.Split(' ')[0], out ModelType task)
            ? task
            : ModelType.Segmentation;
        if (ModelTask is not (ModelType.Segmentation or ModelType.ObbDetection))
        {
            _yolo.Dispose();
            throw new NotSupportedException(
                $"Unsupported model task {ModelTask} ({Path.GetFileName(modelPath)}) — " +
                "expected a segmentation or OBB model.");
        }

        // No axis-aligned box, a faint mask (32/255 ≈ 12%) under a thick outline, and the
        // wire tip as a red dot (Annotate()); class · score · tip text is shown by MainWindow.
        // An OBB model's rotated box is the prediction itself, so that one is drawn.
        _drawing = new SegmentationDrawingOptions
        {
            DrawBoundingBoxes = false,
            DrawLabels = false,
            DrawConfidenceScore = false,
            BorderThickness = 5,
            BoundingBoxOpacity = 128,
            EnableDynamicScaling = true,
            DrawSegmentationPixelMask = true,
            PixelMaskOpacity = 32,
            DrawContour = true,
            ContourThickness = 5,
        };
        _obbDrawing = (DetectionDrawingOptions)_drawing;
        _obbDrawing.DrawBoundingBoxes = true;

        _worker = new Thread(WorkerLoop)
        {
            IsBackground = true,
            Name = "YoloInference",
        };
        _worker.Start();
    }

    /// <summary>Execution backend in use: "GPU (DirectML dev N)" or "CPU".</summary>
    public string Backend { get; }

    /// <summary>The model's task: <see cref="ModelType.Segmentation"/> or
    /// <see cref="ModelType.ObbDetection"/>.</summary>
    public ModelType ModelTask { get; }

    /// <summary>Human-readable model description (type, version, input size, ...).</summary>
    public string ModelInfo => _yolo.ModelInfo ?? "unknown";

    /// <summary>
    /// Hand the newest frame (a single crop) to the inference worker — see
    /// <see cref="Submit(IReadOnlyList{InferenceInput})"/>.
    /// <paramref name="origin"/> is where this crop sits in the full sensor frame (null =
    /// not cropped); it is only used to also print the wire tip in full-frame pixels.
    /// </summary>
    public void Submit(int width, int height, byte[] bgra, (int X, int Y)? origin = null) =>
        Submit(new[] { new InferenceInput(width, height, bgra, origin) });

    /// <summary>
    /// Hand the newest frame — one or more crops of it — to the inference worker. Copies the
    /// pixels out of the (reused) camera/cropper buffers immediately, so the caller may return
    /// at once. If a previous frame is still being processed it is simply overwritten.
    /// </summary>
    public void Submit(IReadOnlyList<InferenceInput> inputs)
    {
        if (inputs.Count == 0) return;
        lock (_lock)
        {
            while (_slots.Count < inputs.Count)
                _slots.Add(new Slot());
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var slot = _slots[i];
                int needed = input.Width * input.Height * 4;
                if (slot.Pending.Length < needed)
                    slot.Pending = new byte[needed];
                Buffer.BlockCopy(input.Bgra, 0, slot.Pending, 0, needed);
                slot.Width = input.Width;
                slot.Height = input.Height;
                slot.Origin = input.Origin;
                slot.Name = input.Name;
            }
            _pendingCount = inputs.Count;
            _hasPending = true;
        }
        _frameSignal.Set();
    }

    private void WorkerLoop()
    {
        while (_running)
        {
            _frameSignal.WaitOne();
            if (!_running) return;

            // Snapshot each crop under the lock — Submit may overwrite the slots (pixels, size,
            // origin, name) for the next frame while this one is still being inferred.
            (Slot Slot, int Width, int Height, (int X, int Y)? Origin, string? Name)[] jobs;
            lock (_lock)
            {
                if (!_hasPending) continue;
                _hasPending = false;
                jobs = new (Slot, int, int, (int X, int Y)?, string?)[_pendingCount];
                for (int i = 0; i < jobs.Length; i++)
                {
                    var slot = _slots[i];
                    if (slot.Bitmap == null || slot.Bitmap.Width != slot.Width || slot.Bitmap.Height != slot.Height)
                    {
                        // Camera frames are BGRA8 packed, which maps directly onto Skia's
                        // Bgra8888 memory layout — no channel conversion needed.
                        slot.Bitmap?.Dispose();
                        slot.Bitmap = new SKBitmap(new SKImageInfo(slot.Width, slot.Height,
                            SKColorType.Bgra8888, SKAlphaType.Unpremul));
                    }
                    Marshal.Copy(slot.Pending, 0, slot.Bitmap.GetPixels(), slot.Width * slot.Height * 4);
                    jobs[i] = (slot, slot.Width, slot.Height, slot.Origin, slot.Name);
                }
            }

            try
            {
                var outputs = new InferenceOutput[jobs.Length];
                var lines = new List<string>();
                for (int i = 0; i < jobs.Length; i++)
                {
                    var (slot, width, height, origin, name) = jobs[i];
                    outputs[i] = new InferenceOutput(width, height, RunInference(slot.Bitmap!, slot, origin, name, lines));
                }
                _nextOutput ^= 1;
                Info = string.Join(Environment.NewLine, lines);
                FrameProcessed?.Invoke(outputs);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(ex);
            }
        }
    }

    /// <summary>Run the model on one crop (drawn into <paramref name="bitmap"/> in place), append
    /// its info lines, and return the annotated pixels in the slot's output buffer for this frame.</summary>
    private byte[] RunInference(SKBitmap bitmap, Slot slot, (int X, int Y)? origin, string? name, List<string> lines)
    {
        // Draw() mutates the SKBitmap in place, overlaying masks / rotated boxes.
        var found = new List<(string Name, double Confidence, SKPoint? Tip)>();
        if (ModelTask == ModelType.ObbDetection)
        {
            var results = _yolo.RunObbDetection(bitmap, confidence: _confidence, iou: 0.7)
                .OrderByDescending(r => r.Confidence).Take(_maxDetections)
                .OrderBy(r => r.BoundingBox.MidX).ToList();
            bitmap.Draw(results, _obbDrawing);
            var xs = results.Select(r => r.BoundingBox.MidX).ToList();
            found.AddRange(results.Select((r, i) => (NameOf(r.Label, i, xs, bitmap.Width, name), r.Confidence,
                (SKPoint?)WireTip.FromObb(r.BoundingBox, r.OrientationAngle))));
        }
        else
        {
            var results = _yolo.RunSegmentation(bitmap, confidence: _confidence, pixelConfedence: 0.5, iou: 0.7)
                .OrderByDescending(r => r.Confidence).Take(_maxDetections)
                .OrderBy(r => r.BoundingBox.MidX).ToList();
            bitmap.Draw(results, _drawing);
            var xs = results.Select(r => r.BoundingBox.MidX).ToList();
            found.AddRange(results.Select((r, i) => (NameOf(r.Label, i, xs, bitmap.Width, name), r.Confidence,
                WireTip.FromMask(r.BitPackedPixelMask, r.BoundingBox))));
        }
        Annotate(bitmap, found, origin, name, lines);

        // Read the annotated pixels back out (still Bgra8888) into the output buffer the
        // receiver is not holding (see FrameProcessed).
        int size = bitmap.ByteCount;
        ref byte[] annotated = ref slot.Outputs[_nextOutput];
        if (annotated.Length != size)
            annotated = new byte[size];
        Marshal.Copy(bitmap.GetPixels(), annotated, 0, size);
        return annotated;
    }

    /// <summary>A named window (config.xml <c>&lt;Rois&gt;</c>) labels its detections with its
    /// name: "wire L". Otherwise "wire L" / "wire R" when the rig has two wires in one crop
    /// (MaxDetections = 2): with both found, by order (results are already left to right);
    /// with only one, by which half of the crop its box center falls in. Any other count keeps
    /// the bare class name.</summary>
    private string NameOf(LabelModel label, int index, IReadOnlyList<int> centersX, int width, string? window)
    {
        if (window != null)
            return $"{label.Name} {window}";
        if (_maxDetections != 2)
            return label.Name;
        bool left = centersX.Count >= 2 ? index == 0 : centersX[index] < width / 2;
        return $"{label.Name} {(left ? "L" : "R")}";
    }

    /// <summary>
    /// Wire tip as a red dot on the crop, and per detection the lines
    /// "wire L 0.75" / "crop (498.0, 334.8)" / "full (1082.0, 718.8)" — crop in crop pixels
    /// (only when cropped), full in sensor pixels. The text is not burnt into the crop
    /// (it came out too small once the crop is letterboxed); MainWindow shows it top-left
    /// of the view instead.
    /// </summary>
    private static void Annotate(SKBitmap bitmap, List<(string Name, double Confidence, SKPoint? Tip)> found,
        (int X, int Y)? origin, string? window, List<string> lines)
    {
        using var canvas = new SKCanvas(bitmap);
        using var paint = new SKPaint { IsAntialias = true };
        float radius = Math.Max(6, bitmap.Width / 100f);

        foreach (var (name, confidence, tip) in found)
        {
            lines.Add(FormattableString.Invariant($"{name} {confidence:0.00}"));
            if (tip is SKPoint p)
            {
                paint.Color = SKColors.White;
                canvas.DrawCircle(p, radius + Math.Max(2, radius / 3), paint);
                paint.Color = SKColors.Red;
                canvas.DrawCircle(p, radius, paint);
                var (ox, oy) = origin ?? (0, 0);
                if (origin is not null)
                    lines.Add(FormattableString.Invariant($"crop ({p.X:0.0}, {p.Y:0.0})"));
                lines.Add(FormattableString.Invariant($"full ({p.X + ox:0.0}, {p.Y + oy:0.0})"));
            }
        }
        if (found.Count == 0)
            lines.Add(window != null ? $"{window}: no detection" : "no detection");
    }

    public void Dispose()
    {
        _running = false;
        _frameSignal.Set();
        bool stopped = _worker.Join(TimeSpan.FromSeconds(2));
        _frameSignal.Dispose();
        _yolo.Dispose();
        if (stopped)              // still in use if the worker is stuck in inference
        {
            lock (_lock)
            {
                foreach (var slot in _slots)
                    slot.Bitmap?.Dispose();
            }
        }
    }
}
