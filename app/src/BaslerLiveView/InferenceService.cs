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

/// <summary>
/// Runs YOLO on incoming BGRA camera frames using YoloDotNet on the DirectML (GPU)
/// execution provider. The task follows the model file: an instance-segmentation model
/// draws masks, an oriented-bounding-box (OBB) model draws rotated boxes — both come out
/// of training/ (train.deploy.kimm.yaml / train.deploy.bestfa.yaml), so switching is just pointing
/// config.xml &lt;Inference&gt;&lt;Model&gt; at the other .onnx.
///
/// Inference is far slower than the grab rate, so frames are processed on a private
/// worker thread with a "latest frame wins" policy: <see cref="Submit"/> only ever
/// keeps the newest frame and drops any backlog, so the grab loop and UI never block
/// waiting on the GPU. Each finished frame is published via <see cref="FrameProcessed"/>.
///
/// A frame is one or more crops — one per ROI (config.xml &lt;Crop&gt;&lt;Roi&gt;; BestFA has
/// two, one wire each). Every crop gets its own inference run on the same session, one
/// after the other, and keeps its own <c>MaxDetections</c> best detections.
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

    /// <summary>
    /// One ROI's crop. The pending pixels are owned by this service (copied out of the reused
    /// camera buffer); the rest are worker-thread buffers reused across frames: a crop is
    /// ~6.5 MB (1280×1280 BGRA), and allocating that per frame lands on the Large Object
    /// Heap, which is only reclaimed by full (gen2) GCs — at tens of frames per second those
    /// show up as stutters.
    /// </summary>
    private sealed class Slot
    {
        // Written by Submit under _lock …
        public byte[] Pending = Array.Empty<byte>();
        public int PendingWidth;
        public int PendingHeight;
        public (int X, int Y)? PendingOrigin;
        // … copied under _lock by the worker, which alone reads these afterwards — a crop
        // resized mid-inference then can't disagree with the bitmap being processed.
        public int Width;
        public int Height;
        public (int X, int Y)? Origin;
        public SKBitmap? Bitmap;
        public byte[] Before = Array.Empty<byte>();   // crop as submitted, before drawing
        public readonly byte[][] Outputs = { Array.Empty<byte>(), Array.Empty<byte>() };
    }

    private readonly List<Slot> _slots = new();
    private int _pendingCount;
    private bool _hasPending;
    private volatile bool _running = true;
    private int _nextOutput;

    /// <summary>Raised per processed frame with one <see cref="InferenceLayer"/> per submitted
    /// crop, in submit order. Two output buffers alternate per crop, so a layer stays valid
    /// only until the next event — the receiver must drop its references to the previous ones
    /// by the time the handler returns (MainWindow swaps them under its overlay lock).</summary>
    public event Action<InferenceLayer[]>? FrameProcessed;

    /// <summary>Detection text of the latest processed frame (class · score · wire tip),
    /// one item per line. Set on the worker thread right before <see cref="FrameProcessed"/>.</summary>
    public string Info { get; private set; } = "";

    /// <summary>Raised for inference failures on the worker thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    /// <param name="modelPath">Path to the ONNX model (segmentation or OBB).</param>
    /// <param name="maxDetections">Detections kept per crop, highest score first
    /// (= wires in view of one ROI).</param>
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
    /// Hand the newest frame — its first <paramref name="count"/> crops, one per ROI — to the
    /// inference worker. Copies the pixels out of the (reused) camera/crop buffers
    /// immediately, so the caller may return at once. If a previous frame is still waiting
    /// it is simply overwritten. Each crop's Origin is where it sits in the full sensor frame
    /// (null = not cropped); it is only used to also print the wire tip in full-frame pixels.
    /// </summary>
    public void Submit(CropSubmission[] crops, int count)
    {
        lock (_lock)
        {
            while (_slots.Count < count)
                _slots.Add(new Slot());
            for (int i = 0; i < count; i++)
            {
                var (width, height, bgra, origin) = crops[i];
                var slot = _slots[i];
                int needed = width * height * 4;
                if (slot.Pending.Length < needed)
                    slot.Pending = new byte[needed];
                Buffer.BlockCopy(bgra, 0, slot.Pending, 0, needed);
                slot.PendingWidth = width;
                slot.PendingHeight = height;
                slot.PendingOrigin = origin;
            }
            _pendingCount = count;
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

            int count;
            lock (_lock)
            {
                if (!_hasPending) continue;
                _hasPending = false;
                count = _pendingCount;
                for (int i = 0; i < count; i++)
                {
                    var slot = _slots[i];
                    (slot.Width, slot.Height, slot.Origin) = (slot.PendingWidth, slot.PendingHeight, slot.PendingOrigin);
                    int size = slot.Width * slot.Height * 4;
                    if (slot.Bitmap == null || slot.Bitmap.Width != slot.Width || slot.Bitmap.Height != slot.Height)
                    {
                        // Camera frames are BGRA8 packed, which maps directly onto Skia's
                        // Bgra8888 memory layout — no channel conversion needed.
                        slot.Bitmap?.Dispose();
                        slot.Bitmap = new SKBitmap(new SKImageInfo(slot.Width, slot.Height,
                            SKColorType.Bgra8888, SKAlphaType.Unpremul));
                    }
                    Marshal.Copy(slot.Pending, 0, slot.Bitmap.GetPixels(), size);
                    if (slot.Before.Length != size)
                        slot.Before = new byte[size];
                    Buffer.BlockCopy(slot.Pending, 0, slot.Before, 0, size);
                }
            }

            try
            {
                var layers = new InferenceLayer[count];
                var info = new List<string>();
                for (int i = 0; i < count; i++)
                {
                    var slot = _slots[i];
                    var lines = RunInference(slot, count > 1 ? i : null);
                    if (count > 1 && lines == NoDetection)
                        lines = $"ROI {i + 1}: {NoDetection}";
                    info.Add(lines);
                    layers[i] = new InferenceLayer(slot.Width, slot.Height, slot.Outputs[_nextOutput]);
                }
                _nextOutput ^= 1;
                Info = string.Join(Environment.NewLine, info);
                FrameProcessed?.Invoke(layers);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(ex);
            }
        }
    }

    private const string NoDetection = "no detection";

    /// <summary>Run one crop: detect, draw onto its bitmap, and write the drawing into the
    /// slot's current output as a layer — the drawn pixels, everything else 0 (see
    /// <see cref="InferenceLayer"/>). Returns the info text lines for this crop.</summary>
    private string RunInference(Slot slot, int? roi)
    {
        var bitmap = slot.Bitmap!;
        var origin = slot.Origin;
        // Draw() mutates the SKBitmap in place, overlaying masks / rotated boxes.
        var found = new List<(string Name, double Confidence, SKPoint? Tip)>();
        if (ModelTask == ModelType.ObbDetection)
        {
            var results = _yolo.RunObbDetection(bitmap, confidence: _confidence, iou: 0.7)
                .OrderByDescending(r => r.Confidence).Take(_maxDetections)
                .OrderBy(r => r.BoundingBox.MidX).ToList();
            bitmap.Draw(results, _obbDrawing);
            var xs = results.Select(r => r.BoundingBox.MidX).ToList();
            found.AddRange(results.Select((r, i) => (NameOf(r.Label, i, xs, bitmap.Width, roi), r.Confidence,
                (SKPoint?)WireTip.FromObb(r.BoundingBox, r.OrientationAngle))));
        }
        else
        {
            var results = _yolo.RunSegmentation(bitmap, confidence: _confidence, pixelConfedence: 0.5, iou: 0.7)
                .OrderByDescending(r => r.Confidence).Take(_maxDetections)
                .OrderBy(r => r.BoundingBox.MidX).ToList();
            bitmap.Draw(results, _drawing);
            var xs = results.Select(r => r.BoundingBox.MidX).ToList();
            found.AddRange(results.Select((r, i) => (NameOf(r.Label, i, xs, bitmap.Width, roi), r.Confidence,
                WireTip.FromMask(r.BitPackedPixelMask, r.BoundingBox))));
        }
        string info = Annotate(bitmap, found, origin);

        // Keep only what was drawn: pixels that differ from the submitted crop, forced
        // opaque; untouched pixels become 0. MainWindow lays that over the live frame, so
        // the view stays live everywhere and overlapping ROIs don't erase each other's
        // drawings. Written into the output buffer the receiver is not holding.
        int size = slot.Width * slot.Height * 4;
        ref byte[] output = ref slot.Outputs[_nextOutput];
        if (output.Length != size)
            output = new byte[size];
        var after = MemoryMarshal.Cast<byte, uint>(bitmap.GetPixelSpan()[..size]);
        var before = MemoryMarshal.Cast<byte, uint>(slot.Before.AsSpan(0, size));
        var layer = MemoryMarshal.Cast<byte, uint>(output.AsSpan(0, size));
        for (int i = 0; i < layer.Length; i++)
            layer[i] = after[i] == before[i] ? 0u : after[i] | 0xFF000000u;
        return info;
    }

    /// <summary>With several ROIs (one wire each) the wire is named after its ROI: "wire 1",
    /// "wire 2" — left to right, same numbering as the labeller. Otherwise "wire L" / "wire R"
    /// when the rig has two wires in one crop (MaxDetections = 2): with both found, by order
    /// (results are already left to right); with only one, by which half of the crop its box
    /// center falls in. Any other count keeps the bare class name.</summary>
    private string NameOf(LabelModel label, int index, IReadOnlyList<int> centersX, int width, int? roi)
    {
        if (roi is int r)
            return _maxDetections == 1 ? $"{label.Name} {r + 1}" : $"{label.Name} {r + 1}.{index + 1}";
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
    private static string Annotate(SKBitmap bitmap, List<(string Name, double Confidence, SKPoint? Tip)> found,
        (int X, int Y)? origin)
    {
        using var canvas = new SKCanvas(bitmap);
        using var paint = new SKPaint { IsAntialias = true };
        float radius = Math.Max(6, bitmap.Width / 100f);

        var lines = new List<string>();
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
        if (lines.Count == 0)
            return NoDetection;
        return string.Join(Environment.NewLine, lines);
    }

    public void Dispose()
    {
        _running = false;
        _frameSignal.Set();
        bool stopped = _worker.Join(TimeSpan.FromSeconds(2));
        _frameSignal.Dispose();
        _yolo.Dispose();
        if (stopped)   // still in use if the worker is stuck in inference
            foreach (var slot in _slots)
                slot.Bitmap?.Dispose();
    }
}

/// <summary>One ROI's crop handed to <see cref="InferenceService.Submit"/>: its size, BGRA
/// pixels, and where it sits in the full sensor frame (null = not cropped).</summary>
public readonly record struct CropSubmission(int Width, int Height, byte[] Bgra, (int X, int Y)? Origin);

/// <summary>One ROI's inference drawing, the size of its crop: BGRA pixels where masks,
/// boxes and the tip dot were drawn (opaque), 0 everywhere else — laid over the live frame
/// at the crop's position.</summary>
public readonly record struct InferenceLayer(int Width, int Height, byte[] Bgra);
