using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Rectangle = System.Windows.Shapes.Rectangle;
using Basler.Pylon;

namespace BaslerLiveView;

public partial class MainWindow : Window
{
    private readonly CameraService _camera = new();
    private readonly CameraConfig _config = CameraConfig.Load(App.ConfigPath);

    // YOLO inference overlay — segmentation or OBB, per the model file (lazily created).
    private InferenceService? _infer;
    private volatile bool _inferEnabled;
    // Model file name comes from config\config.xml (<Inference><Model>),
    // resolved under Models\ beside the exe. Falls back to the config default.
    private readonly string _modelPath;

    // The crop windows feed inference and mirror what training/preprocess.py will cut the
    // recorded PNGs down to — one per ROI (config.xml <Crop><Roi>; BestFA: two, one wire
    // each, same order as the labeller's ROI 1 / ROI 2). The toolbar edits the selected one.
    // With hardware ROI enabled, the camera already crops the (single) window, so software
    // cropping must be bypassed.
    private readonly FrameCropper[] _croppers;
    private int _activeRoi;
    private readonly RoiOutline[] _outlines;
    private readonly CropSubmission[] _submissions;
    private readonly FrameRecorder _recorder;

    // Latest inference drawings from the worker, one per ROI, laid over the full frame at
    // each crop window's position. Written by the inference thread, read by the grab
    // thread → guarded by _overlayLock.
    private readonly object _overlayLock = new();
    private InferenceLayer[] _layers = Array.Empty<InferenceLayer>();
    private bool _hasOverlay;

    // Scratch full-frame buffer used to composite the overlay; grab thread only.
    private byte[] _composed = Array.Empty<byte>();

    // Set once the crop controls hold real values; guards the ValueChanged/
    // TextChanged handlers that fire while we are still populating them.
    private bool _cropUiReady;

    private WriteableBitmap? _bitmap;
    private int _bmpWidth;
    private int _bmpHeight;

    // FPS measurement.
    private readonly Stopwatch _fpsClock = Stopwatch.StartNew();
    private int _frameCount;

    // Set once the window starts closing so late grab-thread frames stop
    // touching the (soon-to-be-gone) Dispatcher.
    private bool _closing;

    /// <summary>ComboBox row: wraps an ICameraInfo with a friendly label.</summary>
    private sealed record CameraItem(ICameraInfo Info, string DisplayName);

    public MainWindow()
    {
        InitializeComponent();

        _modelPath = Path.Combine(AppContext.BaseDirectory, "Models", _config.Model);
        _recorder = new FrameRecorder(
            FrameRecorder.ResolveDirectory(_config.RecordDir), _config.RecordQueueCapacity)
        {
            SaveFps = _config.RecordFps,
        };
        _recorder.ErrorOccurred += ex =>
            Dispatcher.BeginInvoke(() => StatusText.Text = "Recording error: " + ex.Message);
        Title += " — " + _config.CaptureEnvironment;

        // One cropper (+ on-screen outline) per ROI, seeded from config. The toolbar shows the
        // selected ROI; its handlers push the values back onto that cropper.
        _croppers = _config.CropRois.Select(r => new FrameCropper
        {
            Width = r.Width, Height = r.Height, CenterXPercent = r.CenterX, CenterYPercent = r.CenterY,
        }).ToArray();
        _submissions = new CropSubmission[_croppers.Length];
        _outlines = _croppers.Select(_ => new RoiOutline(OverlayCanvas)).ToArray();
        if (_croppers.Length > 1)
        {
            for (int i = 0; i < _croppers.Length; i++)
                RoiCombo.Items.Add($"ROI {i + 1}");
            RoiCombo.SelectedIndex = 0;
            RoiCombo.Visibility = Visibility.Visible;
        }
        CropCheck.IsChecked = _config.CropEnabled;
        ShowActiveRoi();
        _cropUiReady = true;
        ApplyCropSettings();
        CropControls.IsEnabled = !_config.HardwareRoi;
        if (_config.HardwareRoi)
        {
            CropCheck.Content = "Hardware ROI";
            CropControls.ToolTip = "Camera ROI is set in config.xml. Restart the app after changing it.";
            RecordButton.ToolTip = "Save the camera ROI as PNGs (pixels outside the ROI are not captured)";
        }

        LoadReferenceOverlay();

        _camera.StatusChanged += msg => Dispatcher.BeginInvoke(() => StatusText.Text = msg);
        _camera.ErrorOccurred += ex => Dispatcher.BeginInvoke(() => StatusText.Text = "Error: " + ex.Message);
        _camera.FrameReady += OnFrameReady;

        Loaded += (_, _) =>
        {
            RefreshCameras();
            if (_config.Warning != null)
                StatusText.Text = _config.Warning + "  |  " + StatusText.Text;
        };
    }

    /// <summary>Short one-line description of the configured segmentation model:
    /// which file is expected (from config) and whether it is actually present.
    /// The real YOLO version is confirmed on the exe side once segmentation is
    /// toggled on (see <see cref="Inference_Toggled"/>).</summary>
    private string DescribeModel()
    {
        var file = Path.GetFileName(_modelPath);
        if (File.Exists(_modelPath))
        {
            var mb = new FileInfo(_modelPath).Length / (1024.0 * 1024.0);
            return $"Model: {file} ✓ ({mb:0.#} MB) — toggle Inference to load & confirm task/version";
        }
        return $"Model: {file} ✗ MISSING — place it in Models\\ (see config.xml)";
    }

    private void Refresh_Click(object sender, RoutedEventArgs e) => RefreshCameras();

    private void RefreshCameras()
    {
        try
        {
            CameraCombo.Items.Clear();
            foreach (var info in CameraService.Enumerate())
            {
                var model = info[CameraInfoKey.ModelName];
                var serial = info[CameraInfoKey.SerialNumber];
                CameraCombo.Items.Add(new CameraItem(info, $"{model}  [{serial}]"));
            }

            if (CameraCombo.Items.Count > 0)
            {
                CameraCombo.SelectedIndex = 0;
                StatusText.Text = $"Found {CameraCombo.Items.Count} camera(s).  |  {DescribeModel()}";
            }
            else
            {
                StatusText.Text = $"No cameras found. Connect a device, or relaunch with --emulate for the software emulator.  |  {DescribeModel()}";
            }
        }
        catch (Exception ex)
        {
            StatusText.Text = "Enumeration failed: " + ex.Message;
        }
    }

    private void Connect_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var item = CameraCombo.SelectedItem as CameraItem;
            _camera.Open(item?.Info, _config);
            lock (_overlayLock) _hasOverlay = false;
            LoadReferenceOverlay();

            ConnectButton.IsEnabled = false;
            DisconnectButton.IsEnabled = true;
            StartButton.IsEnabled = true;
            CameraCombo.IsEnabled = false;
            RefreshButton.IsEnabled = false;
        }
        catch (Exception ex)
        {
            StatusText.Text = "Connect failed: " + ex.Message;
        }
    }

    private void Disconnect_Click(object sender, RoutedEventArgs e)
    {
        _camera.Close();
        ConnectButton.IsEnabled = true;
        DisconnectButton.IsEnabled = false;
        StartButton.IsEnabled = false;
        StopButton.IsEnabled = false;
        CameraCombo.IsEnabled = true;
        RefreshButton.IsEnabled = true;
    }

    private void Start_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _fpsClock.Restart();
            _frameCount = 0;
            _camera.Start();
            StartButton.IsEnabled = false;
            StopButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            StatusText.Text = "Start failed: " + ex.Message;
        }
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
        _camera.Stop();
        StartButton.IsEnabled = true;
        StopButton.IsEnabled = false;
        FpsText.Text = "";
    }

    // --- Reference overlay ------------------------------------------------

    /// <summary>
    /// Load the reference cutout (the fixed rig — metal tube + wire — cut out of
    /// datasets/kimm/reference/reference.png by training/tools/make_reference_cutout.py) and show it over the
    /// live view. Eye-in-hand camera → the rig occupies the same pixels in every
    /// frame, so a static overlay lines up; only the wire's length differs from the
    /// reference shot. A missing file just leaves the checkbox disabled.
    /// </summary>
    private void LoadReferenceOverlay()
    {
        // Two layers from training/tools/make_reference_cutout.py: the faint cutout fill and
        // the crisp green silhouette outline. Each is optional — whatever file
        // exists is shown; with neither present the checkbox is disabled.
        RefOverlayImage.Opacity = _config.OverlayOpacity;
        RefOutlineImage.Opacity = _config.OutlineOpacity;
        bool fill = TryLoadOverlayLayer(RefOverlayImage, _config.OverlayPath);
        bool line = TryLoadOverlayLayer(RefOutlineImage, _config.OutlinePath);

        if (!fill && !line)
        {
            RefOverlayCheck.IsChecked = false;
            RefOverlayCheck.IsEnabled = false;
            RefOverlayCheck.ToolTip = _config.HardwareRoi
                ? "Connect the camera and use a reference image matching the full sensor size."
                : "Reference cutout not found — generate it with training\\tools\\make_reference_cutout.py";
            return;
        }
        RefOverlayCheck.IsEnabled = true;
        RefOverlayCheck.ToolTip = "Show the reference cutout (fixed tube + wire) over the live view";
        RefOverlayCheck.IsChecked = _config.OverlayEnabled; // handler sets visibility
        UpdateReferenceOverlayVisibility();
    }

    /// <summary>Load one overlay PNG into <paramref name="target"/>; false if the
    /// file is missing or unreadable (the layer then simply stays hidden).</summary>
    private bool TryLoadOverlayLayer(Image target, string configuredPath)
    {
        target.Source = null;
        target.Visibility = Visibility.Collapsed;
        if (_config.HardwareRoi && !_camera.IsOpen) return false;
        var path = FrameRecorder.ResolveDirectory(configuredPath);
        if (!File.Exists(path)) return false;
        try
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.UriSource = new Uri(path);
            bmp.CacheOption = BitmapCacheOption.OnLoad; // read fully now; don't lock the file
            bmp.EndInit();
            bmp.Freeze();
            if (_config.HardwareRoi && _camera.Roi.IsCropped)
            {
                var roi = _camera.Roi;
                // Reference layers are full-sensor images. Crop at the actual
                // hardware-aligned offsets instead of stretching them over the ROI.
                if (bmp.PixelWidth != roi.SensorWidth || bmp.PixelHeight != roi.SensorHeight)
                    return false;
                var cropped = new CroppedBitmap(bmp, new Int32Rect(roi.X, roi.Y, roi.Width, roi.Height));
                cropped.Freeze();
                target.Source = cropped;
            }
            else
            {
                target.Source = bmp;
            }
            return true;
        }
        catch (Exception ex)
        {
            StatusText.Text = $"Reference overlay load failed ({Path.GetFileName(path)}): {ex.Message}";
            return false;
        }
    }

    private void RefOverlay_Toggled(object sender, RoutedEventArgs e) => UpdateReferenceOverlayVisibility();

    private void UpdateReferenceOverlayVisibility()
    {
        bool on = RefOverlayCheck.IsChecked == true;
        RefOverlayImage.Visibility = on && RefOverlayImage.Source != null ? Visibility.Visible : Visibility.Collapsed;
        RefOutlineImage.Visibility = on && RefOutlineImage.Source != null ? Visibility.Visible : Visibility.Collapsed;
    }

    // --- Crop -------------------------------------------------------------

    private void Crop_Changed(object sender, RoutedEventArgs e) => ApplyCropSettings();

    private void CropText_Changed(object sender, TextChangedEventArgs e) => ApplyCropSettings();

    private void CropSlider_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => ApplyCropSettings();

    private void CropCenter_Click(object sender, RoutedEventArgs e)
    {
        CropXSlider.Value = 50;
        CropYSlider.Value = 50;
    }

    private void Roi_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (!_cropUiReady || RoiCombo.SelectedIndex < 0) return;
        _activeRoi = RoiCombo.SelectedIndex;
        _cropUiReady = false;           // populating the controls must not write back
        ShowActiveRoi();
        _cropUiReady = true;
        ApplyCropSettings();
    }

    /// <summary>Load the selected ROI's window into the toolbar controls.</summary>
    private void ShowActiveRoi()
    {
        var c = _croppers[_activeRoi];
        CropWidthBox.Text = c.Width.ToString();
        CropHeightBox.Text = c.Height.ToString();
        CropXSlider.Value = c.CenterXPercent;
        CropYSlider.Value = c.CenterYPercent;
    }

    /// <summary>Croppers in use this frame: every ROI while cropping is on; with cropping off
    /// (or hardware ROI) just the first, which then passes the whole frame through.</summary>
    private int ActiveRoiCount => _croppers[0].Enabled ? _croppers.Length : 1;

    /// <summary>Push the toolbar's crop controls onto the selected ROI's cropper. Invalid or
    /// empty size text is simply ignored, so typing "6" on the way to "640" is harmless.
    /// 0 is accepted and means the full frame on that axis (preprocess.py convention).
    /// The Crop checkbox switches every ROI together.</summary>
    private void ApplyCropSettings()
    {
        if (!_cropUiReady) return;

        bool enabled = !_config.HardwareRoi && CropCheck.IsChecked == true;
        foreach (var c in _croppers)
            c.Enabled = enabled;
        var cropper = _croppers[_activeRoi];
        if (int.TryParse(CropWidthBox.Text, out int w) && w >= 0) cropper.Width = w;
        if (int.TryParse(CropHeightBox.Text, out int h) && h >= 0) cropper.Height = h;
        cropper.CenterXPercent = CropXSlider.Value;
        cropper.CenterYPercent = CropYSlider.Value;

        CropXText.Text = $"{CropXSlider.Value:F1}%";
        CropYText.Text = $"{CropYSlider.Value:F1}%";

        UpdateCropOverlay();
    }

    private void ViewHost_SizeChanged(object sender, SizeChangedEventArgs e) => UpdateCropOverlay();

    /// <summary>
    /// Draw the crop windows over the live image. The image is shown with
    /// <c>Stretch="Uniform"</c>, so it is letterboxed inside the host: reproduce that
    /// fit here to map crop pixels onto screen coordinates. Hidden until a frame has
    /// arrived — the source resolution is unknown before that.
    /// </summary>
    private void UpdateCropOverlay()
    {
        bool ready = _bmpWidth > 0 && _bmpHeight > 0
                     && ViewHost.ActualWidth > 0 && ViewHost.ActualHeight > 0;
        double scale = ready ? Math.Min(ViewHost.ActualWidth / _bmpWidth, ViewHost.ActualHeight / _bmpHeight) : 0;
        double originX = (ViewHost.ActualWidth - _bmpWidth * scale) / 2;
        double originY = (ViewHost.ActualHeight - _bmpHeight * scale) / 2;
        bool multi = _croppers.Length > 1;

        for (int i = 0; i < _croppers.Length; i++)
        {
            var c = _croppers[i];
            int x = 0, y = 0, w = 0, h = 0;
            bool show = ready && c.TryGetWindow(_bmpWidth, _bmpHeight, out x, out y, out w, out h);
            string label = (multi ? $"ROI {i + 1}  " : "") +
                           $"{w}×{h}  ({c.CenterXPercent:F1}%, {c.CenterYPercent:F1}%)";
            // The selected ROI solid, the others dashed (only matters with several).
            _outlines[i].Place(show, originX + x * scale, originY + y * scale, w * scale, h * scale,
                label, active: !multi || i == _activeRoi, labelInside: i > 0);
        }
    }

    /// <summary>One ROI's on-screen outline, same look as the labeller: dark under-stroke so
    /// the red line reads on bright and dark backgrounds, plus a size/position label.</summary>
    private sealed class RoiOutline
    {
        private readonly Rectangle _shadow = new() { Stroke = new SolidColorBrush(Color.FromArgb(0xB0, 0, 0, 0)), StrokeThickness = 4 };
        private readonly Rectangle _line = new() { Stroke = new SolidColorBrush(Color.FromRgb(0xFF, 0x3B, 0x30)), StrokeThickness = 2 };
        private readonly TextBlock _label = new()
        {
            Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0x6B, 0x60)), FontSize = 11, FontWeight = FontWeights.Bold,
        };
        private readonly Border _labelBox;

        public RoiOutline(Canvas canvas)
        {
            _labelBox = new Border { Background = new SolidColorBrush(Color.FromArgb(0xB0, 0, 0, 0)), Padding = new Thickness(4, 1, 4, 1), Child = _label };
            foreach (UIElement e in new UIElement[] { _shadow, _line, _labelBox })
            {
                e.Visibility = Visibility.Collapsed;
                canvas.Children.Add(e);
            }
        }

        public void Place(bool show, double left, double top, double width, double height,
            string label, bool active, bool labelInside)
        {
            var visibility = show ? Visibility.Visible : Visibility.Collapsed;
            _shadow.Visibility = _line.Visibility = _labelBox.Visibility = visibility;
            if (!show) return;

            foreach (var rect in new[] { _shadow, _line })
            {
                Canvas.SetLeft(rect, left);
                Canvas.SetTop(rect, top);
                rect.Width = width;
                rect.Height = height;
            }
            _line.StrokeDashArray = active ? null : new DoubleCollection { 3, 2 };
            _line.Opacity = _labelBox.Opacity = active ? 1 : 0.6;

            _label.Text = label;
            Canvas.SetLeft(_labelBox, left);
            // Above the box normally; tucked inside when it would fall off the top edge. Later
            // ROIs put theirs inside at the bottom so neighbouring labels don't collide.
            _labelBox.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            double labelHeight = _labelBox.DesiredSize.Height;
            Canvas.SetTop(_labelBox, labelInside ? top + height - labelHeight - 2
                : top - labelHeight - 2 >= 0 ? top - labelHeight - 2 : top + 2);
        }
    }

    // --- Recording --------------------------------------------------------

    private void Record_Toggled(object sender, RoutedEventArgs e)
    {
        if (RecordButton.IsChecked == true)
        {
            try
            {
                _recorder.Start();
                var rate = _recorder.SaveFps > 0 ? $"{_recorder.SaveFps:0.##} fps" : "every frame";
                StatusText.Text = $"REC ({rate}) → {_recorder.SessionDirectory}";
            }
            catch (Exception ex)
            {
                RecordButton.IsChecked = false;
                StatusText.Text = "Recording start failed: " + ex.Message;
            }
        }
        else
        {
            _recorder.Stop();
            StatusText.Text = $"REC stopped — {_recorder.SavedCount + _recorder.PendingCount} frame(s) → {_recorder.SessionDirectory}";
        }
    }

    // Raised on the pylon grab-loop thread → marshal to UI, then blit.
    private void OnFrameReady(int width, int height, byte[] bgra)
    {
        // Once the window is closing the Dispatcher is going away; don't block
        // the grab thread on it (that races shutdown and crashes the process).
        if (_closing || Dispatcher.HasShutdownStarted)
            return;

        // Save exactly what the camera supplies: full sensor in software mode,
        // hardware ROI otherwise. Already-cropped recordings must not receive the
        // original sensor crop a second time during training preprocessing.
        // No-op unless recording and the save-rate sampler is due; copies the pixels
        // out before returning.
        _recorder.Submit(width, height, bgra);

        var infer = _infer;
        if (_inferEnabled && infer != null)
        {
            // Inference runs on the crops only — the same regions the training
            // images are cut down to (the session's crop.yaml ROIs in the labeller), so the
            // model sees at inference exactly the framing it was trained on. Each cropper
            // owns its output buffer, so the crops stay valid side by side until Submit.
            int count = ActiveRoiCount;
            for (int i = 0; i < count; i++)
            {
                var cropped = _croppers[i].Crop(width, height, bgra, out int cw, out int ch, out int cx, out int cy);
                // Include the camera ROI offset so "full" remains a sensor coordinate
                // in both modes, even when the camera rounded the requested ROI.
                _submissions[i] = new CropSubmission(cw, ch, cropped,
                    _camera.Roi.InferenceOrigin(cx, cy, !ReferenceEquals(cropped, bgra)));
            }

            // Submit() copies the pixels out of the reused buffers immediately, so we
            // can return without blocking the grab loop; the drawings arrive later via
            // OnInferredFrame and are laid over the frame by Compose() below.
            infer.Submit(_submissions, count);
            bgra = Compose(width, height, bgra);
        }

        // Synchronous Invoke: keeps the grab thread paused until WPF has copied
        // the pixels out of the shared buffer, avoiding tearing/overwrite races.
        try
        {
            Dispatcher.Invoke(() => RenderFrame(width, height, bgra));
        }
        catch (OperationCanceledException)
        {
            // Dispatcher shut down between the check above and the Invoke.
        }
    }

    /// <summary>
    /// Copy the full frame into a scratch buffer and lay each ROI's newest inference drawing
    /// over its own window, so the operator sees the live sensor image with masks, boxes and
    /// tip dots drawn inside the crop windows. Only drawn pixels are copied (a layer is 0
    /// elsewhere), so overlapping ROIs keep both drawings and the rest of the view stays
    /// live. Grab thread only.
    ///
    /// Inference lags the grab rate, so the drawings are a few frames old. If a crop size
    /// changed since the result was produced that layer is skipped rather than stretched;
    /// the next inference brings it back a fraction of a second later.
    /// </summary>
    private byte[] Compose(int width, int height, byte[] bgra)
    {
        int needed = width * height * 4;
        if (_composed.Length < needed)
            _composed = new byte[needed];
        Buffer.BlockCopy(bgra, 0, _composed, 0, needed);

        lock (_overlayLock)
        {
            if (!_hasOverlay) return _composed;
            var dst = MemoryMarshal.Cast<byte, uint>(_composed.AsSpan(0, needed));
            int count = Math.Min(_layers.Length, ActiveRoiCount);
            for (int i = 0; i < count; i++)
            {
                var layer = _layers[i];
                _croppers[i].TryGetWindow(width, height, out int x, out int y, out int cw, out int ch);
                if (cw != layer.Width || ch != layer.Height) continue;
                var src = MemoryMarshal.Cast<byte, uint>(layer.Bgra.AsSpan(0, cw * ch * 4));
                for (int row = 0; row < ch; row++)
                {
                    var s = src.Slice(row * cw, cw);
                    var d = dst.Slice((y + row) * width + x, cw);
                    for (int j = 0; j < s.Length; j++)
                        if (s[j] != 0) d[j] = s[j];
                }
            }
        }

        return _composed;
    }

    // Raised on the inference worker thread. The service alternates two output buffers per
    // ROI, so the previous ones are reused as soon as this returns — swapping the references
    // under _overlayLock (which Compose also holds while reading) is what keeps that safe.
    // They are not rendered directly: display is driven entirely by the grab loop, which lays
    // them over the frame at the crop windows' positions (see Compose).
    private void OnInferredFrame(InferenceLayer[] layers)
    {
        lock (_overlayLock)
        {
            _layers = layers;
            _hasOverlay = true;
        }

        string info = _infer?.Info ?? "";
        Dispatcher.BeginInvoke(() =>
        {
            if (!_inferEnabled) return;
            InferInfoText.Text = info;
            InferInfoBox.Visibility = Visibility.Visible;
        });
    }

    private void Inference_Toggled(object sender, RoutedEventArgs e)
    {
        if (InferenceCheck.IsChecked == true)
        {
            try
            {
                if (_infer == null)
                {
                    // Model load + DirectML session init blocks briefly (~1-2s).
                    _infer = new InferenceService(_modelPath, _config.MaxDetections, _config.Confidence);
                    _infer.FrameProcessed += OnInferredFrame;
                    _infer.ErrorOccurred += ex =>
                        Dispatcher.BeginInvoke(() => StatusText.Text = "Inference error: " + ex.Message);
                }
                _inferEnabled = true;
                StatusText.Text = $"Inference ON — {_infer.ModelTask} · {_infer.ModelInfo} · {_infer.Backend}  [{Path.GetFileName(_modelPath)}]";
            }
            catch (Exception ex)
            {
                _inferEnabled = false;
                InferenceCheck.IsChecked = false;
                StatusText.Text = "Inference load failed: " + ex.Message;
            }
        }
        else
        {
            _inferEnabled = false;
            // Drop the last annotated crop, or it would stay frozen on screen.
            lock (_overlayLock) _hasOverlay = false;
            InferInfoBox.Visibility = Visibility.Collapsed;
            StatusText.Text = "Inference OFF.";
        }
    }

    private void RenderFrame(int width, int height, byte[] bgra)
    {
        if (_bitmap == null || _bmpWidth != width || _bmpHeight != height)
        {
            _bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            _bmpWidth = width;
            _bmpHeight = height;
            LiveImage.Source = _bitmap;

            // Only now is the source resolution known, so the crop rectangle can be
            // placed. Size is stable afterwards, so this stays off the per-frame path.
            UpdateCropOverlay();
        }

        _bitmap.WritePixels(new Int32Rect(0, 0, width, height), bgra, width * 4, 0);

        _frameCount++;
        if (_fpsClock.ElapsedMilliseconds >= 500)
        {
            double fps = _frameCount * 1000.0 / _fpsClock.ElapsedMilliseconds;
            FpsText.Text = $"{fps:F1} fps   {width}×{height}";
            _frameCount = 0;
            _fpsClock.Restart();

            // Piggy-backs on the fps tick so the counters cost nothing per frame.
            RecText.Text = _recorder.IsRecording || _recorder.PendingCount > 0
                ? $"● REC  {_recorder.SavedCount} saved" +
                  (_recorder.PendingCount > 0 ? $"  (+{_recorder.PendingCount} queued)" : "") +
                  (_recorder.DroppedCount > 0 ? $"  {_recorder.DroppedCount} dropped" : "")
                : "";
        }
    }

    // Closing via the window's X does an automatic Stop → Disconnect. The camera
    // teardown blocks until the grab loop drains, so we hold the close, run it
    // off the UI thread (keeping the Dispatcher free to release any in-flight
    // frame callback → no deadlock), then close for real.
    private async void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        if (_closing)
            return; // cleanup already ran on the first pass; let the window close.

        _closing = true;
        e.Cancel = true;                    // hold the close until teardown finishes
        _inferEnabled = false;                // stop routing frames to the segmentation worker
        _camera.FrameReady -= OnFrameReady; // stop feeding frames to the UI
        _recorder.Stop();                   // stop accepting frames (queued ones still flush)

        try
        {
            // Stop()/Close() + segmentation worker teardown off the UI thread, so
            // the Dispatcher stays free to release any in-flight frame callback.
            // The recorder is disposed last so its queue drains to disk first.
            await Task.Run(() =>
            {
                _camera.Dispose();
                _infer?.Dispose();
                _recorder.Dispose();
            });
        }
        catch (Exception ex)
        {
            Debug.WriteLine("Shutdown teardown failed: " + ex);
        }

        Close(); // re-enters Window_Closing with _closing == true → window closes
    }
}
