using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Basler.Pylon;
using BaslerLiveView;

internal static class Program
{
    private static int _passed;

    [STAThread]
    private static int Main()
    {
        // Enumerate ONLY the emulator transport; never open a physical camera.
        Environment.SetEnvironmentVariable("PYLON_CAMEMU", "1");
        try
        {
            CheckConfig();
            CheckDatasetProfiles();
            CheckGeometry();
            CheckCamera();
            Console.WriteLine($"PASS: {_passed} ROI regression checks");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static void Assert(bool ok, string description)
    {
        if (!ok) throw new InvalidOperationException(description);
        _passed++;
        Console.WriteLine("PASS: " + description);
    }

    private static void CheckConfig()
    {
        var path = Path.GetTempFileName();
        try
        {
            foreach (var value in new[] { "true", "false", "invalid", "" })
            {
                string flag = value.Length == 0 ? "" : $"<hardware_roi>{value}</hardware_roi>";
                File.WriteAllText(path, $"<Config><Camera>{flag}</Camera><Crop><Width>320</Width><CenterX>45</CenterX></Crop></Config>");
                var config = CameraConfig.Load(path);
                Assert(config.HardwareRoi == (value == "true"), $"hardware_roi '{value}' has expected mode");
                Assert(config.CropWidth == 320 && config.CropCenterX == 45, "ROI parsing preserves later settings");
            }
        }
        finally { File.Delete(path); }
    }

    private static void CheckGeometry()
    {
        Assert(CameraRoi.Align(643, 4, 16, 1920) == 640, "sizes align down to hardware increment");
        Assert(CameraRoi.Align(1, 4, 16, 1920) == 16, "sizes clamp to minimum");
        Assert(CameraRoi.Align(2000, 4, 16, 1920) == 1920, "sizes clamp to maximum");
        Assert(CameraRoi.Align(10, 4, 3, 19) == 7, "alignment is relative to a nonzero minimum");
        Assert(CameraRoi.AtPercent(45, 8, 0, 1280) == 576, "ROI percentage is aligned to offset increment");
        Assert(CameraRoi.AtPercent(-10, 8, 0, 1280) == 0, "negative percentage clamps left");
        Assert(CameraRoi.AtPercent(110, 8, 0, 1280) == 1280, "excess percentage clamps right");
        var roi = new CameraRoi(576, 80, 640, 1280, 1920, 2080);
        Assert(roi.InferenceOrigin(0, 0, false) == (576, 80), "hardware ROI keeps full-sensor inference coordinates");
        Assert(roi.InferenceOrigin(10, 20, true) == (586, 100), "hardware and software offsets compose");
        var full = new CameraRoi(0, 0, 1920, 2080, 1920, 2080);
        Assert(full.InferenceOrigin(0, 0, false) == null, "full sensor has no crop origin");
        Assert(full.InferenceOrigin(25, 30, true) == (25, 30), "software crop coordinates remain unchanged");
    }

    private static void CheckDatasetProfiles()
    {
        var kimm = CameraConfig.Load(App.ProfilePath("kimm"));
        var bestfa = CameraConfig.Load(App.ProfilePath("bestfa"));
        Assert(kimm.RecordDir == "datasets/kimm/raw", "KIMM recording root is isolated");
        Assert(bestfa.RecordDir == "datasets/bestfa/raw", "BestFA recording root is isolated");
        Assert(bestfa.RecordSessionPrefix == "bestfa", "BestFA uses its own session prefix");
        Assert(bestfa.OverlayPath.StartsWith("datasets/bestfa/reference/"), "BestFA uses its own reference");
        Assert(!bestfa.CropEnabled, "BestFA does not inherit KIMM crop geometry");
        Assert(bestfa.MaxDetections == 2, "BestFA inference retains both wires");
        Assert(bestfa.Model != kimm.Model, "BestFA does not silently use a KIMM model");
        string directory = Path.Combine(AppContext.BaseDirectory, "recordings", Guid.NewGuid().ToString("N"));
        using var recorder = new FrameRecorder(directory, sessionPrefix: "bestfa");
        recorder.Start();
        Assert(new DirectoryInfo(recorder.SessionDirectory).Name == "Mono",
            "raw recordings use the Mono directory");
        Assert(new DirectoryInfo(recorder.SessionDirectory).Parent!.Name.StartsWith("bestfa_"),
            "recording session name matches BestFA");
    }

    private static void CheckCamera()
    {
        var info = CameraFinder.Enumerate(DeviceType.CameraEmulator).First();
        using var camera = new CameraService();
        var config = new CameraConfig
        {
            HardwareRoi = true,
            CropEnabled = true,
            CropWidth = 320,
            CropHeight = 240,
            CropCenterX = 45,
            CropCenterY = 10,
            FrameRate = 30,
        };
        camera.Open(info, config);
        var cropped = camera.Roi;
        Assert(cropped.Width == 320 && cropped.Height == 240 && cropped.IsCropped, "camera accepts hardware ROI");
        Assert(Math.Abs(cropped.X - (cropped.SensorWidth - 320) * 0.45) <= 1 &&
               Math.Abs(cropped.Y - (cropped.SensorHeight - 240) * 0.10) <= 1, "camera applies noncentered offsets");
        CheckFrame(camera, record: true);
        camera.Close();

        config.HardwareRoi = false;
        camera.Open(info, config);
        Assert(!camera.Roi.IsCropped && camera.Roi.Width == cropped.SensorWidth &&
               camera.Roi.Height == cropped.SensorHeight, "false restores the full sensor after hardware ROI");
        CheckFrame(camera, record: false);
        camera.Close();

        config.HardwareRoi = true;
        config.CropWidth = 0;
        camera.Open(info, config);
        Assert(camera.Roi.Width == cropped.SensorWidth && camera.Roi.Height == 240 && camera.Roi.X == 0,
            "zero width means full width without disabling height crop");
        camera.Close();

        config.CropEnabled = false;
        camera.Open(info, config);
        Assert(!camera.Roi.IsCropped, "disabled crop restores full sensor in hardware mode");
        camera.Close();

        config.CropEnabled = true;
        config.CropWidth = config.CropHeight = int.MaxValue;
        camera.Open(info, config);
        Assert(!camera.Roi.IsCropped, "oversized ROI clamps to full sensor");
        camera.Close();

        CheckWindow(info, cropped);
    }

    private static void CheckFrame(CameraService camera, bool record)
    {
        using var ready = new ManualResetEventSlim();
        (int Width, int Height, byte[] Pixels)? captured = null;
        Exception? failure = null;
        void OnFrame(int width, int height, byte[] pixels)
        {
            if (ready.IsSet) return;
            captured = (width, height, pixels[..(width * height * 4)]);
            ready.Set();
        }
        void OnError(Exception ex) { failure = ex; ready.Set(); }
        camera.FrameReady += OnFrame;
        camera.ErrorOccurred += OnError;
        try
        {
            camera.Start();
            Assert(ready.Wait(TimeSpan.FromSeconds(15)), "emulator supplies a frame");
        }
        finally
        {
            camera.Stop();
            camera.FrameReady -= OnFrame;
            camera.ErrorOccurred -= OnError;
        }
        if (failure != null) throw failure;
        var frame = captured ?? throw new InvalidOperationException("No camera frame");
        Assert(frame.Width == camera.Roi.Width && frame.Height == camera.Roi.Height, "frame dimensions match actual camera ROI");
        if (!record) return;

        // Artifacts stay under this test's ignored output directory.
        string directory = Path.Combine(AppContext.BaseDirectory, "recordings", Guid.NewGuid().ToString("N"));
        string session;
        using (var recorder = new FrameRecorder(directory))
        {
            recorder.Start();
            session = recorder.SessionDirectory;
            recorder.Submit(frame.Width, frame.Height, frame.Pixels);
        }
        var png = new BitmapImage(new Uri(Directory.GetFiles(session, "*.png").Single()));
        Assert(png.PixelWidth == frame.Width && png.PixelHeight == frame.Height, "REC preserves hardware ROI dimensions");
    }

    private static void CheckWindow(ICameraInfo info, CameraRoi roi)
    {
        // Use an artificial full-sensor reference to verify pixel alignment, without
        // opening a visible window or loading an inference model.
        string reference = Path.Combine(AppContext.BaseDirectory, "reference-test.png");
        var bitmap = BitmapSource.Create(roi.SensorWidth, roi.SensorHeight, 96, 96,
            PixelFormats.Gray8, null, new byte[roi.SensorWidth * roi.SensorHeight], roi.SensorWidth);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using (var file = File.Create(reference)) encoder.Save(file);

        string configPath = CameraConfig.DefaultPath;
        Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
        byte[]? original = File.Exists(configPath) ? File.ReadAllBytes(configPath) : null;
        try
        {
            foreach (bool hardware in new[] { false, true })
            {
                var xml = new System.Xml.Linq.XDocument(new System.Xml.Linq.XElement("Config",
                    new System.Xml.Linq.XElement("Camera", new System.Xml.Linq.XElement("hardware_roi", hardware)),
                    new System.Xml.Linq.XElement("Crop", new System.Xml.Linq.XElement("Width", 320),
                        new System.Xml.Linq.XElement("Height", 240), new System.Xml.Linq.XElement("CenterX", 45),
                        new System.Xml.Linq.XElement("CenterY", 10)),
                    new System.Xml.Linq.XElement("ReferenceOverlay", new System.Xml.Linq.XElement("Path", reference))));
                xml.Save(configPath);
                var window = new MainWindow();
                var camera = Field<CameraService>(window, "_camera");
                try
                {
                    var cropper = Field<FrameCropper>(window, "_cropper");
                    Assert(cropper.Enabled == !hardware, "hardware mode bypasses software cropping; false preserves it");
                    Assert(((StackPanel)window.FindName("CropControls")).IsEnabled == !hardware,
                        "hardware mode locks ineffective live crop controls");
                    camera.Open(info, Field<CameraConfig>(window, "_config"));
                    typeof(MainWindow).GetMethod("LoadReferenceOverlay", BindingFlags.Instance | BindingFlags.NonPublic)!
                        .Invoke(window, null);
                    var source = ((Image)window.FindName("RefOverlayImage")).Source;
                    if (hardware)
                    {
                        Assert(source is CroppedBitmap crop && crop.SourceRect == new Int32Rect(roi.X, roi.Y, 320, 240),
                            "reference overlay uses actual hardware ROI rectangle");
                    }
                    else
                    {
                        Assert(source is BitmapSource full && full.PixelWidth == roi.SensorWidth,
                            "software mode keeps full-sensor reference overlay");
                    }
                }
                finally
                {
                    camera.Dispose();
                    Field<FrameRecorder>(window, "_recorder").Dispose();
                }
            }
        }
        finally
        {
            if (original != null) File.WriteAllBytes(configPath, original);
            else File.Delete(configPath);
        }
    }

    private static T Field<T>(object target, string name) =>
        (T)target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(target)!;
}
