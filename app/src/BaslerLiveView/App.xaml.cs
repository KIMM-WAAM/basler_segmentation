using System;
using System.IO;
using System.Windows;
using System.Xml.Linq;

namespace BaslerLiveView;

public partial class App : Application
{
    public static string ConfigPath { get; private set; } = CameraConfig.DefaultPath;

    public static string ProfilePath(string dataset) => dataset.ToLowerInvariant() switch
    {
        "kimm" => CameraConfig.DefaultPath,
        "bestfa" => Path.Combine(AppContext.BaseDirectory, "config", "config.bestfa.xml"),
        _ => throw new ArgumentException("Dataset must be kimm or bestfa", nameof(dataset)),
    };

    protected override void OnStartup(StartupEventArgs e)
    {
        int datasetIndex = Array.IndexOf(e.Args, "--dataset");
        if (datasetIndex >= 0)
        {
            try
            {
                if (datasetIndex + 1 >= e.Args.Length)
                    throw new ArgumentException("--dataset requires kimm or bestfa");
                ConfigPath = ProfilePath(e.Args[datasetIndex + 1]);
                // An explicit profile must exist and parse; never fall back to another environment.
                var root = XDocument.Load(ConfigPath).Root;
                if (root?.Name != "Config") throw new InvalidDataException("Invalid capture profile");
                string expected = e.Args[datasetIndex + 1].ToLowerInvariant();
                if (CameraConfig.Load(ConfigPath).RecordSessionPrefix != expected)
                    throw new InvalidDataException("Capture profile has a different session prefix");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Capture profile", MessageBoxButton.OK, MessageBoxImage.Error);
                Shutdown(1);
                return;
            }
        }
        // Optional: launch with `--emulate` (or set PYLON_CAMEMU) to expose
        // Basler's software camera emulator when no physical camera is present.
        // Must be set before any pylon API call (i.e. before the first enumerate).
        foreach (var arg in e.Args)
        {
            if (arg.Equals("--emulate", StringComparison.OrdinalIgnoreCase))
            {
                if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("PYLON_CAMEMU")))
                    Environment.SetEnvironmentVariable("PYLON_CAMEMU", "1");
            }
        }

        base.OnStartup(e);
    }
}
