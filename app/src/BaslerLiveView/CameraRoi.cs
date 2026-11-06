using System;

namespace BaslerLiveView;

/// <summary>The actual acquisition window, read back after the camera aligns it.</summary>
public readonly record struct CameraRoi(int X, int Y, int Width, int Height,
    int SensorWidth, int SensorHeight)
{
    public bool IsCropped => X != 0 || Y != 0 || Width != SensorWidth || Height != SensorHeight;

    /// <summary>Translate an inference crop's origin to full-sensor coordinates.</summary>
    public (int X, int Y)? InferenceOrigin(int cropX, int cropY, bool softwareCrop) =>
        IsCropped || softwareCrop ? (X + cropX, Y + cropY) : null;

    internal static long Align(long value, long increment, long min, long max)
    {
        value = Math.Clamp(value, min, max);
        return increment > 1 ? value - (value - min) % increment : value;
    }

    internal static long AtPercent(double percent, long increment, long min, long max) =>
        Align(min + (long)Math.Round((max - min) * Math.Clamp(percent, 0, 100) / 100.0),
            increment, min, max);
}
