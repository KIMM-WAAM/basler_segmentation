using System;
using System.Collections.Generic;
using SkiaSharp;

namespace BaslerLiveView;

/// <summary>
/// Wire tip — the same definition as WAAM_Monitoring's WireTip and training/render.py (mask_tip),
/// so the app, the monitoring app and the Python tools report the same point (TipMethod = <see cref="Method"/>).
///
/// Segmentation masks (<see cref="FromMask"/>): the tip comes from the shape of the mask's end, not from
/// the whole mask's rectangle.
///  1. Keep the largest 8-connected component (arc glow / spatter fragments near the tip are ignored).
///  2. Axis = principal axis (PCA) of the mask pixels. Which end is the tip comes from the minimum-area
///     rotated rectangle: its short sides when length/width >= <see cref="ElongatedRatio"/>, otherwise
///     (the wire only shows a stub) the more vertical axis — and of its two ends the lower one, since the
///     wire comes down out of the tube. A stub uses that vertical axis instead of PCA.
///  3. Tip = the farthest pixel along the axis (along position) and the lateral centre of the pixels
///     within half a width of it (lateral position).
/// The old definition (midpoint of the rectangle's lower short side) tied the tip to the orientation of
/// the whole mask: on long KIMM wires (500-1100 px) a 1-2 degree turn moved it 10-20 px sideways.
/// Held-out split, predicted vs labelled tip (2026-11-17): KIMM test median 15.0 -> 8.4 px, within 20 px
/// 69 -> 91 %; BestFA val median 2.5 -> 0.9 px. The along-axis position is the same in both definitions.
///
/// OBB models (<see cref="FromObb"/>) have no mask: midpoint of the box's lower short side.
/// </summary>
public static class WireTip
{
    public const string Method = "end-region-pca";
    /// <summary>At least this elongated: the rectangle's short sides are the ends; otherwise pick the end by direction (the axis pointing down).</summary>
    public const float ElongatedRatio = 1.5f;

    /// <summary>Tip from a YoloDotNet bit-packed mask (box-relative, row-major, LSB-first),
    /// in image pixels. Null when the mask is empty.</summary>
    public static SKPoint? FromMask(byte[] packedMask, SKRectI box)
    {
        int w = box.Width, h = box.Height;
        if (packedMask.Length == 0 || w <= 0 || h <= 0)
            return null;
        var pixels = LargestComponent(packedMask, w, h);
        if (pixels.Count == 0)
            return null;

        // Row extents → pixel-edge corners (a pixel x spans [x, x+1]) → hull → min-area rectangle.
        var first = new int[h];
        var last = new int[h];
        Array.Fill(first, -1);
        foreach (int i in pixels)
        {
            int y = i / w, x = i - y * w;
            if (first[y] < 0 || x < first[y]) first[y] = x;
            if (x > last[y]) last[y] = x;
        }
        var corners = new List<SKPoint>(h * 4);
        for (int y = 0; y < h; y++)
        {
            if (first[y] < 0) continue;
            corners.Add(new SKPoint(first[y], y));
            corners.Add(new SKPoint(first[y], y + 1));
            corners.Add(new SKPoint(last[y] + 1, y));
            corners.Add(new SKPoint(last[y] + 1, y + 1));
        }
        var (rectAxis, width, elongated) = TipAxis(MinAreaRect(ConvexHull(corners)));

        // Principal axis of the pixel centres, pointed the same way as the rectangle's tip axis.
        double ax = rectAxis.X, ay = rectAxis.Y;
        if (elongated)
        {
            double mx = 0, my = 0;
            foreach (int i in pixels) { int y = i / w; mx += i - y * w + 0.5; my += y + 0.5; }
            mx /= pixels.Count; my /= pixels.Count;
            double sxx = 0, syy = 0, sxy = 0;
            foreach (int i in pixels)
            {
                int y = i / w;
                double dx = i - y * w + 0.5 - mx, dy = y + 0.5 - my;
                sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
            }
            double angle = 0.5 * Math.Atan2(2 * sxy, sxx - syy); // major-axis direction
            double px = Math.Cos(angle), py = Math.Sin(angle);
            if (px * ax + py * ay < 0) { px = -px; py = -py; }
            ax = px; ay = py;
        }

        // Farthest pixel along the axis; lateral centre of the pixels within half a width of it.
        double smax = double.MinValue;
        foreach (int i in pixels)
        {
            int y = i / w;
            smax = Math.Max(smax, (i - y * w + 0.5) * ax + (y + 0.5) * ay);
        }
        double depth = Math.Max(2.0, 0.5 * width), lat = 0;
        int n = 0;
        foreach (int i in pixels)
        {
            int y = i / w;
            double x = i - y * w + 0.5, yc = y + 0.5;
            if (x * ax + yc * ay < smax - depth) continue;
            lat += -ay * x + ax * yc;
            n++;
        }
        lat /= n;
        double along = smax + 0.5; // outer edge of the farthest pixel
        return new SKPoint((float)(box.Left + ax * along - ay * lat), (float)(box.Top + ay * along + ax * lat));
    }

    /// <summary>
    /// Picks the ends of a rectangle (corners in order): the short-side pair when elongated, else the ends of the more vertical axis.
    /// Returns the unit vector from the far end to the tip end (the lower one), the end-side length (= wire width), and whether elongated.
    /// </summary>
    private static (SKPoint Axis, float Width, bool Elongated) TipAxis(IReadOnlyList<SKPoint> c)
    {
        static SKPoint Mid(SKPoint a, SKPoint b) => new((a.X + b.X) / 2, (a.Y + b.Y) / 2);
        // Ends 0/2 (edges c0c1, c2c3) or ends 1/3 (edges c1c2, c3c0).
        float even = (SKPoint.Distance(c[0], c[1]) + SKPoint.Distance(c[2], c[3])) / 2;
        float odd = (SKPoint.Distance(c[1], c[2]) + SKPoint.Distance(c[3], c[0])) / 2;
        SKPoint evenAxis = Mid(c[2], c[3]) - Mid(c[0], c[1]), oddAxis = Mid(c[3], c[0]) - Mid(c[1], c[2]);
        float shortSide = Math.Min(even, odd), longSide = Math.Max(even, odd);
        bool elongated = longSide >= ElongatedRatio * Math.Max(shortSide, 1e-6f);
        static float Vertical(SKPoint d) => Math.Abs(d.Y) / Math.Max(d.Length, 1e-6f);
        bool useEven = elongated ? even <= odd : Vertical(evenAxis) >= Vertical(oddAxis);
        SKPoint axis = useEven ? evenAxis : oddAxis;
        if (axis.Y < 0) axis = new SKPoint(-axis.X, -axis.Y); // tip end = lower end
        float len = Math.Max(axis.Length, 1e-6f);
        return (new SKPoint(axis.X / len, axis.Y / len), useEven ? even : odd, elongated);
    }

    /// <summary>Largest 8-connected component of the mask, as pixel indices (y * w + x).</summary>
    private static List<int> LargestComponent(byte[] packedMask, int w, int h)
    {
        bool On(int i) => (packedMask[i >> 3] & (1 << (i & 7))) != 0;
        var seen = new bool[w * h];
        var best = new List<int>();
        var current = new List<int>();
        var stack = new Stack<int>();
        for (int start = 0; start < w * h; start++)
        {
            if (seen[start] || !On(start)) continue;
            current.Clear();
            seen[start] = true;
            stack.Push(start);
            while (stack.Count > 0)
            {
                int i = stack.Pop();
                current.Add(i);
                int y = i / w, x = i - y * w;
                for (int dy = -1; dy <= 1; dy++)
                {
                    int ny = y + dy;
                    if (ny < 0 || ny >= h) continue;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx;
                        if (nx < 0 || nx >= w) continue;
                        int j = ny * w + nx;
                        if (seen[j] || !On(j)) continue;
                        seen[j] = true;
                        stack.Push(j);
                    }
                }
            }
            if (current.Count > best.Count) (best, current) = (current, best);
        }
        return best;
    }

    /// <summary>Tip of an OBB model's rotated box (YoloDotNet: axis-aligned rect + angle in
    /// radians about its centre, same corner formula as its YOLO-format export).</summary>
    public static SKPoint FromObb(SKRectI rect, float angleRadians)
    {
        float cx = rect.MidX, cy = rect.MidY, hw = rect.Width / 2f, hh = rect.Height / 2f;
        float cos = MathF.Cos(angleRadians), sin = MathF.Sin(angleRadians);
        return BottomMid(new SKPoint[]
        {
            new(cx - hw * cos + hh * sin, cy - hw * sin - hh * cos),
            new(cx + hw * cos + hh * sin, cy + hw * sin - hh * cos),
            new(cx + hw * cos - hh * sin, cy + hw * sin + hh * cos),
            new(cx - hw * cos - hh * sin, cy - hw * sin + hh * cos),
        });
    }

    /// <summary>Midpoint of the lower of the two short edges of a rectangle (corners in order) — OBB models only;
    /// masks use the end-region tip above.</summary>
    public static SKPoint BottomMid(IReadOnlyList<SKPoint> c)
    {
        static float Len(SKPoint a, SKPoint b) => SKPoint.Distance(a, b);
        static SKPoint Mid(SKPoint a, SKPoint b) => new((a.X + b.X) / 2, (a.Y + b.Y) / 2);

        // Edge i runs corner i → i+1; edges 0/2 and 1/3 are opposite pairs.
        bool evenShort = Len(c[0], c[1]) + Len(c[2], c[3]) <= Len(c[1], c[2]) + Len(c[3], c[0]);
        SKPoint a = evenShort ? Mid(c[0], c[1]) : Mid(c[1], c[2]);
        SKPoint b = evenShort ? Mid(c[2], c[3]) : Mid(c[3], c[0]);
        return a.Y >= b.Y ? a : b;
    }

    // Andrew's monotone chain; returns the hull counter-clockwise (in y-down image space
    // the orientation does not matter for the calipers below).
    private static List<SKPoint> ConvexHull(List<SKPoint> pts)
    {
        pts.Sort((p, q) => p.X != q.X ? p.X.CompareTo(q.X) : p.Y.CompareTo(q.Y));
        static float Cross(SKPoint o, SKPoint a, SKPoint b) =>
            (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);

        var hull = new List<SKPoint>(pts.Count);
        for (int pass = 0; pass < 2; pass++)
        {
            int start = hull.Count;
            for (int k = 0; k < pts.Count; k++)
            {
                var p = pts[pass == 0 ? k : pts.Count - 1 - k];
                while (hull.Count >= start + 2 && Cross(hull[^2], hull[^1], p) <= 0)
                    hull.RemoveAt(hull.Count - 1);
                hull.Add(p);
            }
            hull.RemoveAt(hull.Count - 1);   // last point repeats as the next chain's first
        }
        return hull;
    }

    // Smallest-area rectangle with one side flush against a hull edge; corners in order.
    private static SKPoint[] MinAreaRect(List<SKPoint> hull)
    {
        if (hull.Count == 1)
            return new[] { hull[0], hull[0], hull[0], hull[0] };

        float bestArea = float.MaxValue;
        var best = new SKPoint[4];
        for (int i = 0; i < hull.Count; i++)
        {
            SKPoint e = hull[(i + 1) % hull.Count] - hull[i];
            float len = e.Length;
            if (len < 1e-6f)
                continue;
            var u = new SKPoint(e.X / len, e.Y / len);   // along the edge
            var v = new SKPoint(-u.Y, u.X);              // perpendicular

            float minU = float.MaxValue, maxU = float.MinValue, minV = float.MaxValue, maxV = float.MinValue;
            foreach (var p in hull)
            {
                float pu = p.X * u.X + p.Y * u.Y, pv = p.X * v.X + p.Y * v.Y;
                minU = Math.Min(minU, pu);
                maxU = Math.Max(maxU, pu);
                minV = Math.Min(minV, pv);
                maxV = Math.Max(maxV, pv);
            }

            float area = (maxU - minU) * (maxV - minV);
            if (area >= bestArea)
                continue;
            bestArea = area;
            SKPoint At(float a, float b) => new(u.X * a + v.X * b, u.Y * a + v.Y * b);
            best[0] = At(minU, minV);
            best[1] = At(maxU, minV);
            best[2] = At(maxU, maxV);
            best[3] = At(minU, maxV);
        }
        return best;
    }
}
