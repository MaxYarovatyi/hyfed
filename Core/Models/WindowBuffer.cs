using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Core.Models
{
    /// <summary>
    /// Sliding window buffer that computes temporal delta-features from a stream
    /// of daily BiometricSamples for one patient.
    ///
    /// Window size = 14 days (covers Slope14 lookback).
    /// Gap days are never enqueued — the DayIndex gap is handled naturally
    /// by the lookback range checks in GetValid().
    ///
    /// Usage: one WindowBuffer instance per patient, call Enqueue() day by day.
    /// </summary>

    public sealed class WindowBuffer
    {

        private readonly Queue<BiometricSample> _window;
        private const int MaxSize = 14;
        private const int MinDaysMean = 3;
        private const int MinDaysSlope = 5;
        private const int MinDaysStd = 4;

        public WindowBuffer()
        {
            _window = new Queue<BiometricSample>(MaxSize);
        }

        /// <summary>
        /// Adds a new sample and returns computed delta-features for that day.
        /// Some features may be null if there is insufficient history.
        /// </summary>
        public DeltaFeatureSet Enqueue(BiometricSample sample)
        {
            if (_window.Count == MaxSize)
                _window.Dequeue();
            _window.Enqueue(sample);

            return new DeltaFeatureSet
            {
                PatientId = sample.PatientId,
                DayIndex = sample.DayIndex,
                Date = sample.Date,
                CriticalityScore = sample.CriticalityScore,

                HRV_DeltaMean7 = DeltaMean7(r => r.HRV),
                HRV_Slope14 = Slope14(r => r.HRV),

                Sleep_DeltaMean7 = DeltaMean7(r => r.SleepQuality),
                Sleep_Slope14 = Slope14(r => r.SleepQuality),

                Steps_DeltaMean7 = DeltaMean7(r => r.Steps),
                Steps_Slope14 = Slope14(r => r.Steps),
                Steps_Std7 = Std7(r => r.Steps),

                HR_Std7 = Std7(r => r.HR),
            };
        }

        //Private helpers
        private List<(int dayIndex, double value)> GetValid(Func<BiometricSample, double> selector, int lookback)
        {
            var current = _window.Last();
            return _window.Where(r => r.DayIndex < current.DayIndex && current.DayIndex - r.DayIndex <= lookback)
            .Select(r => (r.DayIndex, selector(r)))
            .ToList();
        }

        private double? DeltaMean7(Func<BiometricSample, double> selector)
        {
            var current = _window.Last();
            var points = GetValid(selector, 7);

            if (points.Count < MinDaysMean) return null;

            return selector(current) - points.Average(p => p.value);
        }

        private double? Slope14(Func<BiometricSample, double> selector)
        {
            var current = _window.Last();
            var history = GetValid(selector, 14);
            var all = history.Append((current.DayIndex, selector(current)))
            .ToList();

            if (all.Count < MinDaysSlope) return null;

            return LeastSquaresSlope(all);
        }

        private double? Std7(Func<BiometricSample, double> selector)
        {
            var current = _window.Last();
            var history = GetValid(selector, 7);
            var all = history.Append((current.DayIndex, selector(current)))
            .Select(p => p.Item2)
            .ToList();

            if (all.Count < MinDaysStd) return null;

            var mean = all.Average();
            var variance = all.Average(v => Math.Pow(v - mean, 2));
            return Math.Sqrt(variance);
        }

        private static double LeastSquaresSlope(List<(int dayIndex, double value)> points)
        {
            int n = points.Count;
            double sumI = points.Sum(p => (double)p.dayIndex);
            double sumX = points.Sum(p => p.value);
            double sumIX = points.Sum(p => p.dayIndex * p.value);
            double sumI2 = points.Sum(p => (double)(p.dayIndex * p.dayIndex));

            double denom = n * sumI2 - sumI * sumI;
            if (Math.Abs(denom) < 1e-9)
                return 0.0;

            return (n * sumIX - sumI * sumX) / denom;
        }
    }
}