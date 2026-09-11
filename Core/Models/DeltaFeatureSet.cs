using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Core.Models
{
    public sealed class DeltaFeatureSet
    {
        public string PatientId { get; init; } = string.Empty;
        public int DayIndex { get; init; }
        public DateOnly Date { get; init; }

        public double CriticalityScore { get; init; }

        // ── HRV features ─────────────────────────────────────────────────────────
        public double? HRV_DeltaMean7 { get; init; }   // deviation from 7-day mean
        public double? HRV_Slope14 { get; init; }   // trend over 14 days

        // ── Sleep features ────────────────────────────────────────────────────────
        public double? Sleep_DeltaMean7 { get; init; }
        public double? Sleep_Slope14 { get; init; }

        // ── Steps features ────────────────────────────────────────────────────────
        public double? Steps_DeltaMean7 { get; init; }
        public double? Steps_Slope14 { get; init; }
        public double? Steps_Std7 { get; init; }

        // ── HR features ───────────────────────────────────────────────────────────
        public double? HR_Std7 { get; init; }          // volatility — relevant for anxiety/bipolar

        public int ValidFeatureCount =>
            (HRV_DeltaMean7.HasValue ? 1 : 0) +
            (HRV_Slope14.HasValue ? 1 : 0) +
            (Sleep_DeltaMean7.HasValue ? 1 : 0) +
            (Sleep_Slope14.HasValue ? 1 : 0) +
            (Steps_DeltaMean7.HasValue ? 1 : 0) +
            (Steps_Slope14.HasValue ? 1 : 0) +
            (Steps_Std7.HasValue ? 1 : 0) +
            (HR_Std7.HasValue ? 1 : 0);

        public const int TotalFeatureCount = 8;
    }
}