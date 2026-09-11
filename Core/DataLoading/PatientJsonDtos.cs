using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Core.DataLoading
{
    internal sealed class PatientJsonDto
    {
        [JsonPropertyName("patientId")]
        public string PatientId { get; init; } = string.Empty;
        [JsonPropertyName("primaryDiagnosis")]
        public string PrimaryDiagnosis { get; init; } = string.Empty;
        [JsonPropertyName("trajectory")]
        public string Trajectory { get; init; } = string.Empty;
        [JsonPropertyName("confidenceVector")]
        public double[] ConfidenceVector { get; init; } = [];
        [JsonPropertyName("records")]
        public RecordJsonDto[] Records { get; init; } = [];
    }

    internal sealed class RecordJsonDto
    {
        [JsonPropertyName("dayIndex")]
        public int DayIndex { get; init; }
        [JsonPropertyName("date")]
        public DateTime Date { get; init; }
        [JsonPropertyName("hasBiometricGap")]
        public bool HasBiometricGap { get; init; }
        [JsonPropertyName("hrv")]
        public double? HRV { get; init; }
        [JsonPropertyName("heartRateAvg")]
        public double? HeartRateAvg { get; init; }
        [JsonPropertyName("steps")]
        public double? Steps { get; init; }
        [JsonPropertyName("sleepDuration")]
        public double? SleepDuration { get; init; }
        [JsonPropertyName("sleepQuality")]
        public double? SleepQuality { get; init; }
        [JsonPropertyName("activityLevel")]
        public double? ActivityLevel { get; init; }
        [JsonPropertyName("criticalityScore")]
        public double? CriticalityScore { get; init; }

    }
}