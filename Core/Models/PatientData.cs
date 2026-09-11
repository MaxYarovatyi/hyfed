using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Core.Models
{
    public sealed class PatientData
    {
        public string PatientId { get; init; } = string.Empty;
        public string Diagnosis { get; init; } = string.Empty;
        public string Trajectory { get; init; } = string.Empty;

        public double[] ConfidenceVector { get; init; } = [];

        public IReadOnlyList<BiometricSample> Samples { get; init; } = [];

        public IReadOnlyList<DeltaFeatureSet> DeltaFeatures { get; init; } = [];

        public int TotalSimulationDays { get; init; }
    }
}