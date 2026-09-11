using MathNet.Numerics.LinearAlgebra;

namespace Core.Models
{
    public sealed class TrainingSample
    {
        public string PatientId { get; init; } = string.Empty;
        public int DayIndex { get; init; }
        public DateOnly Date { get; init; }

        public Vector<double> Features { get; init; } = null!;

        public double Label { get; init; }
        public double LabelMean { get; init; }

        public double SampleWeight { get; init; }

        public int FeatureDimension => Features.Count;

    }
}