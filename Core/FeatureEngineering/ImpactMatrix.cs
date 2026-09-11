using Core.Configuration;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
namespace Core.FeatureEngineering
{
    public sealed class ImpactMatrix
    {
        private readonly Matrix<double> _m;

        public IReadOnlyList<string> FeatureIds { get; }

        public IReadOnlyList<string> DiagnosisNames { get; }

        public int K => _m.RowCount;

        public int D => _m.ColumnCount;

        public ImpactMatrix(FeatureConfig config)
        {
            var matrixConfig = config.ImpactMatrix;

            var enabledIds = config.EnabledDeltaFeatures.Select(f => f.Id)
            .Concat(config.EnabledTransformFeatures.Select(f => f.Id))
            .ToHashSet();

            var orderedFeatureIds = matrixConfig.FeatureOrder
            .Where(enabledIds.Contains)
            .ToList();

            DiagnosisNames = matrixConfig.Diagnoses;
            FeatureIds = orderedFeatureIds;

            int k = DiagnosisNames.Count;
            int d = orderedFeatureIds.Count;

            var data = new double[k, d];
            for (int row = 0; row < k; row++)
            {
                string diagnosis = DiagnosisNames[row];
                double[] fullRow = matrixConfig.Rows[diagnosis];

                for (int col = 0; col < d; col++)
                {
                    string featureId = orderedFeatureIds[col];
                    int fullColIndex = Array.IndexOf(matrixConfig.FeatureOrder, featureId);
                    data[row, col] = fullRow[fullColIndex];
                }
            }

            _m = Matrix<double>.Build.DenseOfArray(data);
        }

        public Vector<double> ComputeFeatureWeights(double[] confidenceVector)
        {
            if (confidenceVector.Length != K)
                throw new ArgumentException($"Confidence vector lenght {confidenceVector.Length} != K={K}.");

            var c = Vector<double>.Build.DenseOfArray(confidenceVector);
            return c * _m;
        }

        public string FormatWeights(Vector<double> featureWeights)
        {
            var lines = FeatureIds
            .Select((id, i) => $" {id,-45} {featureWeights[i]:F3}")
            .ToList();

            return "Feature weights (w_feat = c * M):]\n" + string.Join("\n", lines);
        }
    }

}