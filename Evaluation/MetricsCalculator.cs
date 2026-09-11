using MathNet.Numerics.LinearAlgebra;
using Core.Models;
using System.ComponentModel.DataAnnotations;
using MathNet.Numerics.Statistics;

namespace Evaluation
{
    public static class MetricsCalculator
    {
        private const int GlobalDim = 6;

        // Public API

        public static PatientMetrics Evaluate(IReadOnlyList<TrainingSample> testSamples, Vector<double> wGlobal, Vector<double> wLocal)
        {
            var predictions = testSamples.Select(s =>
            {
                int dLocal = wLocal.Count;
                var xGlobal = s.Features.SubVector(0, GlobalDim);
                var xLocal = s.Features.SubVector(GlobalDim, dLocal);
                return wGlobal.DotProduct(xGlobal) + wLocal.DotProduct(xLocal) + s.LabelMean;
            }).ToList();

            return Compute(testSamples, predictions);
        }

        public static PatientMetrics EvaluateGlobalOnly(IReadOnlyList<TrainingSample> testSamples, Vector<double> wGlobal)
        {
            var predictions = testSamples.Select(s =>
            {
                var xGlobal = s.Features.SubVector(0, GlobalDim);
                return wGlobal.DotProduct(xGlobal) + s.LabelMean;
            }).ToList();

            return Compute(testSamples, predictions);
        }

        public static PatientMetrics EvaluateLocalOnly(IReadOnlyList<TrainingSample> testSampes, Vector<double> wLocal)
        {
            var predictions = testSampes.Select(s =>

                wLocal.DotProduct(s.Features.SubVector(GlobalDim, s.Features.Count - GlobalDim)) + s.LabelMean
            ).ToList();

            return Compute(testSampes, predictions);
        }

        // Train/test split

        /// <summary>
        /// Splits samples by time (not random) - preserves temporal order
        /// </summary>
        public static (List<TrainingSample> Train, List<TrainingSample> Test) Split(
            IReadOnlyList<TrainingSample> samples, double trainFraction = 0.8)
        {
            int trainCount = (int)Math.Floor(samples.Count * trainFraction);
            var ordered = samples.OrderBy(s => s.DayIndex).ToList();

            return (ordered.Take(trainCount).ToList(), ordered.Skip(trainCount).ToList());
        }

        // Core metrics

        private static PatientMetrics Compute(IReadOnlyList<TrainingSample> samples, IReadOnlyList<double> predictions)
        {
            double[] labels = samples.Select(s => s.Label + s.LabelMean).ToArray();



            double[] preds = predictions.ToArray();
            int n = labels.Length;

            double mse = labels.Zip(preds, (y, yh) => (y - yh) * (y - yh)).Average();
            double mae = labels.Zip(preds, (y, yh) => Math.Abs(y - yh)).Average();
            double r2 = ComputeR2(labels, preds);
            double rmse = Math.Sqrt(mse);

            return new PatientMetrics
            {
                N = n,
                MSE = mse,
                RMSE = rmse,
                MAE = mae,
                R2 = r2
            };
        }

        private static double ComputeR2(double[] labels, double[] preds)
        {
            double mean = labels.Average();
            double ssTot = labels.Sum(y => Math.Pow(y - mean, 2));
            double ssRes = labels.Zip(preds, (y, yh) => (y - yh) * (y - yh)).Sum();
            return ssTot < 1e-3 ? 0.0 : 1.0 - ssRes / ssTot;
        }


    }

    // Result types

    public sealed class PatientMetrics
    {
        public int N { get; init; }
        public double MSE { get; init; }
        public double RMSE { get; init; }
        public double MAE { get; init; }
        public double R2 { get; init; }

        public override string ToString() =>
            $"MSE={MSE:F4} RMSE={RMSE:F4} MAE={MAE:F4} R2={R2:F4} (n={N})";
    }

    public sealed class AggregatorMetrics
    {
        public string ConfigName { get; init; } = string.Empty;
        public int PatientCount { get; init; }

        public double MSE_Mean { get; init; }
        public double MSE_Std { get; init; }
        public double RMSE_Mean { get; init; }
        public double RMSE_Std { get; init; }
        public double MAE_Mean { get; init; }
        public double MAE_Std { get; init; }
        public double R2_Mean { get; init; }
        public double R2_Std { get; init; }

        public static AggregatorMetrics From(string configName, IReadOnlyList<PatientMetrics> perPatient)
        {
            double Mean(Func<PatientMetrics, double> sel) =>
                perPatient.Average(sel);

            double Std(Func<PatientMetrics, double> sel)
            {
                double mean = Mean(sel);
                return Math.Sqrt(perPatient.Select(sel).Average(v => (v - mean) * (v - mean)));
            }

            var validR2 = perPatient.Where(p => p.R2 != 0.0).Select(p => p.R2).ToList();

            return new AggregatorMetrics
            {
                ConfigName = configName,
                PatientCount = perPatient.Count,
                MSE_Mean = Mean(m => m.MSE),
                MSE_Std = Std(m => m.MSE),
                RMSE_Mean = Mean(m => m.RMSE),
                RMSE_Std = Std(m => m.RMSE),
                MAE_Mean = Mean(m => m.MAE),
                MAE_Std = Std(m => m.MAE),
                R2_Mean = Statistics.Mean(validR2),
                R2_Std = Statistics.StandardDeviation(validR2)
            };
        }

        public override string ToString() =>
            $"{ConfigName,-35} | " +
            $"MSE={MAE_Mean:F4}±{MSE_Std:F4} | " +
            $"MAE={MAE_Mean:F4}±{MAE_Std:F4} | " +
            $"R2={R2_Mean:F4}±{R2_Std:F4} |" +
            $"n={PatientCount}";
    }
}