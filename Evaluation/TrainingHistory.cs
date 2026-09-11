using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Core.Models;
using MathNet.Numerics.LinearAlgebra;

namespace Evaluation
{
    public sealed class TrainingHistory
    {
        private readonly List<RoundRecord> _rounds = [];
        public string ConfigName { get; init; } = string.Empty;
        public IReadOnlyList<RoundRecord> Rounds => _rounds;

        public void Record(int round, IReadOnlyList<PatientRoundData> patientData)
        {
            _rounds.Add(new RoundRecord
            {
                Round = round,
                MeanTrainLoss = patientData.Average(p => p.TrainLoss),
                StdTrainLoss = Std(patientData.Select(p => p.TrainLoss)),
                MeanDualityGap = patientData.Average(p => p.DualityGap),
                StdDualityGap = Std(patientData.Select(p => p.DualityGap)),
                WGlobalNorm = patientData[0].WGlobalNorm,
                PerPatient = [.. patientData]
            });
        }

        private static double Std(IEnumerable<double> values)
        {
            double mean = values.Average();
            return Math.Sqrt(values.Average(v => (v - mean) * (v - mean)));
        }

    }

    public sealed class RoundRecord
    {
        public int Round { get; init; }
        public double MeanTrainLoss { get; init; }
        public double StdTrainLoss { get; init; }

        public double MeanDualityGap { get; init; }
        public double StdDualityGap { get; init; }

        public double WGlobalNorm { get; init; }

        public IReadOnlyList<PatientRoundData> PerPatient { get; init; } = [];
    }

    public sealed class PatientRoundData
    {
        public string PatientId { get; init; } = string.Empty;
        public double TrainLoss { get; init; }
        public double DualityGap { get; init; }
        public double WLocalNorm { get; init; }
        public double WGlobalNorm { get; init; }
    }

    public static class DualityGapCalculator
    {
        public static double Compute(IReadOnlyList<TrainingSample> samples,
        Vector<double> wGlobal,
        Vector<double> wLocal,
        Vector<double> alpha,
        double lambdaGlobal,
        double lambdaLocal)
        {
            int n = samples.Count;

            // Primal objective
            double primalLoss = 0.0;
            foreach (var s in samples)
            {
                var xG = s.Features.SubVector(0, wGlobal.Count);
                var xL = s.Features.SubVector(wGlobal.Count, wLocal.Count);
                double pred = wGlobal.DotProduct(xG) + wLocal.DotProduct(xL);
                double err = pred - s.Label;
                primalLoss += 0.5 * err * err;
            }


            double regGlobal = (lambdaGlobal / 2.0) * wGlobal.DotProduct(wGlobal);
            double regLocal = (lambdaLocal / 2.0) * wLocal.DotProduct(wLocal);
            double primal = primalLoss / n + regGlobal + regLocal;

            double alphaY = 0;
            var wG_rec = Vector<double>.Build.Dense(wGlobal.Count);
            var wL_rec = Vector<double>.Build.Dense(wLocal.Count);

            for (int i = 0; i < n; i++)
            {
                var s = samples[i];
                double a = alpha[i];
                alphaY += a * s.Label;

                wG_rec += a * s.Features.SubVector(0, wGlobal.Count);
                wL_rec += a * s.Features.SubVector(wGlobal.Count, wLocal.Count);
            }

            double dualTerm_G = wG_rec.DotProduct(wG_rec) / (2.0 * lambdaGlobal * n * n);
            double dualTerm_L = wL_rec.DotProduct(wL_rec) / (2.0 * lambdaLocal * n * n);
            double alphaNorm = alpha.DotProduct(alpha) / (2.0 * n);

            double dual = (alphaY / n) - dualTerm_G - dualTerm_L - alphaNorm;

            return Math.Max(0.0, primal - dual);

            // primal = primal / n + (lambda / 2.0) * w.DotProduct(w);

            // double alphaY = samples.Select((s, i) => alpha[i] * s.Label).Sum();

            // var wReconstruct = Vector<double>.Build.Dense(w.Count);

            // for (int i = 0; i < n; i++)
            //     wReconstruct += alpha[i] * samples[i].Features.SubVector(0, w.Count);

            // double alphaNormSq = alpha.DotProduct(alpha);
            // double dual = alphaY / n - wReconstruct.DotProduct(wReconstruct) / (2.0 * lambda * n * n) - alphaNormSq / (2.0 * n);

            // return Math.Max(0.0, primal - dual);
        }
    }
}