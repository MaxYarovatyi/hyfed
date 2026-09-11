using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Statistics;

namespace Evaluation
{
    public record PatientEvaluationResult
    {
        public string PatientId { get; init; } = string.Empty;
        public string Diagnosis { get; init; } = string.Empty;
        public string Trajectory { get; init; } = string.Empty;

        public PatientMetrics Train { get; init; } = new();
        public PatientMetrics Test { get; init; } = new();

        public double LabelMean { get; init; }
        public double LabelStd { get; init; }
        public double WLocalNorm { get; init; }
        public int TrainCount { get; init; }
        public int TestCount { get; init; }
    }

    public record RoundSummary
    {
        public int Round { get; init; }
        public double MeanDualityGap { get; init; }
        public double StdDualityGap { get; init; }
        public double MeanTrainLoss { get; init; }
        public double StdTrainLoss { get; init; }
        public double WGlobalNorm { get; init; }
        public double MeanDeltaNorm { get; init; }
    }
    public static class ResultExporter
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        /// patient_metrics.csv
        public static void WritePatientMetrics(IEnumerable<PatientEvaluationResult> results, string path)
        {
            var sb = new StringBuilder();

            sb.AppendLine("PatientId,Diagnosis,Trajectory," +
            "TrainR2,TestR2,TrainMAE,TestMAE,TrainMSE,TestMSE,TrainRMSE,TestRMSE," +
            "LabelMean,LabelStd,WLocalNorm,TrainCount,TestCount");

            foreach (var r in results)
            {
                sb.AppendLine(string.Join(',',
                r.PatientId, r.Diagnosis, r.Trajectory,
                F(r.Train.R2), F(r.Test.R2), F(r.Train.MAE), F(r.Test.MAE), F(r.Train.MSE), F(r.Test.MSE), F(r.Train.RMSE), F(r.Test.RMSE),
                F(r.LabelMean), F(r.LabelStd), F(r.WLocalNorm), r.TrainCount.ToString(Inv), r.TestCount.ToString(Inv)));
            }

            Write(path, sb);
        }

        /// convergence.csv
        public static void WriteConvergence(IEnumerable<RoundSummary> rounds, string path)
        {
            var sb = new StringBuilder();

            sb.AppendLine("Round,MeanDualityGap,StdDUalityGap," +
            "MeanTrainLoss,StdTrainLoss,WGlobalNorm,MeanDeltaNorm");

            foreach (var r in rounds)
            {
                sb.AppendLine(string.Join(',',
                r.Round.ToString(Inv), F(r.MeanDualityGap), F(r.StdDualityGap),
                F(r.MeanTrainLoss), F(r.StdTrainLoss), F(r.WGlobalNorm), F(r.MeanDeltaNorm)));
            }

            Write(path, sb);
        }

        public static void WriteAblationSummary(IEnumerable<AblationRunResult> runs, string path)
        {
            var sb = new StringBuilder();

            sb.AppendLine("Config," +
            "MeanTestR2,StdTestR2,MeanTestMAE,StdTestMAE,MeanTestMSE,StdTestMSE," +
            "MeanTrainR2,StdTrainR2,MeanTrainMAE,StdTrainMAE");

            foreach (var r in runs)
            {
                sb.AppendLine(string.Join(',',
                r.ConfigName,
                F(r.MeanTestR2), F(r.StdTestR2),
                F(r.MeanTestMAE), F(r.StdTestMAE),
                F(r.MeanTestMSE), F(r.StdTestMSE),
                F(r.MeanTrainR2), F(r.StdTrainR2),
                F(r.MeanTrainMAE), F(r.StdTrainMAE)));
            }

            Write(path, sb);
        }

        // Converts PatientRoundData list to RoundSummary list
        // Call after training loop, pass the full history per round
        public static List<RoundSummary> BuildRoundSummaries(IReadOnlyList<RoundRecord> allRounds)
        {
            return allRounds.Select(r =>
            {
                return new RoundSummary
                {
                    Round = r.Round,
                    MeanTrainLoss = r.MeanTrainLoss,
                    StdTrainLoss = r.StdTrainLoss,
                    MeanDualityGap = r.MeanDualityGap,
                    StdDualityGap = r.StdDualityGap,
                    WGlobalNorm = r.WGlobalNorm,
                    MeanDeltaNorm = r.PerPatient.Average(p => p.WLocalNorm)
                };
            }).ToList();
        }


        // Private helpers

        private static void Write(string path, StringBuilder sb)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
            Console.WriteLine($"[Exporter] Written: {path}");
        }

        private static string F(double v) =>
            double.IsNaN(v) ? "Nan" : v.ToString("G8", Inv);


    }
    // Aggregate result for one ablation config — populate from your AblationStudy
    public record AblationRunResult
    {
        public string ConfigName { get; init; } = string.Empty;
        public double MeanTestR2 { get; init; }
        public double StdTestR2 { get; init; }
        public double MeanTestMAE { get; init; }
        public double StdTestMAE { get; init; }
        public double MeanTestMSE { get; init; }
        public double StdTestMSE { get; init; }
        public double MeanTestRMSE { get; init; }
        public double StdTestRMSE { get; init; }
        public double MeanTrainR2 { get; init; }
        public double StdTrainR2 { get; init; }
        public double MeanTrainMAE { get; init; }
        public double StdTrainMAE { get; init; }
        public double MeanTrainRMSE { get; init; }
        public double StdTrainRMSE { get; init; }
    }
}