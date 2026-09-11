using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Core.Models;
using MathNet.Numerics.Statistics;

namespace Evaluation
{
    public sealed class ModelEvaluator
    {
        private readonly AblationConfig _config;

        public ModelEvaluator(AblationConfig config)
        {
            _config = config;
        }

        // Public API

        public EvaluationResult Evaluate(IReadOnlyDictionary<string, IReadOnlyList<TrainingSample>> allSamples)
        {
            // Split each patients data
            var splits = allSamples.ToDictionary(
                kv => kv.Key,
                kv => MetricsCalculator.Split(kv.Value, _config.TrainFraction)
            );

            var valid = splits
                .Where(kv => kv.Value.Train.Count > 0 && kv.Value.Test.Count > 0)
                .ToDictionary(kv => kv.Key, kv => kv.Value);

            var trainData = valid.Select(kv => new PatientTrainData
            {
                PatientId = kv.Key,
                TrainSamples = kv.Value.Train
            }).ToList();

            var history = new TrainingHistory { ConfigName = _config.Name };

            // Train
            var runner = new InProcessFLRunner(_config);
            var modelStates = runner.Run(trainData, history).ToDictionary(s => s.PatientId);

            //Evaluate on test split per patient
            var perPatient = new List<(string PatientId, PatientMetrics TrainMetrics, PatientMetrics TestMetrics)>();


            foreach (var (patientId, split) in valid)
            {
                if (!modelStates.TryGetValue(patientId, out var state))
                    continue;

                var metrics = _config.Variant switch
                {
                    ModelVariant.HyFDCA =>
                        (
                        MetricsCalculator.Evaluate(split.Train, state.WGlobal, state.WLocal),
                        MetricsCalculator.Evaluate(split.Test, state.WGlobal, state.WLocal)),

                    ModelVariant.GlobalOnly => (
                        MetricsCalculator.EvaluateGlobalOnly(split.Train, state.WGlobal),
                        MetricsCalculator.EvaluateGlobalOnly(split.Test, state.WGlobal)),

                    ModelVariant.LocalOnly => (
                        MetricsCalculator.EvaluateLocalOnly(split.Train, state.WLocal),
                        MetricsCalculator.EvaluateLocalOnly(split.Test, state.WLocal)),

                    _ => throw new ArgumentOutOfRangeException()
                };
                perPatient.Add((patientId, metrics.Item1, metrics.Item2));
            }


            var aggregatedTest = AggregatorMetrics.From(_config.Name, perPatient.Select(p => p.TestMetrics).ToList());

            var aggregatedTrain = AggregatorMetrics.From(_config.Name, perPatient.Select(p => p.TrainMetrics).ToList());
            return new EvaluationResult
            {
                ConfigName = _config.Name,
                Variant = _config.Variant,
                PerPatient = perPatient.Select(p => new PatientResult
                {
                    PatientId = p.PatientId,
                    Metrics = p.TestMetrics,
                    TrainMetrics = p.TrainMetrics,
                    TestCount = p.TestMetrics.N,
                    TrainCount = p.TrainMetrics.N,
                    WLocalNorm = modelStates[p.PatientId].WLocal?.L2Norm() ?? 0.0,
                }).ToList(),
                Aggregated = aggregatedTest,
                AggregatedTrain = aggregatedTrain,
                History = history
            };
        }
    }

    // Result types

    public sealed class PatientResult
    {
        public string PatientId { get; init; } = string.Empty;
        public PatientMetrics Metrics { get; init; } = null!;
        public PatientMetrics TrainMetrics { get; init; } = null!;
        public double WLocalNorm { get; init; }
        public int TrainCount { get; init; }
        public int TestCount { get; init; }
    }

    public sealed class EvaluationResult
    {
        public string ConfigName { get; init; } = string.Empty;
        public ModelVariant Variant { get; init; }
        public IReadOnlyList<PatientResult> PerPatient { get; init; } = [];
        public AggregatorMetrics Aggregated { get; init; } = null!;
        public AggregatorMetrics AggregatedTrain { get; init; } = null!;
        public TrainingHistory History { get; init; } = null!;

        public void PrintSummary()
        {
            Console.WriteLine($"\n{'–',50}");
            Console.WriteLine(Aggregated.ToString());
            Console.WriteLine($"Per-patient R2:");
            foreach (var p in PerPatient.OrderBy(p => p.PatientId))
                Console.WriteLine($"{p.PatientId,-15} R2={p.Metrics.R2:F4} MAE={p.Metrics.MAE:F4}");
        }
    }
}