using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Core.Models;

namespace Evaluation
{
    public sealed class AblationStudy
    {
        private readonly IReadOnlyDictionary<string, IReadOnlyList<TrainingSample>> _allSamples;

        public AblationStudy(IReadOnlyDictionary<string, IReadOnlyList<TrainingSample>> allSamples)
        {
            _allSamples = allSamples;
        }

        // Public API

        public IReadOnlyList<EvaluationResult> RunAll()
        {
            var configs = BuildAllConfigs().ToList();
            var results = new List<EvaluationResult>();


            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine($"  Ablation Study  |  {configs.Count} configs  |  {_allSamples.Count} patients");
            Console.WriteLine("═══════════════════════════════════════════════════════");

            for (int i = 0; i < configs.Count; i++)
            {
                var config = configs[i];
                Console.WriteLine($"\n[{i + 1}]/{configs.Count} {config.Name} ... ");

                var evaluator = new ModelEvaluator(config);
                var result = evaluator.Evaluate(_allSamples);
                results.Add(result);

                Console.WriteLine($"R²={result.Aggregated.R2_Mean:F4} MAE={result.Aggregated.MAE_Mean:F4}");
            }

            // PrintComparison(results);
            return results;
        }


        /// <summary>
        /// Runs a single named configuration
        /// </summary>
        public EvaluationResult RunSingle(AblationConfig config)
        {
            Console.WriteLine($"Running: {config.Name}");
            var evaluator = new ModelEvaluator(config);
            var result = evaluator.Evaluate(_allSamples);
            // result.PrintSummary();
            return result;
        }

        // Config definitions

        private static IEnumerable<AblationConfig> BuildAllConfigs()
        {
            // Model variant comparison
            yield return AblationConfig.GlobalOnly;
            yield return AblationConfig.LocalOnly;
            yield return AblationConfig.Baseline;
            yield return AblationConfig.NoImpactMatrix;

            foreach (var c in AblationConfig.HSweep([1, 3, 5]))
                yield return c;

            foreach (var c in AblationConfig.LambdaLocalSweep([0.01, 0.05, 0.1, 0.5]))
                yield return c;

            foreach (var c in AblationConfig.DpSweep([1.0, 5.0, 10.0]))
                yield return c;
        }

        private static void PrintComparison(IReadOnlyList<EvaluationResult> results)
        {
            Console.WriteLine("\n\n═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("  ABLATION STUDY RESULTS");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine($"  {"Config",-35} {"MSE",10} {"MAE",10} {"R²",10}");
            Console.WriteLine(new string('─', 70));

            // Group by category
            var groups = new (string Label, string[] Prefixes)[]
            {
            ("Model variants",     ["GlobalOnly", "LocalOnly", "HyFDCA_baseline", "HyFDCA_no_impact"]),
            ("H sweep",            ["HyFDCA_H"]),
            ("λ_local sweep",      ["HyFDCA_lambda"]),
            ("DP sweep",           ["HyFDCA_eps"]),
            };

            foreach (var (label, prefixes) in groups)
            {
                var group = results
                    .Where(r => prefixes.Any(p => r.ConfigName.StartsWith(p)))
                    .ToList();

                if (group.Count == 0) continue;

                Console.WriteLine($"\n  ── {label}");
                foreach (var r in group)
                {
                    var m = r.Aggregated;
                    Console.WriteLine(
                        $"  {r.ConfigName,-35} " +
                        $"{m.MSE_Mean,8:F4}±{m.MSE_Std:F4}  " +
                        $"{m.MAE_Mean,8:F4}±{m.MAE_Std:F4}  " +
                        $"{m.R2_Mean,8:F4}±{m.R2_Std:F4}");
                }
            }

            Console.WriteLine("\n═══════════════════════════════════════════════════════════════════");
        }
    }
}