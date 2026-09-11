

using MathNet.Numerics.LinearAlgebra;
using Core.Algorithm;
using MathNet.Numerics.Statistics;
using Core.Models;

namespace Tests
{
    public class SolverSmokeTest
    {
        private const int GlobalDim = 6;
        private const int LocalDim = 13;
        private const int TotalDim = GlobalDim + LocalDim;

        public static void Run()
        {
            Console.WriteLine("═══════════════════════════════════════════════════");
            Console.WriteLine("  HyFDCA Solver Smoke Test");
            Console.WriteLine("═══════════════════════════════════════════════════\n");

            // RunSolverTest();
            // RunGaussianMechanismTest();
            RunAggregatorTest();
        }

        private static void RunSolverTest()
        {
            Console.WriteLine("── Test 1: LocalSdcaSolver convergence ─────────────\n");

            const int n = 120;   // synthetic samples (~1 patient, ~4 months)
            const int rounds = 10;    // FL rounds to simulate
            var rng = new Random(42);

            var samples = GenerateSamples(n, rng);

            var config = new SdcaConfig
            {
                LambdaGlobal = 0.01,
                LambdaLocal = 0.05,
                EpochsPerRound = 1
            };

            var solver = new LocalSdcaSolver(config, seed: 42);

            var wGlobal = Vector<double>.Build.Dense(GlobalDim);
            var state = DualVariableState.Zero(n, LocalDim);

            Console.WriteLine($"{"Round",-8} {"Loss",-12} {"w[0] HRV",-12} {"w[3] Sleep",-12}");
            Console.WriteLine(new string('─', 48));

            double prevLoss = double.MaxValue;
            bool lossDecreasing = true;

            for (int round = 0; round < rounds; round++)
            {
                var result = solver.RunRound(samples, wGlobal, state);

                wGlobal += result.DeltaWGlobal;

                state = result.UpdatedState;

                Console.WriteLine($"{round + 1,-8} {result.TrainingLoss,-12:F6} " +
                $"{wGlobal[0],-12:F4} {wGlobal[3],-12:F4}");

                if (result.TrainingLoss > prevLoss + 1e-6)
                    lossDecreasing = false;

                prevLoss = result.TrainingLoss;
            }


            Console.WriteLine();
            Console.WriteLine($"Final w_global:  [{string.Join(", ", wGlobal.Select(v => v.ToString("F3")))}]");
            Console.WriteLine($"Expected approx: [0.5, 0.0, 0.0, 0.3, 0.0, 0.0]");
            Console.WriteLine($"Final w_local norm: {state.WLocal.L2Norm():F4}  (expected ≈ 0)");
            Console.WriteLine();

            // Assertions
            Assert("Loss is decreasing", lossDecreasing);
            Assert("w[0] converges toward 0.5", Math.Abs(wGlobal[0] - 0.5) < 0.15);
            Assert("w[3] converges toward 0.3", Math.Abs(wGlobal[3] - 0.3) < 0.15);
            Assert("Final loss < initial", prevLoss < 0.1);
        }

        private static void RunAggregatorTest()
        {
            Console.WriteLine("\n── Test 3: GlobalAggregator FedAvg ─────────────────\n");

            const int clients = 5;
            const int rounds = 15;

            var rng = new Random(42);

            var aggregatorConfig = new AggregatorConfig
            {
                GlobalFeatureDim = GlobalDim,
                ConvergenceTolerance = 1e-4,
                ConvergenceRoundsRequired = 3
            };

            var sdcaConfig = new SdcaConfig
            {
                LambdaGlobal = 0.01,
                LambdaLocal = 0.5,
                EpochsPerRound = 1,
            };

            var aggregator = new GlobalAggregator(aggregatorConfig);
            var mechanism = new GaussianMechanism(seed: 42);
            var budget = new DpBudget { Epsilon = 1.0, Delta = 1e-5 };

            var clientSamples = Enumerable.Range(0, clients)
            .Select(k => GenerateSamples(100 + k * 10, rng))
            .ToList();

            var solvers = Enumerable.Range(0, clients)
            .Select(k => new LocalSdcaSolver(sdcaConfig, seed: k))
            .ToList();

            var states = clientSamples
            .Select(s => DualVariableState.Zero(s.Count, LocalDim))
            .ToList();

            Console.WriteLine($"{"Round",-8} {"‖Δw‖",-12} {"w[0]",-10} {"w[3]",-10} {"Converged"}");
            Console.WriteLine(new string('─', 55));

            AggregationResult latestResult = null!;

            for (int round = 0; round < rounds; round++)
            {
                var updates = new List<ClientUpdate>();

                for (int k = 0; k < clients; k++)
                {
                    var result = solvers[k].RunRound(
                        clientSamples[k],
                        aggregator.WGlobal,
                        states[k]
                    );


                    updates.Add(new ClientUpdate
                    {
                        ClientId = $"client_{k}",
                        DeltaWGlobal = result.DeltaWGlobal,
                        SampleCount = clientSamples[k].Count
                    });

                    states[k] = result.UpdatedState;
                }

                latestResult = aggregator.Aggregate(updates);
                var w = latestResult.NewWGlobal;
                Console.WriteLine($"{latestResult.Round,-8} {latestResult.UpdateNorm,-12:F6} " +
                                              $"{w[0],-10:F4} {w[3],-10:F4} " +
                                              $"{(latestResult.IsConverged ? "✅" : "")}");

                if (latestResult.IsConverged) break;

            }
            var wFinal = latestResult.NewWGlobal;
            Console.WriteLine();
            Console.WriteLine($"Final w_global: [{string.Join(", ", wFinal.Select(v => v.ToString("F3")))}]");
            Console.WriteLine($"Expected approx: [0.5, 0.0, 0.0, 0.4, 0.0, 0.0]");
            Console.WriteLine();

            Assert("w[0] converges toward 0.5", Math.Abs(wFinal[0] - 0.5) < 0.06);
            Assert("w[3] converges toward 0.3", Math.Abs(wFinal[3] - 0.3) < 0.05);
            Assert("w[1] stays near 0", Math.Abs(wFinal[1]) < 0.05);
        }

        private static void RunGaussianMechanismTest()
        {
            Console.WriteLine("\n── Test 2: GaussianMechanism statistics ────────────\n");

            const int n = 120;
            const double lambda = 0.01;
            const double epsilon = 1.0;
            const double delta = 1e-5;
            const int repetitions = 10_000;
            const double clippingNorm = 0.5;

            var mechanism = new GaussianMechanism(seed: 42);
            var budget = new DpBudget { Epsilon = epsilon, Delta = delta };


            var deltaW = Vector<double>.Build.Dense(n, 0.0);

            double expectedSensitivity = GaussianMechanism.ComputeSensitivity(lambda, n);
            double expectedSigma = GaussianMechanism.ComputeSigma(
                expectedSensitivity, epsilon, delta);

            Console.WriteLine($"C={clippingNorm}, n={n}, ε={epsilon}, δ={delta}");
            Console.WriteLine($"Expected σ:           {expectedSigma:F4}");
            Console.WriteLine($"Repetitions:          {repetitions:N0}\n");

            // Collect noise samples from coordinate 0 across many runs
            var noiseSamples = new double[repetitions];
            for (int i = 0; i < repetitions; i++)
            {
                var result = mechanism.Privatize(deltaW, budget, clippingNorm);
                noiseSamples[i] = result.NoisedVector[0];   // coordinate 0, input was 0 → pure noise
            }

            double empiricalMean = Statistics.Mean(noiseSamples);
            double empiricalStd = Statistics.StandardDeviation(noiseSamples);

            Console.WriteLine($"Empirical mean:  {empiricalMean:F4}  (expected ≈ 0.0)");
            Console.WriteLine($"Empirical std:   {empiricalStd:F4}  (expected ≈ {expectedSigma:F4})");
            Console.WriteLine();

            Assert("Empirical mean ≈ 0", Math.Abs(empiricalMean) < 0.05);
            Assert("Empirical std ≈ σ", Math.Abs(empiricalStd - expectedSigma) < 0.2);
        }


        private static List<TrainingSample> GenerateSamples(int n, Random rng)
        {
            var samples = new List<TrainingSample>(n);

            for (int i = 0; i < n; i++)
            {
                var features = Vector<double>.Build.Dense(TotalDim,
                    _ => SampleNormal(rng, 0, 1));

                // Local features less informative
                for (int j = GlobalDim; j < TotalDim; j++)
                    features[j] = SampleNormal(rng, 0, 1);

                double noise = SampleNormal(rng, 0, 0.05);
                double label = 0.5 * features[0] + 0.3 * features[3] + noise;
                // label = Math.Clamp(label, 0.0, 1.0);

                samples.Add(new TrainingSample
                {
                    PatientId = "TEST",
                    DayIndex = i,
                    Date = DateOnly.FromDayNumber(i),
                    Features = features,
                    Label = label,
                    SampleWeight = 1.0,   // all features valid in synthetic data
                });
            }

            return samples;
        }


        // Helpers

        private static double SampleNormal(Random rng, double mean, double std)
        {
            // Box-Muller transform
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return mean + std * z;
        }

        private static void Assert(string description, bool condition)
        {
            string status = condition ? "✅ PASS" : "❌ FAIL";
            Console.WriteLine($"  {status}  {description}");
        }
    }
}