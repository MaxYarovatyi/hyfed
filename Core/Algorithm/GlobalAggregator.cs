using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Core.Algorithm
{
    public sealed class GlobalAggregator
    {
        private readonly AggregatorConfig _config;
        private int _consecutiveConvergedRounds = 0;

        /// <summary>Current global model weights</summary>
        public Vector<double> WGlobal { get; private set; }

        /// <summary>Current FL round number</summary>
        public int Round { get; private set; }

        public GlobalAggregator(AggregatorConfig config)
        {
            _config = config;
            WGlobal = Vector<double>.Build.Dense(config.GlobalFeatureDim);
        }


        // Public API

        /// <summary>
        /// Aggregates client updates and returns the result of this round.
        /// </summary>
        /// <param name="updates">Updates from all participating clients this round</param>
        public AggregationResult Aggregate(IReadOnlyList<ClientUpdate> updates)
        {

            if (updates.Count == 0)
                throw new ArgumentException("No client updates to aggregate");

            Round++;

            int totalSamples = updates.Sum(u => u.SampleCount);
            var weightedDelta = Vector<double>.Build.Dense(_config.GlobalFeatureDim);

            foreach (var update in updates)
            {
                double weight = (double)update.SampleCount / totalSamples;
                weightedDelta += weight * update.DeltaWGlobal;
            }

            WGlobal += _config.ServerLearningRate * weightedDelta;

            if (WGlobal.Exists(double.IsNaN))
            {
                WGlobal = Vector<double>.Build.Dense(_config.GlobalFeatureDim);
                Console.WriteLine($"[Aggregator] Warning: NaN detected in w_global at round {Round}. Reseting to zero");
            }

            double updateNorm = weightedDelta.L2Norm();
            bool isConverged = CheckConvergence(updateNorm);
            Console.WriteLine($"[Aggregator] R{Round} | " +
                $"‖weightedΔw‖={updateNorm:F6} | " +
                $"wG=[{string.Join(",", WGlobal.Select(v => v.ToString("F3")))}]");
            return new AggregationResult
            {
                Round = Round,
                NewWGlobal = WGlobal.Clone(),
                UpdateNorm = updateNorm,
                ClientCount = updates.Count,
                TotalSamples = totalSamples,
                IsConverged = isConverged
            };
        }


        // Convergence

        private bool CheckConvergence(double updateNorm)
        {
            if (Round > _config.MaxRounds)
                return true;
            if (updateNorm < _config.ConvergenceTolerance)
                _consecutiveConvergedRounds++;
            else
                _consecutiveConvergedRounds = 0;

            return _consecutiveConvergedRounds >= _config.ConvergenceRoundsRequired;
        }
    }


    // Supporting types

    public sealed class AggregatorConfig
    {
        /// <summary>
        /// Dimensionality of w_global – must match GlobalDim in solver
        /// </summary>
        public int GlobalFeatureDim { get; init; } = 6;


        /// <summary>
        /// Second norm of delta_w below this threshold counts as a converged round
        /// </summary>
        public double ConvergenceTolerance { get; init; } = 1e-4;


        /// <summary>
        /// How many consecutive converged rounds before declaring convergence
        /// K = 3 avoids false positives from DP noise fluctuations
        /// </summary>
        public int ConvergenceRoundsRequired { get; init; } = 3;

        /// <summary>
        /// Server learning rate n applied to the aggregated update
        /// n = 1.0 is standart FedAvg
        /// n < 1.0 dampens influence of anomalous clients
        /// </summary>
        public double ServerLearningRate { get; init; } = .5;

        /// <summary>
        /// Maximum FL rounds before forced stop
        /// </summary>
        public int MaxRounds { get; init; } = 50;
    }

    /// <summary>
    /// Update sent from one client to the coordinator per FL round
    /// </summary>
    public sealed class ClientUpdate
    {
        public string ClientId { get; init; } = string.Empty;


        public Vector<double> DeltaWGlobal { get; init; } = null!;

        public int SampleCount { get; init; }
    }


    public sealed class AggregationResult
    {
        public int Round { get; init; }
        public Vector<double> NewWGlobal { get; init; } = null!;
        public double UpdateNorm { get; init; }
        public int ClientCount { get; init; }
        public int TotalSamples { get; init; }
        public bool IsConverged { get; init; }

        public override string ToString() =>
            $"Round {Round}: ‖Δw‖={UpdateNorm:F6} " +
            $"clients={ClientCount} samples={TotalSamples}" +
            $"{(IsConverged ? " → CONVERGED" : "")}";
    }
}