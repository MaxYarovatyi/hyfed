using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Core.Algorithm;

namespace Evaluation
{
    public sealed class AblationConfig
    {
        public string Name { get; init; } = string.Empty;

        public ModelVariant Variant { get; init; } = ModelVariant.HyFDCA;
        public SdcaConfig SdcaConfig { get; init; } = new();

        public AggregatorConfig AggregatorConfig { get; init; } = new();
        public DpBudget? DpBudget { get; init; }
        public double ClippingNorm { get; init; } = 0.5;
        public bool UseImpactMatrix { get; init; } = true;
        public double TrainFraction { get; init; } = 0.8;

        // Preset configuration

        public static AblationConfig Baseline => new()
        {
            Name = "HyFDCA_baseline",
            Variant = ModelVariant.HyFDCA,
            SdcaConfig = new SdcaConfig
            {
                LambdaGlobal = 0.01,
                LambdaLocal = 0.05,
                EpochsPerRound = 1
            },
            AggregatorConfig = new AggregatorConfig { MaxRounds = 50 },
            DpBudget = new DpBudget { Epsilon = 5.0, Delta = 1e-5 },
            UseImpactMatrix = true
        };

        public static AblationConfig GlobalOnly => new()
        {
            Name = "GlobalOnly_federated",
            Variant = ModelVariant.GlobalOnly,
            SdcaConfig = new SdcaConfig
            {
                LambdaGlobal = 0.01,
                LambdaLocal = 0.05,
                EpochsPerRound = 1,
            },
            AggregatorConfig = new AggregatorConfig { MaxRounds = 50 },
            DpBudget = null,
            UseImpactMatrix = false
        };

        public static AblationConfig LocalOnly => new()
        {
            Name = "LocalOnly_isolated",
            Variant = ModelVariant.LocalOnly,
            SdcaConfig = new SdcaConfig
            {
                LambdaGlobal = 0.01,
                LambdaLocal = 0.05,
                EpochsPerRound = 5,
            },
            AggregatorConfig = new AggregatorConfig { MaxRounds = 1 },
            DpBudget = null,
            UseImpactMatrix = true
        };

        public static AblationConfig NoImpactMatrix => new()
        {
            Name = "HyFDCA_no_impact_matrix",
            Variant = ModelVariant.LocalOnly,
            SdcaConfig = new SdcaConfig
            {
                LambdaGlobal = 0.01,
                LambdaLocal = 0.05,
                EpochsPerRound = 1,
            },
            AggregatorConfig = new AggregatorConfig { MaxRounds = 50 },
            DpBudget = null,
            UseImpactMatrix = false
        };

        public static IEnumerable<AblationConfig> HSweep(int[] epochsCount) =>
            epochsCount.Select(h => new AblationConfig
            {
                Name = $"HyFDCA_H_{h}",
                Variant = ModelVariant.HyFDCA,
                SdcaConfig = new SdcaConfig
                {
                    LambdaGlobal = 0.01,
                    LambdaLocal = 0.05,
                    EpochsPerRound = h,
                },
                AggregatorConfig = new AggregatorConfig { MaxRounds = 50 },
                UseImpactMatrix = true,
            });

        public static IEnumerable<AblationConfig> LambdaLocalSweep(double[] lambdas) =>
            lambdas.Select(l => new AblationConfig
            {
                Name = $"HyFDCA_lambdaL_{l:F4}",
                Variant = ModelVariant.HyFDCA,
                SdcaConfig = new SdcaConfig
                {
                    LambdaGlobal = 0.01,
                    LambdaLocal = l,
                    EpochsPerRound = 1,
                },
                AggregatorConfig = new AggregatorConfig { MaxRounds = 50 },
                UseImpactMatrix = true,
            });

        public static IEnumerable<AblationConfig> DpSweep(double[] epsilons) =>
            epsilons.Select(eps => new AblationConfig
            {
                Name = $"HyFDCA_eps_{eps}",
                Variant = ModelVariant.HyFDCA,
                SdcaConfig = new SdcaConfig
                {
                    LambdaGlobal = 0.01,
                    LambdaLocal = 0.05,
                    EpochsPerRound = 1,
                },
                AggregatorConfig = new AggregatorConfig { MaxRounds = 50 },
                DpBudget = new DpBudget { Epsilon = eps, Delta = 1e-5 },
                ClippingNorm = 0.1,
                UseImpactMatrix = true,
            });

    }

    public enum ModelVariant
    {
        HyFDCA,
        GlobalOnly,
        LocalOnly
    }
}