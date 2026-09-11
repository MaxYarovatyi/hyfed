using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Core.Algorithm;
using Core.Models;
using MathNet.Numerics.LinearAlgebra;

namespace Evaluation
{
    public class InProcessFLRunner
    {
        private readonly AblationConfig _config;
        private readonly GaussianMechanism? _mechanism;

        public InProcessFLRunner(AblationConfig config)
        {
            _config = config;
            _mechanism = config.DpBudget != null ? new GaussianMechanism(seed: 42) : null;
        }

        // Public API

        public IReadOnlyList<PatientModelState> Run(IReadOnlyList<PatientTrainData> patients, TrainingHistory? history = null)
        {
            return _config.Variant switch
            {
                ModelVariant.HyFDCA => RunFederated(patients, history),
                ModelVariant.GlobalOnly => RunFederated(patients, history),
                ModelVariant.LocalOnly => RunLocalOnly(patients, history),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        // Federated (HyFDCA / GlobalOnly)

        private IReadOnlyList<PatientModelState> RunFederated(IReadOnlyList<PatientTrainData> patients, TrainingHistory? history)
        {
            int maxRounds = _config.AggregatorConfig.MaxRounds;

            var aggregator = new GlobalAggregator(_config.AggregatorConfig);
            var solvers = patients.Select((_, k) =>
                new LocalSdcaSolver(_config.SdcaConfig, seed: k)).ToList();
            var states = patients.Select(p =>
                DualVariableState.Zero(p.TrainSamples.Count, p.LocalDim)).ToList();

            for (int round = 0; round < maxRounds; round++)
            {

                var updates = new List<ClientUpdate>();
                var roundData = new List<PatientRoundData>();

                for (int k = 0; k < patients.Count; k++)
                {

                    var result = solvers[k].RunRound(
                        patients[k].TrainSamples,
                        aggregator.WGlobal,
                        states[k]
                    );

                    states[k] = result.UpdatedState;

                    // states[k].AlphaGlobal.Clear();
                    // states[k].AlphaLocal.Clear();

                    // Duality gap on global model
                    double gap = DualityGapCalculator.Compute(
                        patients[k].TrainSamples,
                        aggregator.WGlobal,
                        states[k].WLocal,
                        states[k].AlphaGlobal,
                        _config.SdcaConfig.LambdaGlobal,
                        _config.SdcaConfig.LambdaLocal
                        );

                    roundData.Add(new PatientRoundData
                    {
                        PatientId = patients[k].PatientId,
                        TrainLoss = result.TrainingLoss,
                        DualityGap = gap,
                        WLocalNorm = states[k].WLocal.L2Norm(),
                        WGlobalNorm = aggregator.WGlobal.L2Norm()
                    });


                    var deltaW = ApplyDpIfNeeded(result.DeltaWGlobal, patients[k].TrainSamples.Count);

                    updates.Add(new ClientUpdate
                    {
                        ClientId = patients[k].PatientId,
                        DeltaWGlobal = deltaW,
                        SampleCount = patients[k].TrainSamples.Count
                    });
                }

                history?.Record(round, roundData);

                var aggResult = aggregator.Aggregate(updates);
                if (aggResult.IsConverged) break;
            }

            return patients.Select((p, k) => new PatientModelState
            {
                PatientId = p.PatientId,
                WGlobal = aggregator.WGlobal.Clone(),
                WLocal = states[k].WLocal.Clone()
            }).ToList();
        }


        // Local only
        private IReadOnlyList<PatientModelState> RunLocalOnly(IReadOnlyList<PatientTrainData> patients, TrainingHistory? history)
        {
            int totalRounds = _config.AggregatorConfig.MaxRounds;
            var wGlobalZero = Vector<double>.Build.Dense(6);

            return patients.Select((p, k) =>
            {
                var solver = new LocalSdcaSolver(_config.SdcaConfig, seed: k);
                var state = DualVariableState.Zero(p.TrainSamples.Count, p.LocalDim);

                for (int round = 0; round < totalRounds; round++)
                {
                    var result = solver.RunRound(p.TrainSamples, wGlobalZero, state);
                    state = result.UpdatedState;

                    // if (history != null)
                    // {
                    //     double gap = DualityGapCalculator.Compute(
                    //         p.TrainSamples,
                    //         state.WLocal,
                    //         state.AlphaLocal,
                    //         _config.SdcaConfig.LambdaLocal);

                    //     history.Record(round, [new PatientRoundData
                    //     {
                    //         PatientId = p.PatientId,
                    //         TrainLoss = result.TrainingLoss,
                    //         DualityGap = gap,
                    //         WLocalNorm = state.WLocal.L2Norm(),
                    //         WGlobalNorm = 0.0
                    //     }]);
                    // }
                }

                return new PatientModelState
                {
                    PatientId = p.PatientId,
                    WGlobal = wGlobalZero,
                    WLocal = state.WLocal.Clone()
                };
            }).ToList();
        }


        // DP

        private Vector<double> ApplyDpIfNeeded(Vector<double> deltaW, int n)
        {
            if (_mechanism == null || _config.DpBudget == null)
                return deltaW;

            return _mechanism.Privatize(deltaW, _config.DpBudget, _config.ClippingNorm).NoisedVector;
        }
    }


    // Supporting types
    public sealed class PatientTrainData
    {
        public string PatientId { get; init; } = string.Empty;
        public IReadOnlyList<TrainingSample> TrainSamples { get; init; } = [];

        public int LocalDim => TrainSamples[0].Features.Count - 6;
    }

    public sealed class PatientModelState
    {
        public string PatientId { get; init; } = string.Empty;
        public Vector<double> WGlobal { get; init; } = null!;
        public Vector<double> WLocal { get; init; } = null!;
    }
}