using MathNet.Numerics.LinearAlgebra;
using Core.Models;

namespace Core.Algorithm
{
    /// <summary>
    /// Core HyFDCA local solver. Implements one FL round of primal-dual optimization
    /// for a single patient using SDCA
    /// </summary>
    public sealed class LocalSdcaSolver
    {
        private readonly SdcaConfig _config;
        private readonly Random _rng;

        //Feature split indices
        private const int GlobalDim = 6;

        public LocalSdcaSolver(SdcaConfig config, int? seed = null)
        {
            _config = config;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();
        }


        //Public API

        /// <summary>
        /// Runs H epochs of SDCA for one FL round
        /// </summary>
        /// <param name="samples">Training samplse for this patient </param>
        /// <param name="wGlobalFromCoordinator">w_global received form coordinator this round </param>
        /// <param name="state">Current dual variable state (warm start from previous round)</param>
        /// <returns></returns>
        public SolverResult RunRound(IReadOnlyList<TrainingSample> samples,
        Vector<double> wGlobalFromCoordinator,
        DualVariableState state)
        {
            int n = samples.Count;
            int dLocal = state.WLocal.Count;



            //Working copies – mutated during optimization
            var wGlobal = wGlobalFromCoordinator.Clone();
            var wLocal = state.WLocal.Clone();
            var alphaG = state.AlphaGlobal.Clone();
            var alphaL = state.AlphaLocal.Clone();

            //H epochs over all samples
            for (int epoch = 0; epoch < _config.EpochsPerRound; epoch++)
            {
                var shuffled = ShuffledIndices(n);



                foreach (int i in shuffled)
                {
                    var sample = samples[i];
                    var xGLobal = sample.Features.SubVector(0, GlobalDim);
                    var xLocal = sample.Features.SubVector(GlobalDim, dLocal);
                    double y = sample.Label;
                    double sw = sample.SampleWeight;


                    // Global SDCA Step
                    // Residual for global: what global model still needs to explain
                    // given current local model contribution
                    double rGlobal = y - wGlobal.DotProduct(xGLobal) - wLocal.DotProduct(xLocal);

                    double deltaAlphaG = ComputeDelta(
                        residual: rGlobal,
                        alpha_i: alphaG[i],
                        x: xGLobal,
                        lambda: _config.LambdaGlobal,
                        n: n,
                        sampleWeight: sw);

                    alphaG[i] += deltaAlphaG;
                    //wGlobal.MapIndexedInplace((_, v) => v); //no-op placeholder
                    wGlobal += deltaAlphaG / (_config.LambdaGlobal * n) * xGLobal;
                }
                foreach (int i in shuffled)
                {
                    var sample = samples[i];
                    var xGLobal = sample.Features.SubVector(0, GlobalDim);
                    var xLocal = sample.Features.SubVector(GlobalDim, dLocal);

                    double rLocal = sample.Label - wGlobal.DotProduct(xGLobal) - wLocal.DotProduct(xLocal);

                    double deltaAlphaL = ComputeDelta(rLocal, alphaL[i], xLocal, _config.LambdaLocal, n, sample.SampleWeight);
                    alphaL[i] += deltaAlphaL;
                    wLocal += deltaAlphaL / (_config.LambdaLocal * n) * xLocal;
                }


                // foreach (var i in shuffled)
                // {
                //     var sample = samples[i];
                //     var xGLobal = sample.Features.SubVector(0, GlobalDim);
                //     var xLocal = sample.Features.SubVector(GlobalDim, dLocal);
                //     double y = sample.Label;
                //     double sw = sample.SampleWeight;

                //     // Local SDCA step
                //     // Residual for local: what local model needs to explain
                //     // given updated global model contribution

                //     double rLocal = y - wGlobal.DotProduct(xGLobal) - wLocal.DotProduct(xLocal);

                //     double deltaAlphaL = ComputeDelta(
                //         residual: rLocal,
                //         alpha_i: alphaL[i],
                //         x: xLocal,
                //         lambda: _config.LambdaLocal,
                //         n: n,
                //         sampleWeight: sw
                //     );

                //     alphaL[i] += deltaAlphaL;
                //     wLocal += (deltaAlphaL / (_config.LambdaLocal * n)) * xLocal;
                // }
            }

            var deltaWGlobal = wGlobal - wGlobalFromCoordinator;

            double loss = ComputeLoss(samples, wGlobal, wLocal, dLocal);

            return new SolverResult
            {
                DeltaWGlobal = deltaWGlobal,
                UpdatedState = new DualVariableState
                {
                    AlphaGlobal = alphaG,
                    AlphaLocal = alphaL,
                    WLocal = wLocal
                },
                TrainingLoss = loss
            };
        }


        // SDCA coordinate update

        /// <summary>
        /// Closed-form SDCA update for squared loss with sample weighting
        /// </summary>
        private static double ComputeDelta(double residual, double alpha_i, Vector<double> x, double lambda, int n, double sampleWeight)
        {
            double lambdaN = lambda * n;
            double xNormSq = x.DotProduct(x);

            double numerator = sampleWeight * residual - alpha_i;
            double denominator = sampleWeight * xNormSq / lambdaN + 1.0;

            return numerator / denominator;
        }

        // Loss computation

        /// <summary>
        /// Weighted MSE
        /// </summary>
        private static double ComputeLoss(IReadOnlyList<TrainingSample> samples, Vector<double> wGlobal, Vector<double> wLocal, int dLocal)
        {
            double weightedSse = 0.0;
            double totalWeight = 0.0;

            foreach (var s in samples)
            {
                var xGlobal = s.Features.SubVector(0, GlobalDim);
                var xLocal = s.Features.SubVector(GlobalDim, dLocal);
                double yHat = wGlobal.DotProduct(xGlobal) + wLocal.DotProduct(xLocal);
                double err = s.Label - yHat;
                weightedSse += s.SampleWeight * err * err;
                totalWeight += s.SampleWeight;
            }

            return totalWeight > 0 ? weightedSse / totalWeight : 0.0;
        }


        //Helpers
        private int[] ShuffledIndices(int n)
        {
            var indices = Enumerable.Range(0, n).ToArray();
            for (int i = n - 1; i > 0; i--)
            {
                int j = _rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            return indices;
        }
    }
}