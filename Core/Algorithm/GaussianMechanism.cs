using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Core.Algorithm
{
    // Applies Gaussian mechanism to delta_a_global before transmission to coordinator
    public sealed class GaussianMechanism
    {
        private readonly Random _rng;

        public GaussianMechanism(int? seed = null)
        {
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        // Public API

        public NoisedUpdate Privatize(Vector<double> deltaW, DpBudget budget, double clippingNorm)
        {
            double norm = deltaW.L2Norm();
            var clipped = norm > clippingNorm
            ? deltaW * (clippingNorm / norm)
            : deltaW.Clone();

            double sigma = ComputeSigma(clippingNorm, budget.Epsilon, budget.Delta);

            var noise = SampleGaussianNoise(clipped.Count, sigma);
            var noised = clipped + noise;

            return new NoisedUpdate
            {
                NoisedVector = noised,
                Sigma = sigma,
                Sensitivity = clippingNorm
            };
        }


        // Core computations

        /// <summary>
        /// L2 sensitivity of one SDCA dual coordinate update
        /// </summary>
        public static double ComputeSensitivity(double lambda, int n) => 1.0 / (lambda * n);


        /// <summary>
        /// Gaussian mechanism noise scale for (eps, sigma)
        /// 
        /// Valid for eps in (0, 1] and sigma in (0, 1)
        /// </summary>
        public static double ComputeSigma(double sensitivity, double epsilon, double delta)
        {
            if (epsilon <= 0) throw new ArgumentException("eps must be positive");
            if (delta is <= 0 or >= 1) throw new ArgumentException("sigma must be in (0,1)");

            return sensitivity * Math.Sqrt(2.0 * Math.Log(1.25 / delta)) / epsilon;
        }

        /// <summary>
        /// Samples n independent N(theta, sigma2) values using Box-Muller via MathNet
        /// </summary>
        private Vector<double> SampleGaussianNoise(int count, double sigma)
        {
            var dist = new Normal(mean: 0.0, stddev: sigma, randomSource: _rng);
            var values = new double[count];
            dist.Samples(values);
            return Vector<double>.Build.DenseOfArray(values);
        }


    }

    // Supporting types

    /// <summary> Per-round differential privacy budget </summary>
    public sealed class DpBudget
    {
        /// <summary> Privacy loss parameter. Lower = stronger privacy </summary>
        public double Epsilon { get; init; } = 1.0;

        /// <summary> Failure probability. Typically 1/n or 1e-5 </summary>
        public double Delta { get; init; } = 1e-5;
    }

    /// <summary>
    /// Result of GaussianMechanism.Privatize – noised delta_a ready for transmission
    /// </summary>
    public sealed class NoisedUpdate
    {
        /// <summary> delta_a + noise. This is what the coordinator receives </summary>
        public Vector<double> NoisedVector { get; init; } = null!;

        /// <summary> Noise scale sigma used. Logged for privacy accounting </summary>
        public double Sigma { get; init; }

        /// <summary> Sensitivity = 1/(lambda*n). Logged for audit trail </summary>
        public double Sensitivity { get; init; }
    }
}