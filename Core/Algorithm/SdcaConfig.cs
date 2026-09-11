using MathNet.Numerics.LinearAlgebra;

namespace Core.Algorithm
{
    public sealed class SdcaConfig
    {
        /// <summary>
        /// L2 regularization for global model
        /// </summary>
        public double LambdaGlobal { get; init; } = 0.01;


        /// <summary>
        /// L2 regularization for local model
        /// Higher than LambdaGlobal – local model has less data support
        /// (only residuals, not raw labels)
        /// </summary>
        public double LambdaLocal { get; init; } = 0.05;


        /// <summary>
        /// Number of local epochs per FL round
        /// </summary>
        public int EpochsPerRound { get; init; } = 1;

    }

    public sealed class DualVariableState
    {
        /// <summary>
        /// Dual variables for the global model
        /// Updated each FL round, delta_a_global is sent to coordinator (with noise)
        /// </summary>
        public Vector<double> AlphaGlobal { get; set; } = null!;

        /// <summary>
        /// Dual variables for the local model
        /// Never leaves the device
        /// </summary>
        public Vector<double> AlphaLocal { get; set; } = null!;

        /// <summary>
        /// Current local weight vector
        /// Reconstructed from a_local after each round
        /// Never leaves the device
        /// </summary>
        public Vector<double> WLocal { get; set; } = null!;



        /// <summary>
        /// Initializes zero state for a new patient 
        /// </summary>
        public static DualVariableState Zero(int n, int dLocal) => new()
        {
            AlphaGlobal = Vector<double>.Build.Dense(n),
            AlphaLocal = Vector<double>.Build.Dense(n),
            WLocal = Vector<double>.Build.Dense(dLocal)
        };
    }


    /// <summary>
    /// Output of one LocalSdcaSolver run
    /// </summary>
    public sealed class SolverResult
    {
        /// <summary>
        /// delta_a_global = a_global_new - a_global_old
        /// This is what gets noised and sent to the coordinator
        /// </summary>
        public Vector<double> DeltaWGlobal { get; init; } = null!;


        /// <summary>
        /// Updated dual variable state
        /// Replaces the previous state on device after the round
        /// </summary>
        public DualVariableState UpdatedState { get; init; } = null!;


        /// <summary>
        /// Training loss after this round
        /// Used for convergence monitoring
        /// </summary>
        public double TrainingLoss { get; init; }
    }
}