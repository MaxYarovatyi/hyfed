using MathNet.Numerics.LinearAlgebra;
using communication;
using Core.Algorithm;
using Core.Models;
using Grpc.Core;
using Grpc.Net.Client;
using Google.Protobuf.WellKnownTypes;


namespace Client
{
    /// <summary>
    /// Orchestrates the FL round lifecycle on a mobile client.
    /// 
    /// Round flow:
    /// 1. SubscribeToRounds() – open persistent stream, listen for w_global broadcast
    /// 2. On each broadcast:
    ///     a. Run local sdca solver with new w_global
    ///     b. Apply GaussianMeachanism to delta_w_global
    ///     c. Call SubmitUpdate() with noised delta_W
    /// 3. If coordinator signals IsConverged -> stop training, keep w_local for inference
    /// 
    /// One instance per patient (one client = one patient)
    /// DiagnosisType is fixed at construction – determines local feature space
    /// </summary>
    public sealed class FLClientWorker : IAsyncDisposable
    {
        private readonly string _clientId;
        private readonly GrpcChannel _channel;
        private readonly FLCoordinator.FLCoordinatorClient _grpc;
        private readonly LocalSdcaSolver _solver;
        private readonly GaussianMechanism _mechanism;
        private readonly DpBudget _dpBudget;
        private readonly double _clippingNorm;
        private readonly SdcaConfig _sdcaConfig;

        // State persisted between rounds
        private DualVariableState? _state;

        // Training samples – updated as new data arrives from wearable
        private IReadOnlyList<TrainingSample> _samples = [];

        public FLClientWorker(
            string clientId,
        string coordinatorAddress,
        SdcaConfig sdcaConfig,
        DpBudget dpBudget,
        double clippingNorm = 0.1,
        int? seed = null)
        {
            _clientId = clientId;
            _sdcaConfig = sdcaConfig;
            _dpBudget = dpBudget;
            _clippingNorm = clippingNorm;

            _channel = GrpcChannel.ForAddress(coordinatorAddress, new GrpcChannelOptions
            {
                HttpHandler = new SocketsHttpHandler
                {
                    EnableMultipleHttp2Connections = true
                }
            });
            _grpc = new FLCoordinator.FLCoordinatorClient(_channel);
            _solver = new LocalSdcaSolver(sdcaConfig, seed);
            _mechanism = new GaussianMechanism(seed);
        }


        // Public API

        /// <summary>
        /// Updates local training data. Called when new daily samples are available
        /// </summary>
        public void UpdateSamples(IReadOnlyList<TrainingSample> samples)
        {
            _samples = samples;

            // Re-initialize dual variable state if sample count changed
            if (_state == null || _state.AlphaGlobal.Count != samples.Count)
            {
                int dLocal = _state?.WLocal.Count ?? samples[0].Features.Count - 6;
                _state = DualVariableState.Zero(samples.Count, dLocal);
            }
        }

        public async Task RunAsync(CancellationToken cancellationToken = default)
        {
            var subscribeRequest = new SubscribeRequest { ClientId = _clientId };
            using var stream = _grpc.SubscribeToRounds(subscribeRequest, cancellationToken: cancellationToken);

            await foreach (var broadcast in stream.ResponseStream.ReadAllAsync(cancellationToken))
            {
                if (_samples.Count == 0)
                {
                    Console.WriteLine($"[{_clientId}] No smaples yet, skipping round {broadcast.RoundId}");
                    continue;
                }

                await ParticipateInRoundAsync(broadcast);

                if (broadcast.IsConverged)
                {
                    Console.WriteLine($"[{_clientId}] Coordinator signaled convergence. Stopping");
                    break;
                }
            }
        }

        /// <summary>
        /// Current local model weights for inference
        /// w_local is never sent ro coordinator
        /// </summary>
        public Vector<double>? WLocal => _state?.WLocal;

        // Round participation

        private async Task ParticipateInRoundAsync(RoundBroadcast broadcast)
        {
            var wGlobal = Vector<double>.Build.DenseOfEnumerable(broadcast.WGlobal);
            var oldAlphaG = _state!.AlphaGlobal;
            _state!.AlphaGlobal.Clear();
            _state!.AlphaLocal.Clear();

            // 1. Local training
            var result = _solver.RunRound(_samples, wGlobal, _state!);

            double norm = result.DeltaWGlobal.L2Norm();
            double clipRatio = norm > _clippingNorm ? _clippingNorm / norm : 1.0;
            Console.WriteLine($"[{_clientId}] R{broadcast.RoundId} | " +
    $"wG_in=[{string.Join(",", wGlobal.Select(v => v.ToString("F3")))}] | " +
    $"ΔwG=[{string.Join(",", result.DeltaWGlobal.Select(v => v.ToString("F4")))}] | " +
    $"‖ΔwG‖={result.DeltaWGlobal.L2Norm():F4} | " +
    $"wL_norm={result.UpdatedState.WLocal.L2Norm():F4} | " +
    $"loss={result.TrainingLoss:F4}");



            _state = result.UpdatedState;

            // 2. Apply DP: L2 clip + Gaussian noise
            var noised = _mechanism.Privatize(result.DeltaWGlobal, _dpBudget, _clippingNorm);
            Console.WriteLine($"[{_clientId}] DP | " +
                $"σ={noised.Sigma:F4} | " +
                $"‖Δw_before‖={result.DeltaWGlobal.L2Norm():F4} | " +
                $"‖Δw_after‖={noised.NoisedVector.L2Norm():F4}");


            var newAlphaG = result.UpdatedState.AlphaGlobal;

            for (int i = 0; i < oldAlphaG.Count; i++)
            {
                double deltaAlpha_i = newAlphaG[i] - oldAlphaG[i];
                result.UpdatedState.AlphaGlobal[i] = oldAlphaG[i] + (deltaAlpha_i * clipRatio);
            }

            // 3. Submit to coordinator
            var request = new ClientUpdateRequest
            {
                ClientId = _clientId,
                RoundId = broadcast.RoundId,
                SampleCount = _samples.Count
            };
            request.DeltaWGlobal.AddRange(noised.NoisedVector);

            var response = await _grpc.SubmitUpdateAsync(request);

            if (!response.Accepted && response.RejectReason.StartsWith("stale_round"))
            {
                var current = await _grpc.GetGlobalModelAsync(new GetGlobalModelRequest { ClientId = _clientId });

                var wGlobalCurrent = Vector<double>.Build.DenseOfEnumerable(current.WGlobal);
                var retryResult = _solver.RunRound(_samples, wGlobalCurrent, _state!);
                _state = retryResult.UpdatedState;

                var retryNoised = _mechanism.Privatize(retryResult.DeltaWGlobal, _dpBudget, _clippingNorm);
                var retryRequest = new ClientUpdateRequest
                {
                    ClientId = _clientId,
                    RoundId = current.RoundId,
                    SampleCount = _samples.Count
                };
                retryRequest.DeltaWGlobal.AddRange(retryNoised.NoisedVector);
                await _grpc.SubmitUpdateAsync(retryRequest);

                Console.WriteLine($"[{_clientId}] Resynced to round: {current.RoundId}");

            }
            else
                Console.WriteLine($"[{_clientId}] Round {broadcast.RoundId} submitted." +
                $"Loss = {result.TrainingLoss:F4}" +
                $"Updates: {response.UpdatesReceived}/{response.MinClientsRequired}");
        }

        public async ValueTask DisposeAsync() => await _channel.ShutdownAsync();

    }
}