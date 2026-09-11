using Grpc.Core;
using Core.Algorithm;
using MathNet.Numerics.LinearAlgebra;
using communication;

namespace Coordinator.Services
{
    public sealed class FLCoordinatorService : FLCoordinator.FLCoordinatorBase
    {
        private readonly GlobalAggregator _aggregator;
        private readonly CoordinatorConfig _config;
        private readonly List<ClientUpdateRequest> _pendingUpdates = [];

        //Active streaming subscribers: client_id -> response stream
        private readonly Dictionary<string, IServerStreamWriter<RoundBroadcast>> _subscribers = [];
        private readonly SemaphoreSlim _lock = new(1, 1);

        public FLCoordinatorService(GlobalAggregator aggregator, CoordinatorConfig config)
        {
            _aggregator = aggregator;
            _config = config;
        }


        // RPC: GetGlobalModel

        public override Task<GlobalModelResponse> GetGlobalModel(GetGlobalModelRequest request, ServerCallContext context)
        {
            var response = new GlobalModelResponse
            {
                RoundId = _aggregator.Round
            };
            response.WGlobal.AddRange(_aggregator.WGlobal);
            return Task.FromResult(response);
        }

        // RPC: SubmitUpdate

        public override async Task<SubmitUpdateResponse> SubmitUpdate(ClientUpdateRequest request, ServerCallContext context)
        {
            await _lock.WaitAsync();
            try
            {
                if (request.RoundId != _aggregator.Round)
                {
                    return new SubmitUpdateResponse
                    {
                        Accepted = false,
                        RejectReason = $"stale_round: expected {_aggregator.Round}, got {request.RoundId}",
                        UpdatesReceived = _pendingUpdates.Count,
                        MinClientsRequired = _config.MinClientsPerRound
                    };
                }

                // Reject duplicates with same round
                if (_pendingUpdates.Any(u => u.ClientId == request.ClientId))
                {
                    return new SubmitUpdateResponse
                    {
                        Accepted = false,
                        RejectReason = "duplicate_client"
                    };

                }
                _pendingUpdates.Add(request);

                int received = _pendingUpdates.Count;
                int required = _config.MinClientsPerRound;

                if (received >= required)
                    await AggregateAndBroadcastAsync();

                return new SubmitUpdateResponse
                {
                    Accepted = true,
                    UpdatesReceived = received,
                    MinClientsRequired = required
                };
            }
            finally
            {
                _lock.Release();
            }
        }

        // RPC: Subscribe to rounds

        public override async Task SubscribeToRounds(
            SubscribeRequest request,
            IServerStreamWriter<RoundBroadcast> responseStream,
            ServerCallContext context
        )
        {
            await _lock.WaitAsync();
            _subscribers[request.ClientId] = responseStream;
            _lock.Release();

            var initial = new RoundBroadcast
            {
                RoundId = _aggregator.Round,
                IsConverged = false,
            };
            initial.WGlobal.AddRange(_aggregator.WGlobal);
            await responseStream.WriteAsync(initial);

            // Keep stream open until client disconnects
            try
            {
                await Task.Delay(Timeout.Infinite, context.CancellationToken);
            }
            catch (OperationCanceledException)
            {
                await _lock.WaitAsync();
                _subscribers.Remove(request.ClientId);
                _lock.Release();
            }
        }

        // Aggregation
        private async Task AggregateAndBroadcastAsync()
        {
            var clientUpdates = _pendingUpdates
            .Select(r => new ClientUpdate
            {
                ClientId = r.ClientId,
                DeltaWGlobal = Vector<double>.Build.DenseOfEnumerable(r.DeltaWGlobal),
                SampleCount = r.SampleCount
            }).ToList();

            _pendingUpdates.Clear();

            var result = _aggregator.Aggregate(clientUpdates);

            // Broadcast new w_global to all subscribed clients
            var broadcast = new RoundBroadcast
            {
                RoundId = result.Round,
                IsConverged = result.IsConverged
            };
            broadcast.WGlobal.AddRange(result.NewWGlobal);

            var broadcastTasks = _subscribers.Values
            .Select(stream => stream.WriteAsync(broadcast));

            await Task.WhenAll(broadcastTasks);
        }

    }

    public sealed class CoordinatorConfig
    {
        public int MinClientsPerRound { get; init; } = 3;
    }
}