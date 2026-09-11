


using Client;
using Core.Algorithm;
using Core.Configuration;
using Core.DataLoading;
using Core.FeatureEngineering;

AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);

const string CoordinatorAddress = "http://localhost:5050";
const int ClientCount = 5;
const int GlobalDim = 6;
const int LocalDim = 13;
const int TotalDim = GlobalDim + LocalDim;

Console.WriteLine("══════════════════════════════════════════════");
Console.WriteLine("  HyFDCA FL Client Simulator");
Console.WriteLine($"  {ClientCount} virtual clients → {CoordinatorAddress}");
Console.WriteLine("══════════════════════════════════════════════\n");

var sdcaConfig = new SdcaConfig
{
    LambdaGlobal = 0.02,
    LambdaLocal = 0.01,
    EpochsPerRound = 2
};

var dpBudget = new DpBudget
{
    Epsilon = 8.0,
    Delta = 1e-5
};

var patientDataLoader = new PatientDataLoader();
var featureConfigLoader = new FeatureConfigLoader();
var featureConfig = featureConfigLoader.LoadFromFile("C:/projects/hyfed/Core/Configuration/fl-feature-config.json");
var matrix = new ImpactMatrix(featureConfig);
var files = Directory.GetFiles("C:/projects/abitur/psychdatagenerator/data/patients_individual/", "*.json").Take(20);


var workers = files.Select((file, k) =>
{
    var patientData = patientDataLoader.LoadFromFile(file);

    var pipeline = new LocalRegressorPipeline(featureConfig, matrix);
    var samples = pipeline.Process(patientData);
    var worker = new FLClientWorker(
        clientId: $"patient_{k:D3}",
        coordinatorAddress: CoordinatorAddress,
        sdcaConfig: sdcaConfig,
        dpBudget: dpBudget,
        clippingNorm: 0.1,
        seed: k
    );

    worker.UpdateSamples(samples);
    return worker;
}).ToList();

using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));
var tasks = workers.Select(w => w.RunAsync(cts.Token)).ToList();

try
{
    await Task.WhenAll(tasks);
    Console.WriteLine("\n All Clients finished");
}
catch (OperationCanceledException)
{
    Console.WriteLine("\n Simulation timed out");
}
finally
{
    foreach (var w in workers) await w.DisposeAsync();
}