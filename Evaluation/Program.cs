using Core.Configuration;
using Core.DataLoading;
using Core.FeatureEngineering;
using Core.Models;
using Evaluation;
using MathNet.Numerics.Statistics;

var dataDirectory = "C:/projects/abitur/psychdatagenerator/data/patients_individual";

var patientDataLoader = new PatientDataLoader();
var featureConfigLoader = new FeatureConfigLoader();
var featureConfig = featureConfigLoader.LoadFromFile("C:/projects/hyfed/core/configuration/fl-feature-config.json");
var matrix = new ImpactMatrix(featureConfig);
var pipeline = new LocalRegressorPipeline(featureConfig, matrix);

var patientLookup = patientDataLoader.LoadFromDirectory(dataDirectory).ToDictionary(p => p.PatientId);

var allSamples = Directory.GetFiles(dataDirectory, "*.json")
    .ToDictionary(
        file => Path.GetFileNameWithoutExtension(file),
        file => pipeline.Process(patientDataLoader.LoadFromFile(file)));

// var sample = allSamples.First();
// var study = new AblationStudy(new Dictionary<string, IReadOnlyList<TrainingSample>>() { { sample.Key, sample.Value } });
var study = new AblationStudy(allSamples);
var results = study.RunAll();

var baseline = results.First(r => r.ConfigName.ToLower().Contains("baseline"));

var patientRows = baseline.PerPatient.Select(p =>
{
    var data = patientLookup[p.PatientId];
    var labels = allSamples[p.PatientId].Select(p => p.Label + p.LabelMean);
    return new PatientEvaluationResult
    {
        PatientId = p.PatientId,
        Diagnosis = data.Diagnosis,
        Trajectory = data.Trajectory,
        Test = p.Metrics,
        Train = p.TrainMetrics,
        LabelMean = allSamples[p.PatientId][0].LabelMean,
        LabelStd = Statistics.StandardDeviation(labels),
        TestCount = p.TestCount,
        TrainCount = p.TrainCount,
        WLocalNorm = p.WLocalNorm
    };
});

var ablationRows = results.Select(r => new AblationRunResult
{
    ConfigName = r.ConfigName,
    MeanTestR2 = r.Aggregated.R2_Mean,
    StdTestR2 = r.Aggregated.R2_Std,
    MeanTestMAE = r.Aggregated.MAE_Mean,
    StdTestMAE = r.Aggregated.MAE_Std,
    MeanTestMSE = r.Aggregated.MSE_Mean,
    StdTestMSE = r.Aggregated.MSE_Std,
    MeanTestRMSE = r.Aggregated.RMSE_Mean,
    StdTestRMSE = r.Aggregated.RMSE_Std,
    MeanTrainR2 = r.AggregatedTrain.R2_Mean,
    StdTrainR2 = r.AggregatedTrain.R2_Std,
    MeanTrainMAE = r.AggregatedTrain.MAE_Mean,
    StdTrainMAE = r.AggregatedTrain.MAE_Std,
    MeanTrainRMSE = r.AggregatedTrain.RMSE_Mean,
    StdTrainRMSE = r.AggregatedTrain.RMSE_Std
});

ResultExporter.WriteAblationSummary(ablationRows, "./ablation_results.csv");

ResultExporter.WritePatientMetrics(patientRows, "./patient_results");

var rounds = ResultExporter.BuildRoundSummaries(baseline.History.Rounds);

ResultExporter.WriteConvergence(rounds, "./convergence_result.csv");

// var basel = study.RunSingle(AblationConfig.Baseline);

// var baserounds = ResultExporter.BuildRoundSummaries(basel.History.Rounds);
// ResultExporter.WriteConvergence(baserounds, "./convergence_baseline.csv");