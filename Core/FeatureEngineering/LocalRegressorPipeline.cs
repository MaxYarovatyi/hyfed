using Core.Configuration;
using Core.Models;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using static Core.Configuration.FeatureConfig;

namespace Core.FeatureEngineering
{
    public sealed class LocalRegressorPipeline
    {
        private readonly FeatureConfig _config;
        private readonly ImpactMatrix _matrix;
        private PatientNormalizedStats? _normStats;

        public LocalRegressorPipeline(FeatureConfig config, ImpactMatrix matrix)
        {
            _config = config;
            _matrix = matrix;
        }

        /// <summary>
        /// Transforms all valid samples of a patient into TrainingSamples.
        /// Samples where delta features are all null (first few days) are
        /// still included but receive lower SampleWeight.
        /// </summary>
        public IReadOnlyList<TrainingSample> Process(PatientData patient)
        {
            _normStats = ComputeNormStats(patient.Samples);

            var wFeat = _matrix.ComputeFeatureWeights(patient.ConfidenceVector);

            var results = new List<TrainingSample>(patient.Samples.Count);

            for (int i = 0; i < patient.Samples.Count; i++)
            {
                var sample = patient.Samples[i];
                var delta = patient.DeltaFeatures[i];

                var trainingSample = BuildSample(sample, delta, wFeat);
                results.Add(trainingSample);
            }

            return results;
        }


        //Private

        /// <summary>
        /// Computes mean and std for each global feature across all valid samples.
        /// Uses population std (not sample std) — consistent with z-score convention.
        /// If std ≈ 0 (constant feature), uses 1.0 to avoid division by zero.
        /// </summary>

        private static PatientNormalizedStats ComputeNormStats(IReadOnlyList<BiometricSample> samples)
        {
            double[] raw(Func<BiometricSample, double> sel) => [.. samples.Select(sel)];

            return new PatientNormalizedStats
            {
                HRV_Mean = Statistics.Mean(raw(s => s.HRV)),
                HRV_Std = SafeStd(raw(s => s.HRV)),
                HR_Mean = Statistics.Mean(raw(s => s.HR)),
                HR_Std = SafeStd(raw(s => s.HR)),
                Steps_Mean = Statistics.Mean(raw(s => s.Steps)),
                Steps_Std = SafeStd(raw(s => s.Steps)),
                SleepDuration_Mean = Statistics.Mean(raw(s => s.SleepDuration)),
                SleepDuration_Std = SafeStd(raw(s => s.SleepDuration)),
                SleepQuality_Mean = Statistics.Mean(raw(s => s.SleepQuality)),
                SleepQuality_Std = SafeStd(raw(s => s.SleepQuality)),
                ActivityLevel_Mean = Statistics.Mean(raw(s => s.ActivityLevel)),
                ActivityLevel_Std = SafeStd(raw(s => s.ActivityLevel)),

                Label_Mean = Statistics.Mean(samples.Where(s => s.CriticalityScore != 0).Select(s => s.CriticalityScore))
            };
        }

        private DeltaFeatureSet NormalizeDelta(DeltaFeatureSet delta, PatientNormalizedStats stats)
        {
            double Norm(double? val, double std) => val.HasValue ? val.Value / std : 0.0;

            return new DeltaFeatureSet
            {
                HRV_DeltaMean7 = Norm(delta.HRV_DeltaMean7, stats.HRV_Std),
                HRV_Slope14 = Norm(delta.HRV_Slope14, stats.HRV_Std),
                Sleep_DeltaMean7 = Norm(delta.Sleep_DeltaMean7, stats.SleepDuration_Std),
                Sleep_Slope14 = Norm(delta.Sleep_Slope14, stats.SleepDuration_Std),
                Steps_DeltaMean7 = Norm(delta.Steps_DeltaMean7, stats.Steps_Std),
                Steps_Slope14 = Norm(delta.Steps_Slope14, stats.Steps_Std),
                Steps_Std7 = Norm(delta.Steps_Std7, stats.Steps_Std),
                HR_Std7 = Norm(delta.HR_Std7, stats.HR_Std),
            };
        }

        private static double SafeStd(double[] values)
        {
            double std = Statistics.StandardDeviation(values);
            return std < 1e-9 ? 1.0 : std;
        }

        private TrainingSample BuildSample(BiometricSample sample, DeltaFeatureSet delta, Vector<double> wFeat)
        {
            var xGlobal = NormalizeGlobal(sample);
            var normDelta = NormalizeDelta(delta, _normStats!);
            var xUnion = AssembleLocalUnion(sample, normDelta, xGlobal);

            var xLocal = xUnion.PointwiseMultiply(wFeat);

            var features = Vector<double>.Build.DenseOfEnumerable(xGlobal.Concat(xLocal));

            double weight = (double)delta.ValidFeatureCount / DeltaFeatureSet.TotalFeatureCount;
            weight = Math.Max(weight, 0.1);


            return new TrainingSample
            {
                PatientId = sample.PatientId,
                DayIndex = sample.DayIndex,
                Date = sample.Date,
                Features = features,
                Label = sample.CriticalityScore - _normStats!.Label_Mean,
                LabelMean = _normStats!.Label_Mean,
                SampleWeight = weight,
            };
        }


        private Vector<double> NormalizeGlobal(BiometricSample s)
        {
            var n = _normStats!;
            var v = Vector<double>.Build.Dense([
                (s.HRV - n.HRV_Mean)/n.HRV_Std,
                (s.HR - n.HR_Mean)/n.HR_Std,
                (s.Steps- n.Steps_Mean)/n.Steps_Std,
                (s.SleepDuration-n.SleepDuration_Mean)/n.SleepDuration_Std,
                (s.SleepQuality-n.SleepQuality_Mean)/n.SleepQuality_Std,
                (s.ActivityLevel - n.ActivityLevel_Mean)/n.ActivityLevel_Std,
            ]);

            return v;
        }
        /// <summary>
        /// Assembles x_union in the exact order defined by featureOrder in config.
        /// Delta features: null → 0.0 (handled by SampleWeight instead).
        /// Transform features: computed from raw (non-normalized) biometrics.
        /// </summary>
        private Vector<double> AssembleLocalUnion(BiometricSample sample, DeltaFeatureSet delta, Vector<double> xGlobal)
        {
            var enabledDelta = _config.EnabledDeltaFeatures.ToList();
            var enabledTransform = _config.EnabledTransformFeatures.ToList();

            double GetDeltaValue(string id) => id switch
            {
                "HRV_DeltaMean7" => delta.HRV_DeltaMean7 ?? 0.0,
                "HRV_Slope14" => delta.HRV_Slope14 ?? 0.0,
                "Sleep_DeltaMean7" => delta.Sleep_DeltaMean7 ?? 0.0,
                "Sleep_Slope14" => delta.Sleep_Slope14 ?? 0.0,
                "Steps_DeltaMean7" => delta.Steps_DeltaMean7 ?? 0.0,
                "Steps_Slope14" => delta.Steps_Slope14 ?? 0.0,
                "Steps_Std7" => delta.Steps_Std7 ?? 0.0,
                "HR_Std7" => delta.HR_Std7 ?? 0.0,
                _ => throw new InvalidOperationException($"Unknown delta feature id: {id}")
            };

            double GetNorm(string source) => source switch
            {
                "HRV" => xGlobal[0],
                "HR" => xGlobal[1],
                "Steps" => xGlobal[2],
                "SleepDuration" => xGlobal[3],
                "SleepQuality" => xGlobal[4],
                "ActivityLevel" => xGlobal[5],
                _ => throw new InvalidOperationException($"Unknown biometric source: {source}")
            };

            double GetRaw(string source) => source switch
            {
                "HRV" => sample.HRV,
                "HR" => sample.HR,
                "Steps" => sample.Steps,
                "SleepDuration" => sample.SleepDuration,
                "SleepQuality" => sample.SleepQuality,
                "ActivityLevel" => sample.ActivityLevel,
                _ => throw new InvalidOperationException($"Unknown biometric source: {source}")
            };

            double ComputeTransform(TransformedFeatureConfig t) => t.Type switch
            {
                TransformedFeatureType.Square => GetNorm(t.SourceA) * GetNorm(t.SourceA),
                TransformedFeatureType.Sqrt => Math.Sqrt(Math.Max(0.0, GetNorm(t.SourceA))),
                TransformedFeatureType.Log1p => Math.Log(1.0 + Math.Max(0.0, GetNorm(t.SourceA))),
                TransformedFeatureType.Interaction => GetNorm(t.SourceA) * GetNorm(t.SourceB),
                _ => throw new InvalidOperationException($"Unknown transform type: {t.Type}")
            };

            var values = enabledDelta.Select(f => GetDeltaValue(f.Id))
                .Concat(enabledTransform.Select(ComputeTransform))
                .ToArray();

            return Vector<double>.Build.DenseOfArray(values);
        }

        private sealed class PatientNormalizedStats
        {
            public double HRV_Mean { get; init; }
            public double HRV_Std { get; init; }
            public double HR_Mean { get; init; }
            public double HR_Std { get; init; }
            public double Steps_Mean { get; init; }
            public double Steps_Std { get; init; }
            public double SleepDuration_Mean { get; init; }
            public double SleepDuration_Std { get; init; }
            public double SleepQuality_Mean { get; init; }
            public double SleepQuality_Std { get; init; }
            public double ActivityLevel_Mean { get; init; }
            public double ActivityLevel_Std { get; init; }

            public double Label_Mean { get; init; }
        }
    }


}