using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Core.Models;

namespace Core.DataLoading
{
    public sealed class PatientDataLoader
    {
        private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

        public PatientData LoadFromFile(string filePath)
        {
            var json = File.ReadAllText(filePath);
            return LoadFromJson(json);
        }

        public PatientData LoadFromJson(string json)
        {
            var dto = JsonSerializer.Deserialize<PatientJsonDto>(json, JsonOptions) ?? throw new InvalidDataException($"Failed to serialize patient JSON");

            return BuildPatient(dto);
        }

        public IReadOnlyList<PatientData> LoadFromDirectory(string directoryPath)
        {
            var files = Directory.GetFiles(directoryPath, "*.json");
            var result = new List<PatientData>(files.Length);

            foreach (var file in files)
            {
                try
                {
                    result.Add(LoadFromFile(file));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Skipping {Path.GetFileName(file)}: {ex.Message}");
                }
            }

            return result;
        }

        private static PatientData BuildPatient(PatientJsonDto dto)
        {


            var validRecords = dto.Records.Where(r => !r.HasBiometricGap)
            .Where(r => IsComplete(r))
            .OrderBy(r => r.DayIndex)
            .ToList();
            int totalDays = validRecords.Count;

            var samples = validRecords.Select(r => MapToSample(dto.PatientId, r)).ToList();

            var buffer = new WindowBuffer();
            var deltaFeatures = samples.Select(s => buffer.Enqueue(s)).ToList();

            return new PatientData
            {
                PatientId = dto.PatientId,
                Trajectory = dto.Trajectory,
                Diagnosis = dto.PrimaryDiagnosis,
                ConfidenceVector = dto.ConfidenceVector,
                Samples = samples,
                DeltaFeatures = deltaFeatures,
                TotalSimulationDays = totalDays
            };
        }

        private static bool IsComplete(RecordJsonDto r) =>
        r.HRV.HasValue &&
        r.HeartRateAvg.HasValue &&
        r.Steps.HasValue &&
        r.SleepDuration.HasValue &&
        r.SleepQuality.HasValue &&
        r.ActivityLevel.HasValue &&
        r.CriticalityScore.HasValue;

        private static BiometricSample MapToSample(string patientId, RecordJsonDto r) =>
        new()
        {
            PatientId = patientId,
            DayIndex = r.DayIndex,
            Date = DateOnly.FromDateTime(r.Date),
            HRV = r.HRV!.Value,
            HR = r.HeartRateAvg!.Value,
            Steps = r.Steps!.Value,
            SleepDuration = r.SleepDuration!.Value,
            SleepQuality = r.SleepQuality!.Value,
            ActivityLevel = r.ActivityLevel!.Value,
            CriticalityScore = r.CriticalityScore!.Value
        };
    }
}