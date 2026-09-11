using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace Core.Configuration
{
    public sealed class FeatureConfigLoader
    {
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNameCaseInsensitive = true,
            Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
        };

        public FeatureConfig LoadFromFile(string filePath)
        {
            var json = File.ReadAllText(filePath);
            var config = JsonSerializer.Deserialize<FeatureConfig>(json, JsonOptions) ?? throw new InvalidDataException("Failed to deserialize feature config");

            Validate(config);
            return config;
        }

        private static void Validate(FeatureConfig config)
        {
            ValidateNoDuplicateIds(config);
            ValidateTransformedSourceNames(config);
            ValidateImpactMatrixAlignment(config);
        }

        private static void ValidateNoDuplicateIds(FeatureConfig config)
        {
            var allIds = config.DeltaFeatures.Select(f => f.Id).Concat(config.TransformFeatures.Select(f => f.Id)).ToList();

            var duplicates = allIds.GroupBy(id => id)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

            if (duplicates.Count > 0)
                throw new InvalidDataException($"Duplicate features IDs in config: {string.Join(", ", duplicates)}");
        }

        private static void ValidateImpactMatrixAlignment(FeatureConfig config)
        {
            var enabledIds = config.EnabledDeltaFeatures.Select(f => f.Id)
            .Concat(config.EnabledTransformFeatures.Select(f => f.Id))
            .ToHashSet();

            var matrixIds = config.ImpactMatrix.FeatureOrder.ToHashSet();

            var missingFromMatrix = enabledIds.Except(matrixIds).ToList();
            if (missingFromMatrix.Count > 0)
                throw new InvalidDataException($"Enabled features missing form ImpactMatrix.featureOrder: " + $"{string.Join(", ", missingFromMatrix)}");

            // var extraInMatrix = matrixIds.Except(enabledIds).ToList();
            // if (extraInMatrix.Count > 0)
            //     throw new InvalidDataException($"ImpactMatrix.featureOrder contains disabled/unknown features: " + $"{string.Join(", ", extraInMatrix)}");

            int expectedCols = config.ImpactMatrix.FeatureOrder.Length;
            foreach (var (diagnosis, row) in config.ImpactMatrix.Rows)
            {
                if (row.Length != expectedCols)
                    throw new InvalidDataException($"ImpactMatrix row '{diagnosis}' has {row.Length} columns, expected {expectedCols}.");
            }

            foreach (var diagnosis in config.ImpactMatrix.Diagnoses)
            {
                if (!config.ImpactMatrix.Rows.ContainsKey(diagnosis))
                    throw new InvalidDataException($"Impact matrix missing row for diagnosis '{diagnosis}'");
            }
        }

        private static void ValidateTransformedSourceNames(FeatureConfig config)
        {
            var validSources = new HashSet<string>
            {
                "HRV", "HR", "Steps", "SleepDuration", "SleepQuality", "ActivityLevel"
            };

            foreach (var transformed in config.TransformFeatures.Where(f => f.Enabled))
            {
                if (!validSources.Contains(transformed.SourceA))
                    throw new InvalidDataException($"Transformed feature '{transformed.Id}' has unknown sourceA '{transformed.SourceA}'."
                    + $"Valid: {string.Join(", ", validSources)}");

                if (transformed.Type == FeatureConfig.TransformedFeatureType.Interaction && !validSources.Contains(transformed.SourceB))
                    throw new InvalidDataException($"Transformed feature '{transformed.Id}' has unknown sourceB '{transformed.SourceB}'" +
                    $"Valid: {string.Join(", ", validSources)}");
            }
        }
    }
}