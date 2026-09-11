using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Core.Configuration
{
    public sealed class FeatureConfig
    {
        [JsonPropertyName("deltaFeatures")]
        public DeltaFeatureConfig[] DeltaFeatures { get; init; } = [];
        [JsonPropertyName("transformFeatures")]
        public TransformedFeatureConfig[] TransformFeatures { get; init; } = [];
        [JsonPropertyName("impactMatrix")]
        public ImpactMatrixConfig ImpactMatrix { get; init; } = new();


        public IEnumerable<DeltaFeatureConfig> EnabledDeltaFeatures => DeltaFeatures.Where(f => f.Enabled);

        public IEnumerable<TransformedFeatureConfig> EnabledTransformFeatures => TransformFeatures.Where(f => f.Enabled);

        public int LocalFeatureDimension => EnabledDeltaFeatures.Count() + EnabledTransformFeatures.Count();

        public sealed class DeltaFeatureConfig
        {
            [JsonPropertyName("id")]
            public string Id { get; init; } = string.Empty;
            [JsonPropertyName("description")]
            public string Description { get; init; } = string.Empty;
            [JsonPropertyName("enabled")]
            public bool Enabled { get; init; } = true;
        }

        [JsonConverter(typeof(JsonStringEnumConverter))]
        public enum TransformedFeatureType
        {
            Square,
            Interaction,
            Sqrt,
            Log1p
        }

        public sealed class TransformedFeatureConfig
        {
            [JsonPropertyName("id")]
            public string Id { get; init; } = string.Empty;
            [JsonPropertyName("description")]
            public string Description { get; init; } = string.Empty;
            [JsonPropertyName("type")]
            public TransformedFeatureType Type { get; init; }

            [JsonPropertyName("sourceA")]
            public string SourceA { get; init; } = string.Empty;
            [JsonPropertyName("sourceB")]
            public string SourceB { get; init; } = string.Empty;

            [JsonPropertyName("enabled")]
            public bool Enabled { get; init; } = true;
        }


        public sealed class ImpactMatrixConfig
        {
            [JsonPropertyName("diagnoses")]
            public string[] Diagnoses { get; init; } = [];

            [JsonPropertyName("featureOrder")]
            public string[] FeatureOrder { get; init; } = [];

            [JsonPropertyName("rows")]
            public Dictionary<string, double[]> Rows { get; init; } = [];
        }


    }
}