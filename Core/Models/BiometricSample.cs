using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Core.Models
{
    public sealed class BiometricSample
    {
        public string PatientId { get; init; } = string.Empty;

        public int DayIndex { get; init; }
        public DateOnly Date { get; init; }


        // Global features | Shared across all clients and aggregated by coordinator
        public double HRV { get; init; }
        public double HR { get; init; }
        public double Steps { get; init; }
        public double SleepDuration { get; init; }
        public double SleepQuality { get; init; }
        public double ActivityLevel { get; init; }

        public double CriticalityScore { get; init; }

        public double[] ToRawArray() =>
        [HRV, HR, Steps, SleepDuration, SleepQuality, ActivityLevel];

        public override string ToString() =>
            $"[{Date}] Patient={PatientId} y={CriticalityScore:F3} " +
            $"HRV={HRV} HR={HR} Steps={Steps} Sleep={SleepDuration}h";
    }
}