# CLAUDE.md

Federated learning research project (HyFDCA) in C# / .NET 10. See README.md for the algorithm.

## Commands
- `dotnet build hyfed.sln`
- `dotnet run --project Coordinator` then `dotnet run --project Client`
- `dotnet run --project Tests` — solver smoke test; run after any change in `Core/Algorithm`
- `dotnet run --project Evaluation` — ablation study (slow; reads data from absolute paths in `Evaluation/Program.cs`)

## Layout
- Math lives in `Core` only; `Coordinator` and `Client` are thin gRPC wrappers
- gRPC contract: `Communication/Protos/fl_service.proto` — changing it requires rebuilding both Coordinator and Client
- Feature set is configured in `Core/Configuration/fl-feature-config.json`
- Dimensions must match: global dim (Coordinator `AggregatorConfig.GlobalFeatureDim`, Client `GlobalDim`) and local dim (`LocalDim`)

## Rules
- Linear algebra through MathNet.Numerics `Vector<double>`/`Matrix<double>`; no hand-rolled loops for vector math
- Differential privacy: every update leaving a client must go through `GaussianMechanism.Privatize` with clipping
- Keep results reproducible: pass explicit seeds to randomised components in experiments
- Do not commit bin/obj or large generated data sets
