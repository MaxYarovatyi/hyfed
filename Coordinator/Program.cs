using Coordinator.Services;
using Core.Algorithm;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi

builder.Services.AddGrpc(options =>
{
    options.EnableDetailedErrors = true;
});

builder.Services.AddSingleton(new AggregatorConfig
{
    GlobalFeatureDim = 6,
    ConvergenceTolerance = 1e-4,
    ConvergenceRoundsRequired = 3,
    MaxRounds = 5
});

builder.Services.AddSingleton(new CoordinatorConfig
{
    MinClientsPerRound = 5,
});

builder.Services.AddSingleton(sp =>
    new GlobalAggregator(sp.GetRequiredService<AggregatorConfig>()));

builder.Services.AddSingleton<FLCoordinatorService>();

builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenLocalhost(5050, o => o.Protocols = Microsoft.AspNetCore.Server.Kestrel.Core.HttpProtocols.Http2);
});


var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

//app.UseHttpsRedirection();
app.MapGrpcService<FLCoordinatorService>();

app.MapGet("/", () => "FL coordinator running. gRPC on port 5050");

Console.WriteLine("FL coordinator starting on http://localhost:5050");
app.Run();
