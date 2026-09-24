using Microsoft.Extensions.Hosting;

namespace NLaya.Extensions.AI;

/// <summary>Loads a registered agent or Router while the host starts (see <see cref="LayaSettings.Warmup"/>).</summary>
internal sealed class LayaWarmupService(Func<LayaSettings> settings, Action warmup) : IHostedService
{
    public Task StartAsync(CancellationToken cancellationToken) =>
        settings().Warmup ? Task.Run(warmup, cancellationToken) : Task.CompletedTask;

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
