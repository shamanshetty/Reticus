import { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import FlagPanel from '@/components/FlagPanel';
import MetricCard from '@/components/MetricCard';
import RunHeaderBar from '@/components/RunHeaderBar';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { plotLayout } from '@/lib/utils';
import { metricInfo } from '@/lib/metricInfo';
import { useRunStream } from '@/hooks/useRunStream';
import { useTheme } from '@/lib/theme';
import { Radio } from 'lucide-react';

type Props = {
  project: string;
  runId: string;
  runLabel?: string;
  runs: Array<{ run_id: string; config: Record<string, unknown> }>;
};

export default function Training({ project, runId, runLabel, runs }: Props) {
  useTheme();
  const PLOT_LAYOUT_BASE = plotLayout();
  // Live epochs/status/flags via WebSocket (falls back to polling)
  const { epochs, status, flags, live } = useRunStream(runId);
  const [preprocessingProgress, setPreprocessingProgress] = useState<any[]>([]);
  const [evaluationProgress, setEvaluationProgress] = useState<any[]>([]);

  const totalEpochs = useMemo(() => {
    const run = runs.find((r) => r.run_id === runId);
    return Number(run?.config?.epochs ?? run?.config?.num_epochs ?? 0) || 0;
  }, [runs, runId]);

  useEffect(() => {
    if (!runId) return;
    const load = () => {
      fetch(`/api/project/preprocessing`).then((r) => r.json()).then(setPreprocessingProgress).catch(() => {});
      fetch(`/api/runs/${runId}/evaluation`).then((r) => r.json()).then(setEvaluationProgress).catch(() => {});
    };
    load();
    const interval = window.setInterval(load, 3000);
    return () => window.clearInterval(interval);
  }, [runId]);

  const last = epochs.length ? epochs[epochs.length - 1] : ({} as any);
  const currentEpoch = Number(last.epoch ?? 0);
  const progressPct = totalEpochs > 0 ? Math.min(100, (currentEpoch / totalEpochs) * 100) : 0;

  const etaStr = useMemo(() => {
    const eta = Number(last.eta_seconds);
    if (!eta || !isFinite(eta)) return null;
    if (eta < 60) return `${eta}s remaining`;
    if (eta < 3600) return `${Math.round(eta / 60)}m remaining`;
    return `${(eta / 3600).toFixed(1)}h remaining`;
  }, [last.eta_seconds]);

  const phase = status.phase ?? 'training';
  const phaseProgress = status.phase_progress;

  const fmtDateTime = (iso?: string | null) => {
    if (!iso) return null;
    const d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  };
  const startedStr = fmtDateTime(status.started_at);
  const endedStr = fmtDateTime(status.ended_at);

  const phaseVariant = (p: string) => {
    if (p === 'completed') return 'success' as const;
    if (p === 'preprocessing') return 'secondary' as const;
    if (p === 'training') return 'default' as const;
    if (p === 'evaluation') return 'warning' as const;
    return 'default' as const;
  };

  const phaseLabel = (p: string) => {
    if (p === 'preprocessing') return 'Preprocessing';
    if (p === 'training') return 'Training';
    if (p === 'evaluation') return 'Evaluation';
    if (p === 'completed') return 'Completed';
    return p;
  };

  // Preprocessing: last progress entry
  const lastPre = preprocessingProgress.length
    ? preprocessingProgress[preprocessingProgress.length - 1]
    : null;
  const preTotal = lastPre?.total_images ?? 0;
  const preScanned = lastPre?.images_scanned ?? 0;
  const prePct = preTotal > 0 ? Math.min(100, (preScanned / preTotal) * 100) : 0;

  // Evaluation: last progress entry
  const lastEval = evaluationProgress.length
    ? evaluationProgress[evaluationProgress.length - 1]
    : null;
  const evalTotal = lastEval?.total_predictions ?? 0;
  const evalDone = lastEval?.predictions_logged ?? 0;
  const evalPct = evalTotal > 0 ? Math.min(100, (evalDone / evalTotal) * 100) : 0;
  const evalStage = lastEval?.stage ?? 'predictions';
  const gradcamTotal = lastEval?.total_gradcam ?? 0;
  const gradcamDone = lastEval?.gradcam_generated ?? 0;
  const gradcamPct = gradcamTotal > 0 ? Math.min(100, (gradcamDone / gradcamTotal) * 100) : 0;

  // System metrics: pull from most recent available source
  const sysMetrics = phase === 'preprocessing' ? lastPre
    : phase === 'evaluation' ? lastEval
    : last;

  const x = epochs.map((r) => r.epoch);

  // Sparkline series (full history) for the stat strip
  const lrSpark = epochs.map((r) => r.lr);
  const gpuSpark = epochs.map((r) => r.gpu_mem_gb);
  const cpuSpark = epochs.map((r) => r.cpu_pct);
  const ramSpark = epochs.map((r) => r.ram_gb);

  // ── Training dynamics (derived from epoch timestamps + metrics) ──────────
  const tsList = epochs.map((r) => Number(r.ts)).filter((v) => isFinite(v) && v > 0);
  const epochDurations: number[] = [];
  for (let i = 1; i < tsList.length; i++) epochDurations.push(tsList[i] - tsList[i - 1]);
  const medianEpochSec = epochDurations.length
    ? [...epochDurations].sort((a, b) => a - b)[Math.floor(epochDurations.length / 2)]
    : null;
  const overfitGapSeries = epochs.map((r) => {
    const tr = r.train_acc ?? r.train_map;
    const va = r.val_acc ?? r.val_map;
    return tr != null && va != null ? Number(tr) - Number(va) : null;
  });
  const lastGap = overfitGapSeries.length ? overfitGapSeries[overfitGapSeries.length - 1] : null;
  const hasGap = overfitGapSeries.some((v) => v != null);

  // Headline metric for the sticky run bar
  const headMetric = last.val_acc ?? last.val_map;
  const headLabel = last.val_acc != null ? 'Val Acc' : last.val_map != null ? 'Val mAP' : undefined;
  const headContext =
    phase === 'training'
      ? `epoch ${currentEpoch || '—'}${totalEpochs > 0 ? ` / ${totalEpochs}` : ''}${etaStr ? ` · ${etaStr}` : ''}`
      : phase === 'preprocessing' && lastPre
      ? `${preScanned} / ${preTotal} scanned`
      : phase === 'evaluation' && lastEval
      ? (evalStage === 'gradcam' ? `GradCAM ${gradcamDone}/${gradcamTotal}` : `Predictions ${evalDone}/${evalTotal}`)
      : undefined;

  const lossTraces = [
    { x, y: epochs.map((r) => r.train_loss), type: 'scatter', mode: 'lines+markers', name: 'train loss', line: { color: '#40B5AD', width: 2.5, shape: 'spline' }, marker: { size: 5 } },
    { x, y: epochs.map((r) => r.val_loss), type: 'scatter', mode: 'lines+markers', name: 'val loss', line: { color: '#DC143C', width: 2.5, shape: 'spline' }, marker: { size: 5 } },
  ];

  const metricTraces = [
    { x, y: epochs.map((r) => r.train_acc ?? r.train_map), type: 'scatter', mode: 'lines+markers', name: 'train', line: { color: '#708090', width: 2.5, shape: 'spline' }, marker: { size: 5 } },
    { x, y: epochs.map((r) => r.val_acc ?? r.val_map), type: 'scatter', mode: 'lines+markers', name: 'val', line: { color: '#50C878', width: 2.5, shape: 'spline' }, marker: { size: 5 } },
  ];

  return (
    <section className="space-y-4">
      <RunHeaderBar
        runId={runLabel || runId}
        status={status.status}
        phase={phase}
        metricLabel={headLabel}
        metricValue={headMetric}
        context={headContext}
      />

      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">Training</p>
          <h1 className="page-header-title">Live Monitor</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {runLabel || runId || 'no run selected'}</p>
          {(startedStr || endedStr) && (
            <p className="mt-0.5 text-xs text-muted-foreground">
              {startedStr && <>Started {startedStr}</>}
              {startedStr && endedStr && ' · '}
              {endedStr && <>Ended {endedStr}</>}
            </p>
          )}
        </div>
        <div className="flex flex-col items-end gap-1.5">
          <div className="flex items-center gap-2">
            {live && (
              <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald">
                <Radio className="h-3.5 w-3.5 animate-pulse" /> Live
              </span>
            )}
            <Badge variant={phaseVariant(phase)}>{phaseLabel(phase)}</Badge>
            {phase === 'training' && (
              <span className="reticus-chip">
                Epoch {currentEpoch || '—'}{totalEpochs > 0 ? ` / ${totalEpochs}` : ''}
              </span>
            )}
          </div>
          {phase === 'training' && etaStr && (
            <span className="text-xs text-muted-foreground">{etaStr}</span>
          )}
          {phase === 'preprocessing' && lastPre && (
            <span className="text-xs text-muted-foreground">
              {preScanned} / {preTotal} images scanned
            </span>
          )}
          {phase === 'evaluation' && lastEval && (
            <span className="text-xs text-muted-foreground">
              {evalStage === 'gradcam'
                ? `GradCAM ${gradcamDone} / ${gradcamTotal}`
                : `Predictions ${evalDone} / ${evalTotal}`}
            </span>
          )}
        </div>
      </div>

      {/* Phase-specific progress bar */}
      {phase === 'preprocessing' && preTotal > 0 && (
        <div className="reticus-card p-4 space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Scanning dataset</span>
            <span>{prePct.toFixed(0)}%</span>
          </div>
          <Progress value={prePct} className="h-2" />
          {lastPre && (
            <div className="flex gap-4 text-xs text-muted-foreground pt-1">
              <span>Invalid: {lastPre.invalid_images ?? 0}</span>
              <span>Duplicates: {lastPre.duplicates_found ?? 0}</span>
            </div>
          )}
        </div>
      )}

      {phase === 'training' && totalEpochs > 0 && (
        <div className="reticus-card p-4 space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Progress</span>
            <span>{progressPct.toFixed(0)}%</span>
          </div>
          <Progress value={progressPct} className="h-2" />
        </div>
      )}

      {phase === 'evaluation' && (
        <div className="reticus-card p-4 space-y-3">
          {evalTotal > 0 && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Predictions</span>
                <span>{evalPct.toFixed(0)}%</span>
              </div>
              <Progress value={evalPct} className="h-2" />
            </div>
          )}
          {gradcamTotal > 0 && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>GradCAM</span>
                <span>{gradcamPct.toFixed(0)}%</span>
              </div>
              <Progress value={gradcamPct} className="h-2" />
            </div>
          )}
        </div>
      )}

      {/* System metrics — always visible, source depends on phase */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {phase === 'training' ? (
          <MetricCard label="Current LR" value={last.lr} tone="accent" info={metricInfo('lr')} spark={lrSpark} index={0} />
        ) : (
          <MetricCard label="Phase" value={phaseLabel(phase)} tone="accent" index={0} />
        )}
        <MetricCard label="GPU Mem GB" value={sysMetrics?.gpu_mem_gb} tone="default" info={metricInfo('gpu_mem_gb')} spark={gpuSpark} index={1} />
        <MetricCard label="CPU %" value={sysMetrics?.cpu_pct != null ? `${sysMetrics.cpu_pct}%` : '—'} tone="good" info={metricInfo('cpu_pct')} spark={cpuSpark} index={2} />
        <MetricCard label="RAM GB" value={sysMetrics?.ram_gb} tone="warn" info={metricInfo('ram_gb')} spark={ramSpark} index={3} />
      </div>

      {/* Training dynamics */}
      {epochs.length > 1 && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <MetricCard
            label="Time / Epoch"
            value={medianEpochSec != null ? `${medianEpochSec.toFixed(1)}s` : '—'}
            tone="default"
            info={metricInfo('time_per_epoch')}
            index={0}
          />
          {hasGap && (
            <MetricCard
              label="Overfit Gap"
              value={lastGap}
              tone={lastGap != null && lastGap > 0.15 ? 'bad' : lastGap != null && lastGap > 0.07 ? 'warn' : 'good'}
              info={metricInfo('overfit_gap')}
              spark={overfitGapSeries}
              index={1}
            />
          )}
          <MetricCard label="Epochs Logged" value={epochs.length} tone="accent" index={2} />
          {medianEpochSec != null && totalEpochs > 0 && currentEpoch < totalEpochs && (
            <MetricCard
              label="Proj. Total"
              value={`${((medianEpochSec * totalEpochs) / 60).toFixed(1)}m`}
              tone="default"
              index={3}
            />
          )}
        </div>
      )}

      {/* Charts — always show training curves when epochs available */}
      {(phase === 'training' || phase === 'completed' || epochs.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-12">
          <div className="space-y-4 lg:col-span-8">
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">Loss Curve</p>
              <Plot
                data={lossTraces as any}
                layout={{ ...PLOT_LAYOUT_BASE, height: 280 } as any}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">Accuracy / mAP</p>
              <Plot
                data={metricTraces as any}
                layout={{ ...PLOT_LAYOUT_BASE, height: 280 } as any}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          </div>
          <div className="lg:col-span-4">
            <FlagPanel flags={flags} />
          </div>
        </div>
      )}

      {/* Flags only (no charts) during preprocessing / empty eval */}
      {phase === 'preprocessing' && flags.length > 0 && (
        <FlagPanel flags={flags} />
      )}
    </section>
  );
}
