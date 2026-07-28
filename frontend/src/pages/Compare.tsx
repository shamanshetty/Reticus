import { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn, fmt3, plotLayout } from '@/lib/utils';
import { useTheme } from '@/lib/theme';
import MetricCard from '@/components/MetricCard';

const COLOR_A = '#40B5AD'; // Run A — teal
const COLOR_B = '#FFB703'; // Run B — amber

type RunSummary = { run_id: string; display_name?: string; config: Record<string, unknown> };
type Props = { project: string; runs: RunSummary[] };

export default function Compare({ project, runs }: Props) {
  useTheme();
  const PLOT_LAYOUT_BASE = plotLayout();
  const [runA, setRunA] = useState('');
  const [runB, setRunB] = useState('');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // runA/runB stay as raw run_ids (used for API calls + React keys); this
  // resolves the friendly name set via History's rename for display only.
  const labelFor = (id: string) => runs.find((r) => r.run_id === id)?.display_name || id;
  const labelA = labelFor(runA);
  const labelB = labelFor(runB);

  useEffect(() => {
    if (runs.length >= 2) {
      setRunA(runs[0].run_id);
      setRunB(runs[1].run_id);
    }
  }, [runs]);

  useEffect(() => {
    if (!runA || !runB || runA === runB) return;
    setLoading(true);
    fetch(`/api/compare?run_a=${runA}&run_b=${runB}`)
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [runA, runB]);

  const epochsA: any[] = data?.epochs_a ?? [];
  const epochsB: any[] = data?.epochs_b ?? [];
  const xA = epochsA.map((r: any) => r.epoch);
  const xB = epochsB.map((r: any) => r.epoch);

  const lossTraces = [
    { x: xA, y: epochsA.map((r: any) => r.train_loss), name: `${labelA} train`, type: 'scatter', mode: 'lines', line: { color: COLOR_A, dash: 'solid', width: 2.5, shape: 'spline' } },
    { x: xA, y: epochsA.map((r: any) => r.val_loss), name: `${labelA} val`, type: 'scatter', mode: 'lines', line: { color: COLOR_A, dash: 'dot', width: 1.5, shape: 'spline' } },
    { x: xB, y: epochsB.map((r: any) => r.train_loss), name: `${labelB} train`, type: 'scatter', mode: 'lines', line: { color: COLOR_B, dash: 'solid', width: 2.5, shape: 'spline' } },
    { x: xB, y: epochsB.map((r: any) => r.val_loss), name: `${labelB} val`, type: 'scatter', mode: 'lines', line: { color: COLOR_B, dash: 'dot', width: 1.5, shape: 'spline' } },
  ];

  const metricTraces = [
    { x: xA, y: epochsA.map((r: any) => r.val_acc ?? r.val_map), name: `${labelA}`, type: 'scatter', mode: 'lines', line: { color: COLOR_A, width: 2.5, shape: 'spline' } },
    { x: xB, y: epochsB.map((r: any) => r.val_acc ?? r.val_map), name: `${labelB}`, type: 'scatter', mode: 'lines', line: { color: COLOR_B, width: 2.5, shape: 'spline' } },
  ];

  const overallA = data?.metrics_a?.overall ?? {};
  const overallB = data?.metrics_b?.overall ?? {};

  return (
    <section className="space-y-4">
      <div>
        <p className="page-header-eyebrow">Compare</p>
        <h1 className="page-header-title">Run Comparison</h1>
        <p className="mt-1 text-sm text-muted-foreground">{project}</p>
      </div>

      {/* Run selectors */}
      <div className="reticus-card p-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR_A }} />
          <span className="text-sm text-muted-foreground">Run A:</span>
          <Select value={runA} onValueChange={setRunA}>
            <SelectTrigger className="w-36"><SelectValue /></SelectTrigger>
            <SelectContent>
              {runs.map((r) => <SelectItem key={r.run_id} value={r.run_id}>{r.display_name || r.run_id}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR_B }} />
          <span className="text-sm text-muted-foreground">Run B:</span>
          <Select value={runB} onValueChange={setRunB}>
            <SelectTrigger className="w-36"><SelectValue /></SelectTrigger>
            <SelectContent>
              {runs.map((r) => <SelectItem key={r.run_id} value={r.run_id}>{r.display_name || r.run_id}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        {loading && <span className="text-xs text-muted-foreground animate-pulse">Loading…</span>}
      </div>

      {data && (
        <>
          {/* Summary */}
          {data.summary && (
            <div className="reticus-card p-4">
              <p className="text-sm font-medium text-foreground">{data.summary}</p>
            </div>
          )}

          {/* Overall metric cards */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <MetricCard label={`${labelA} Macro F1`} value={overallA.macro_f1 ?? overallA.map50} tone="accent" index={0} />
            <MetricCard label={`${labelB} Macro F1`} value={overallB.macro_f1 ?? overallB.map50} tone="good" index={1} />
            <MetricCard label={`${labelA} Accuracy`} value={overallA.accuracy ?? overallA.map50} tone="default" index={2} />
            <MetricCard label={`${labelB} Accuracy`} value={overallB.accuracy ?? overallB.map50} tone="default" index={3} />
          </div>

          {/* Curves side by side */}
          {(xA.length > 0 || xB.length > 0) && (
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="reticus-card p-4">
                <p className="reticus-label mb-3">Loss Curves</p>
                <Plot
                  data={lossTraces as any}
                  layout={{ ...PLOT_LAYOUT_BASE, height: 280 } as any}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
              <div className="reticus-card p-4">
                <p className="reticus-label mb-3">Val Accuracy / mAP</p>
                <Plot
                  data={metricTraces as any}
                  layout={{ ...PLOT_LAYOUT_BASE, height: 280 } as any}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}

          {/* Config diff */}
          {data.config_diff?.length > 0 && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">Config Changes</p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Parameter</TableHead>
                    <TableHead>{labelA}</TableHead>
                    <TableHead>{labelB}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.config_diff.map((row: any) => (
                    <TableRow key={row.key} className="bg-amber/5">
                      <TableCell className="font-medium text-amber">{row.key}</TableCell>
                      <TableCell className="text-muted-foreground">{String(row.run_a ?? '—')}</TableCell>
                      <TableCell className="font-medium text-foreground">{String(row.run_b ?? '—')}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {/* Per-class F1 delta */}
          {data.class_deltas?.length > 0 && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">Per-Class F1 / AP Delta ({labelA} → {labelB})</p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Class</TableHead>
                    <TableHead>{labelA}</TableHead>
                    <TableHead>{labelB}</TableHead>
                    <TableHead>Delta</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.class_deltas.map((row: any) => {
                    const improved = row.delta > 0.005;
                    const degraded = row.delta < -0.005;
                    return (
                      <TableRow key={row.class_name}>
                        <TableCell className="font-medium">{row.class_name}</TableCell>
                        <TableCell className="tabular-nums text-muted-foreground">{fmt3(row.run_a)}</TableCell>
                        <TableCell className="tabular-nums">{fmt3(row.run_b)}</TableCell>
                        <TableCell className={cn('tabular-nums font-semibold', improved ? 'text-emerald' : degraded ? 'text-crimson' : 'text-muted-foreground')}>
                          {row.delta > 0 ? '+' : ''}{fmt3(row.delta)}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </>
      )}

      {!data && !loading && runs.length < 2 && (
        <div className="reticus-card p-8 text-center">
          <p className="text-muted-foreground text-sm">Need at least 2 runs to compare.</p>
        </div>
      )}

      {runA && runB && runA === runB && (
        <div className="reticus-card p-8 text-center">
          <p className="text-muted-foreground text-sm">Pick two different runs to compare.</p>
        </div>
      )}
    </section>
  );
}
