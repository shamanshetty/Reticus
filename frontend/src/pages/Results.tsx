import { useCallback, useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import FlagPanel from '@/components/FlagPanel';
import MetricCard from '@/components/MetricCard';
import ImageGrid from '@/components/ImageGrid';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import RunHeaderBar from '@/components/RunHeaderBar';
import KpiHero from '@/components/KpiHero';
import { fmt3, f1BadgeVariant, plotLayout } from '@/lib/utils';
import { metricInfo } from '@/lib/metricInfo';
import { useTheme } from '@/lib/theme';
import { useRunStream } from '@/hooks/useRunStream';
import { ArrowUpDown, ChevronLeft, ChevronRight, Info, Target, Gauge, Layers, Crosshair, Loader2 } from 'lucide-react';

type SortKey = 'class_name' | 'precision' | 'recall' | 'f1' | 'ap50' | 'support';
type Tab = 'metrics' | 'predictions' | 'explore';

type Props = { project: string; runId: string; runLabel?: string };

export default function Results({ project, runId, runLabel }: Props) {
  useTheme(); // re-render on theme toggle so chart layout recolors
  const PLOT_LAYOUT_BASE = plotLayout();
  const [metrics, setMetrics] = useState<any>({});
  const [flags, setFlags] = useState<any[]>([]);
  const [prCurves, setPrCurves] = useState<any[]>([]);
  const [preds, setPreds] = useState<any>({ items: [], total: 0, total_pages: 1, page: 1 });
  const [errors, setErrors] = useState<any>({ items: [], total_errors: 0, task: 'classification' });
  const [thr, setThr] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<Tab>('metrics');
  const [confNorm, setConfNorm] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>('f1');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [page, setPage] = useState(1);
  const [filterClass, setFilterClass] = useState<string>('__all__');
  const [filterCorrect, setFilterCorrect] = useState<string>('__all__');
  const [confRange, setConfRange] = useState<number[]>([0, 1]);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const { status } = useRunStream(runId);

  // Re-fetch on phase/status change, not just runId: predictions and
  // gradcam are written after this tab may already be open.
  useEffect(() => {
    if (!runId) return;
    setLoadingMetrics(true);
    Promise.allSettled([
      fetch(`/api/runs/${runId}/metrics`).then((r) => r.json()).then(setMetrics).catch(() => setMetrics({})),
      fetch(`/api/runs/${runId}/flags`).then((r) => r.json()).then(setFlags).catch(() => setFlags([])),
      fetch(`/api/runs/${runId}/pr_curves`).then((r) => r.json()).then(setPrCurves).catch(() => setPrCurves([])),
    ]).then(() => setLoadingMetrics(false));
  }, [runId, status.phase, status.status]);

  const fetchPreds = useCallback(() => {
    if (!runId) return;
    const params = new URLSearchParams({ page: String(page) });
    if (filterClass !== '__all__') params.set('filter_class', filterClass);
    if (filterCorrect !== '__all__') params.set('filter_correct', filterCorrect === 'correct' ? 'true' : 'false');
    params.set('conf_min', String(confRange[0]));
    params.set('conf_max', String(confRange[1]));
    fetch(`/api/runs/${runId}/predictions?${params}`).then((r) => r.json()).then(setPreds).catch(() => {});
  }, [runId, page, filterClass, filterCorrect, confRange]);

  useEffect(() => { if (activeTab === 'predictions') fetchPreds(); }, [activeTab, fetchPreds, status.phase, status.status]);

  useEffect(() => {
    if (activeTab !== 'explore' || !runId) return;
    fetch(`/api/runs/${runId}/errors?limit=48`).then((r) => r.json()).then(setErrors).catch(() => {});
    fetch(`/api/runs/${runId}/threshold`).then((r) => r.json()).then(setThr).catch(() => setThr(null));
  }, [activeTab, runId, status.phase, status.status]);

  const overall = metrics.overall ?? {};
  const classNames: string[] = metrics.class_names ?? [];
  const confMatrix: number[][] = confNorm ? (metrics.confusion_matrix_normalized ?? []) : (metrics.confusion_matrix_raw ?? []);
  const task: string = metrics.task ?? 'classification';

  const sortedPerClass = useMemo(() => {
    const rows = [...(metrics.per_class ?? [])];
    rows.sort((a, b) => {
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      if (typeof va === 'string') return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortDir === 'asc' ? Number(va) - Number(vb) : Number(vb) - Number(va);
    });
    return rows;
  }, [metrics.per_class, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(key); setSortDir('desc'); }
  };

  const prTraces = prCurves.map((c: any) => ({
    x: c.recall,
    y: c.precision,
    type: 'scatter',
    mode: 'lines',
    name: c.class_name,
    line: { width: 1.5 },
  }));

  const f1Key = task === 'od' ? 'ap50' : 'f1';

  // Radar: per-class precision / recall / F1
  const perClass: any[] = metrics.per_class ?? [];
  const radarAxes = perClass.map((p) => p.class_name);
  const closed = (vals: number[]) => (vals.length ? [...vals, vals[0]] : vals);
  const radarTraces = perClass.length >= 3 ? [
    { type: 'scatterpolar', name: 'Precision', r: closed(perClass.map((p) => p.precision ?? 0)), theta: closed(radarAxes), fill: 'toself', fillcolor: 'rgba(64,181,173,0.14)', line: { color: '#40B5AD' } },
    { type: 'scatterpolar', name: 'Recall', r: closed(perClass.map((p) => p.recall ?? 0)), theta: closed(radarAxes), fill: 'toself', fillcolor: 'rgba(255,183,3,0.10)', line: { color: '#FFB703' } },
    { type: 'scatterpolar', name: task === 'od' ? 'AP50' : 'F1', r: closed(perClass.map((p) => p[f1Key] ?? 0)), theta: closed(radarAxes), fill: 'toself', fillcolor: 'rgba(80,200,120,0.08)', line: { color: '#50C878' } },
  ] : [];

  // ROC curves (need probs)
  const rocCurves: any[] = metrics.roc_curves ?? [];
  const rocTraces = rocCurves.map((c: any) => ({
    x: c.fpr, y: c.tpr, type: 'scatter', mode: 'lines', name: `${c.class_name} (AUC ${c.auc?.toFixed(3)})`, line: { width: 2 },
  }));

  // Calibration / reliability
  const calib = metrics.calibration ?? null;
  const confHist = metrics.confidence_hist ?? null;

  const predItems = (preds.items ?? []).map((row: any) => ({
    img: `/static/${(row.img ?? '').replace(/\\/g, '/')}`,
    title: row.true !== undefined
      ? `T: ${classNames[row.true] ?? row.true}  P: ${classNames[row.pred] ?? row.pred}`
      : `img ${row.idx}`,
    subtitle: row.correct === true ? '✓ Correct' : row.correct === false ? '✗ Wrong' : `IoU ${row.iou ?? ''}`,
    metadata: {
      confidence: row.conf ?? row.iou ?? '—',
      ...(row.true !== undefined ? { 'true label': classNames[row.true] ?? row.true } : {}),
      ...(row.pred !== undefined ? { 'pred label': classNames[row.pred] ?? row.pred } : {}),
    },
  }));

  const errorItems = (errors.items ?? []).map((row: any) => ({
    img: `/static/${(row.img ?? '').replace(/\\/g, '/')}`,
    title: row.true !== undefined
      ? `T: ${classNames[row.true] ?? row.true} → P: ${classNames[row.pred] ?? row.pred}`
      : `img ${row.idx}`,
    subtitle: row.conf !== undefined ? `conf ${fmt3(row.conf)}` : row.iou !== undefined ? `IoU ${fmt3(row.iou)}` : '',
    metadata: {
      ...(row.true !== undefined ? { 'true label': classNames[row.true] ?? row.true } : {}),
      ...(row.pred !== undefined ? { 'pred label': classNames[row.pred] ?? row.pred } : {}),
      ...(row.conf !== undefined ? { confidence: row.conf } : {}),
      ...(row.iou !== undefined ? { iou: row.iou } : {}),
    },
  }));

  // Click a confusion-matrix cell -> filter predictions to that class, jump to tab
  const onMatrixClick = (e: any) => {
    const pt = e?.points?.[0];
    if (!pt) return;
    const predName = String(pt.x); // x axis = predicted class
    const classIdx = classNames.indexOf(predName);
    if (classIdx < 0) return;
    setFilterClass(String(classIdx));
    setFilterCorrect('__all__');
    setPage(1);
    setActiveTab('predictions');
  };

  const headMetric = task === 'od' ? overall.map50 : overall.accuracy;
  const headLabel = task === 'od' ? 'mAP@50' : 'Accuracy';

  return (
    <section className="space-y-4">
      <RunHeaderBar
        runId={runLabel || runId}
        status="completed"
        metricLabel={headLabel}
        metricValue={headMetric}
        context={`${classNames.length} classes`}
      />

      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">Results</p>
          <h1 className="page-header-title">Metrics &amp; Predictions</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {runLabel || runId || 'no run selected'}</p>
        </div>
        <div className="flex gap-2">
          <Button variant={activeTab === 'metrics' ? 'default' : 'outline'} size="sm" onClick={() => setActiveTab('metrics')}>Metrics</Button>
          <Button variant={activeTab === 'predictions' ? 'default' : 'outline'} size="sm" onClick={() => setActiveTab('predictions')}>Predictions</Button>
          <Button variant={activeTab === 'explore' ? 'default' : 'outline'} size="sm" onClick={() => setActiveTab('explore')}>Explore</Button>
        </div>
      </div>

      {activeTab === 'metrics' && loadingMetrics && (
        <div className="reticus-card flex items-center justify-center gap-2 p-12 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Loading…
        </div>
      )}

      {activeTab === 'metrics' && !loadingMetrics && (
        <>
          <KpiHero
            items={[
              { label: task === 'od' ? 'mAP@50' : 'Accuracy', value: overall.accuracy ?? overall.map50, format: 'pct', icon: Target, info: metricInfo(task === 'od' ? 'map50' : 'accuracy'), hero: true },
              { label: 'Macro F1', value: overall.macro_f1, format: 'num', icon: Gauge, info: metricInfo('macro_f1') },
              { label: 'Weighted F1', value: overall.weighted_f1, format: 'num', icon: Layers, info: metricInfo('weighted_f1') },
              { label: task === 'od' ? 'mAP@50:95' : 'Micro F1', value: metrics.map50_95 ?? overall.micro_f1, format: 'num', icon: Crosshair, info: metricInfo(task === 'od' ? 'map50_95' : 'micro_f1') },
            ]}
          />

          {/* Advanced metrics row */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            {task === 'od' ? (
              <>
                <MetricCard label="Precision" value={overall.precision} tone="default" info={metricInfo('precision')} index={0} />
                <MetricCard label="Recall" value={overall.recall} tone="default" info={metricInfo('recall')} index={1} />
                <MetricCard label="Mean IoU" value={overall.mean_iou} tone="accent" info={metricInfo('mean_iou')} index={2} />
                <MetricCard label="False Pos" value={overall.fp} tone="warn" info={metricInfo('fp')} index={3} />
                <MetricCard label="False Neg" value={overall.fn} tone="bad" info={metricInfo('fn')} index={4} />
              </>
            ) : (
              <>
                <MetricCard label="Balanced Acc" value={overall.balanced_accuracy} tone="good" info={metricInfo('balanced_accuracy')} index={0} />
                <MetricCard label="Macro Precision" value={overall.macro_precision} tone="default" info={metricInfo('macro_precision')} index={1} />
                <MetricCard label="Macro Recall" value={overall.macro_recall} tone="default" info={metricInfo('macro_recall')} index={2} />
                <MetricCard label="Cohen κ" value={overall.cohen_kappa} tone="accent" info={metricInfo('cohen_kappa')} index={3} />
                <MetricCard label="MCC" value={overall.mcc} tone="accent" info={metricInfo('mcc')} index={4} />
              </>
            )}
          </div>

          <div className="grid gap-4 lg:grid-cols-12">
            <div className="space-y-4 lg:col-span-8">
              {confMatrix.length > 0 && (
                <div className="reticus-card p-4">
                  <div className="mb-1 flex items-center justify-between">
                    <p className="reticus-label inline-flex items-center gap-1">
                      Confusion Matrix
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button type="button" className="text-muted-foreground/60 transition-colors hover:text-foreground">
                            <Info className="h-3 w-3" />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          Rows = true class, columns = predicted. Click any cell to inspect those predictions.
                        </TooltipContent>
                      </Tooltip>
                    </p>
                    <Button variant="outline" size="sm" onClick={() => setConfNorm((v) => !v)}>
                      {confNorm ? 'Raw counts' : 'Normalised'}
                    </Button>
                  </div>
                  <p className="mb-2 text-[11px] text-muted-foreground">Click a cell to filter predictions →</p>
                  <Plot
                    data={[{
                      z: confMatrix,
                      x: classNames,
                      y: classNames,
                      type: 'heatmap',
                      colorscale: [[0, 'rgba(64,181,173,0.05)'], [0.5, 'rgba(64,181,173,0.45)'], [1, '#003153']],
                      showscale: true,
                      hovertemplate: 'true: %{y}<br>pred: %{x}<br>' + (confNorm ? 'rate: %{z:.2f}' : 'count: %{z}') + '<extra></extra>',
                    }] as any}
                    layout={{ ...PLOT_LAYOUT_BASE, height: 340, margin: { l: 80, r: 40, t: 10, b: 80 } } as any}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    onClick={onMatrixClick}
                  />
                </div>
              )}

              {sortedPerClass.length > 0 && (
                <div className="reticus-card p-4">
                  <p className="reticus-label mb-3">Per-Class Metrics</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {(['class_name', 'precision', 'recall', f1Key, 'support'] as SortKey[]).map((k) => (
                          <TableHead key={k}>
                            <button
                              className="flex items-center gap-1 hover:text-foreground transition-colors"
                              onClick={() => toggleSort(k)}
                            >
                              {k === f1Key ? (task === 'od' ? 'AP50' : 'F1') : k.replace('_', ' ')}
                              <ArrowUpDown className="h-3 w-3 opacity-50" />
                            </button>
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedPerClass.map((row: any) => {
                        const f1 = Number(row[f1Key] ?? 0);
                        return (
                          <TableRow key={row.class_name}>
                            <TableCell className="font-medium">{row.class_name}</TableCell>
                            <TableCell className="tabular-nums">{fmt3(row.precision)}</TableCell>
                            <TableCell className="tabular-nums">{fmt3(row.recall)}</TableCell>
                            <TableCell>
                              <Badge variant={f1BadgeVariant(f1) as any} className="tabular-nums">
                                {fmt3(f1)}
                              </Badge>
                            </TableCell>
                            <TableCell className="tabular-nums text-muted-foreground">{row.support}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              )}

              {prTraces.length > 0 && (
                <div className="reticus-card p-4">
                  <p className="reticus-label mb-3">Precision-Recall Curves</p>
                  <Plot
                    data={prTraces as any}
                    layout={{
                      ...PLOT_LAYOUT_BASE,
                      height: 260,
                      xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'Recall', range: [0, 1] },
                      yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: 'Precision', range: [0, 1] },
                    } as any}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              )}

              {radarTraces.length > 0 && (
                <div className="reticus-card p-4">
                  <p className="reticus-label mb-1">Per-Class Precision · Recall · {task === 'od' ? 'AP50' : 'F1'}</p>
                  <p className="mb-2 text-[11px] text-muted-foreground">Balanced classes fill the wheel evenly; dents reveal weak spots.</p>
                  <Plot
                    data={radarTraces as any}
                    layout={{
                      ...PLOT_LAYOUT_BASE,
                      height: 340,
                      polar: {
                        bgcolor: 'transparent',
                        radialaxis: { range: [0, 1], gridcolor: PLOT_LAYOUT_BASE.xaxis.gridcolor, color: PLOT_LAYOUT_BASE.font.color, tickfont: { size: 9 } },
                        angularaxis: { gridcolor: PLOT_LAYOUT_BASE.xaxis.gridcolor, color: PLOT_LAYOUT_BASE.font.color },
                      },
                      showlegend: true,
                    } as any}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>
            <div className="lg:col-span-4">
              <FlagPanel flags={flags} />
            </div>
          </div>
        </>
      )}

      {activeTab === 'predictions' && (
        <div className="space-y-4">
          {/* Filter bar */}
          <div className="reticus-card p-3 flex flex-wrap items-center gap-3">
            <Select value={filterClass} onValueChange={(v) => { setFilterClass(v); setPage(1); }}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="All classes" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All classes</SelectItem>
                {classNames.map((c, i) => <SelectItem key={c} value={String(i)}>{c}</SelectItem>)}
              </SelectContent>
            </Select>

            <Select value={filterCorrect} onValueChange={(v) => { setFilterCorrect(v); setPage(1); }}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All</SelectItem>
                <SelectItem value="correct">Correct only</SelectItem>
                <SelectItem value="wrong">Wrong only</SelectItem>
              </SelectContent>
            </Select>

            <div className="flex items-center gap-3 flex-1 min-w-[200px]">
              <span className="text-xs text-muted-foreground whitespace-nowrap">{task === 'od' ? 'IoU' : 'Conf'}: {confRange[0].toFixed(1)}–{confRange[1].toFixed(1)}</span>
              <Slider
                min={0} max={1} step={0.05}
                value={confRange}
                onValueChange={(v) => { setConfRange(v); setPage(1); }}
                className="flex-1"
              />
            </div>

            <span className="text-xs text-muted-foreground ml-auto">{preds.total} items</span>
          </div>

          <ImageGrid items={predItems} cols={4} />

          {/* Pagination */}
          {preds.total_pages > 1 && (
            <div className="flex items-center justify-center gap-3">
              <Button variant="outline" size="sm" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1}>
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-sm text-muted-foreground">Page {preds.page} / {preds.total_pages}</span>
              <Button variant="outline" size="sm" onClick={() => setPage((p) => Math.min(preds.total_pages, p + 1))} disabled={page >= preds.total_pages}>
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'explore' && (
        <div className="space-y-4">
          {/* Threshold explorer (classification) */}
          {thr && thr.total > 0 && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-1">Confidence Threshold Explorer</p>
              <p className="mb-3 text-xs text-muted-foreground">
                As you raise the confidence cut-off, accuracy on the kept predictions rises but you
                cover fewer samples. Find the sweet spot.
              </p>
              <Plot
                data={[
                  { x: thr.thresholds, y: thr.accuracy, type: 'scatter', mode: 'lines+markers', name: 'accuracy (kept)', line: { color: '#40B5AD', width: 2.5, shape: 'spline' } },
                  { x: thr.thresholds, y: thr.coverage, type: 'scatter', mode: 'lines+markers', name: 'coverage', line: { color: '#FFB703', width: 2.5, shape: 'spline' } },
                ] as any}
                layout={{
                  ...PLOT_LAYOUT_BASE,
                  height: 300,
                  xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'Confidence threshold', range: [0, 1] },
                  yaxis: { ...PLOT_LAYOUT_BASE.yaxis, range: [0, 1.02] },
                } as any}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* ROC + Calibration */}
          {(rocTraces.length > 0 || calib) && (
            <div className="grid gap-4 lg:grid-cols-2">
              {rocTraces.length > 0 && (
                <div className="reticus-card p-4">
                  <p className="reticus-label mb-1">ROC Curves (one-vs-rest)</p>
                  <p className="mb-2 text-[11px] text-muted-foreground">Closer to the top-left = better. Diagonal = random.</p>
                  <Plot
                    data={[
                      ...rocTraces,
                      { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', name: 'chance', line: { dash: 'dot', color: 'rgba(150,150,150,0.5)', width: 1 }, showlegend: false },
                    ] as any}
                    layout={{
                      ...PLOT_LAYOUT_BASE, height: 320,
                      xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'False Positive Rate', range: [0, 1] },
                      yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: 'True Positive Rate', range: [0, 1.02] },
                    } as any}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
              {calib && (
                <div className="reticus-card p-4">
                  <div className="mb-1 flex items-center justify-between">
                    <p className="reticus-label">Reliability / Calibration</p>
                    {calib.ece != null && <span className="text-xs text-muted-foreground">ECE {fmt3(calib.ece)}</span>}
                  </div>
                  <p className="mb-2 text-[11px] text-muted-foreground">Bars on the diagonal = well-calibrated confidence.</p>
                  <Plot
                    data={[
                      { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', name: 'perfect', line: { dash: 'dot', color: 'rgba(150,150,150,0.5)', width: 1 } },
                      { x: calib.bin_confidence, y: calib.bin_accuracy, type: 'bar', name: 'accuracy', marker: { color: '#40B5AD' }, width: 0.045 },
                    ] as any}
                    layout={{
                      ...PLOT_LAYOUT_BASE, height: 320, showlegend: false,
                      xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'Confidence', range: [0, 1] },
                      yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: 'Accuracy', range: [0, 1.02] },
                    } as any}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>
          )}

          {/* Confidence histogram */}
          {confHist && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-2">Confidence Distribution (correct vs wrong)</p>
              <Plot
                data={[
                  { x: confHist.edges, y: confHist.correct, type: 'bar', name: 'correct', marker: { color: '#50C878' } },
                  { x: confHist.edges, y: confHist.wrong, type: 'bar', name: 'wrong', marker: { color: '#DC143C' } },
                ] as any}
                layout={{ ...PLOT_LAYOUT_BASE, height: 260, barmode: 'stack', xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'Max confidence', range: [0, 1] } } as any}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Error explorer */}
          <div className="reticus-card p-4">
            <div className="mb-1 flex items-center justify-between">
              <p className="reticus-label">Hardest Errors</p>
              <span className="text-xs text-muted-foreground">{errors.total_errors} shown</span>
            </div>
            <p className="mb-3 text-xs text-muted-foreground">
              {errors.task === 'od'
                ? 'Lowest-IoU predictions — where localization failed worst.'
                : 'Confident mistakes — wrong predictions the model was most sure about. Best place to find label noise and systematic failures.'}
            </p>
            {errorItems.length > 0 ? (
              <ImageGrid items={errorItems} cols={4} />
            ) : (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No errors to show (or predictions not logged). Call <code className="text-teal">r.log_predictions(...)</code>.
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
