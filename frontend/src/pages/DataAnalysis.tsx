import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import FlagPanel from '@/components/FlagPanel';
import MetricCard from '@/components/MetricCard';
import ImageGrid from '@/components/ImageGrid';
import { plotLayout, PLOT_COLORWAY } from '@/lib/utils';
import { metricInfo } from '@/lib/metricInfo';
import { useTheme } from '@/lib/theme';

const BAR_COLORS = PLOT_COLORWAY;

type Props = {
  project: string;
  runId: string;
  runLabel?: string;
  dataset: Record<string, any>;
  datasetFlags: any[];
};

export default function DataAnalysis({ project, runId, runLabel, dataset, datasetFlags }: Props) {
  useTheme();
  const PLOT_LAYOUT_BASE = plotLayout();
  const classNames = Object.keys((dataset.class_distribution?.train ?? dataset.class_distribution?.all ?? {}));

  const distributionTraces = useMemo(() => {
    const dist = dataset.class_distribution ?? {};
    return Object.entries(dist).map(([split, counts], i) => ({
      x: Object.keys(counts as object),
      y: Object.values(counts as object),
      type: 'bar' as const,
      name: split,
      marker: { color: BAR_COLORS[i % BAR_COLORS.length] },
    }));
  }, [dataset]);

  const sizeStats = dataset.image_size_stats ?? {};

  const sampleItems = useMemo(() => {
    const grid = dataset.sample_grid ?? {};
    return Object.entries(grid).flatMap(([cls, paths]) =>
      (paths as string[]).slice(0, 3).map((p) => ({
        img: `/static/${p.replace(/\\/g, '/')}`,
        title: cls,
        subtitle: p.split(/[\\/]/).pop() ?? '',
      }))
    );
  }, [dataset]);

  const isOd = dataset.task === 'od';
  // image_size_stats.count is a direct file-system scan - correct for any
  // dataset layout. class_distribution sums to the same value only for
  // classification's subdirectory-per-class layout; for OD it holds box
  // counts, not image counts, so it must not be used as the image total.
  const totalImages = dataset.image_size_stats?.count ?? 0;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">Data Analysis</p>
          <h1 className="page-header-title">Dataset Overview</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {runLabel || runId || 'no run selected'}</p>
        </div>
        <span className="reticus-chip">{dataset.task ?? 'classification'}</span>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Total Images" value={totalImages || '—'} tone="accent" index={0} />
        <MetricCard label="Classes" value={classNames.length || '—'} tone="default" index={1} />
        <MetricCard label="Duplicates" value={dataset.duplicate_count ?? '—'} tone="warn" info={metricInfo('duplicate_count')} index={2} />
        <MetricCard label="Corrupted" value={dataset.corrupted_count ?? '—'} tone="bad" info={metricInfo('corrupted_count')} index={3} />
      </div>

      <div className="grid gap-4 lg:grid-cols-12">
        <div className="space-y-4 lg:col-span-8">
          {distributionTraces.length > 0 && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">{isOd ? 'Box Count per Class' : 'Class Distribution'}</p>
              <Plot
                data={distributionTraces as any}
                layout={{
                  ...PLOT_LAYOUT_BASE,
                  barmode: 'group',
                  height: 300,
                  yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: isOd ? 'annotated boxes' : 'images' },
                } as any}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {isOd && (dataset.od_low_box_classes?.length > 0 || dataset.od_degenerate_count > 0) && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-2">Detection Annotation Health</p>
              <div className="space-y-1.5 text-sm">
                {dataset.od_low_box_classes?.length > 0 && (
                  <p className="text-muted-foreground">
                    Low box count (&lt;100 annotations): <span className="text-foreground font-medium">{dataset.od_low_box_classes.join(', ')}</span>
                  </p>
                )}
                {dataset.od_degenerate_count > 0 && (
                  <p className="text-crimson">
                    {dataset.od_degenerate_count} degenerate box{dataset.od_degenerate_count === 1 ? '' : 'es'} found (zero or negative width/height in label files).
                  </p>
                )}
              </div>
            </div>
          )}

          {sizeStats.count > 0 && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">Image Size Statistics</p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-sm">
                <div><span className="text-muted-foreground">Count: </span>{sizeStats.count}</div>
                <div><span className="text-muted-foreground">Min: </span>{sizeStats.min?.join('×')}</div>
                <div><span className="text-muted-foreground">Max: </span>{sizeStats.max?.join('×')}</div>
                <div><span className="text-muted-foreground">Mean: </span>{sizeStats.mean?.map((v: number) => Math.round(v)).join('×')}</div>
              </div>
            </div>
          )}

          {sampleItems.length > 0 && (
            <div className="reticus-card p-4">
              <p className="reticus-label mb-3">Sample Grid</p>
              <ImageGrid items={sampleItems} cols={4} />
            </div>
          )}
        </div>

        <div className="space-y-4 lg:col-span-4">
          <FlagPanel flags={datasetFlags} title="Dataset Flags" />
        </div>
      </div>
    </section>
  );
}
