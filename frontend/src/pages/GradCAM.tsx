import { useEffect, useState } from 'react';
import ImageGrid from '@/components/ImageGrid';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useRunStream } from '@/hooks/useRunStream';
import { Loader2 } from 'lucide-react';

type Props = { project: string; runId: string; runLabel?: string };

export default function GradCAM({ project, runId, runLabel }: Props) {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [filterClass, setFilterClass] = useState('__all__');
  const { status } = useRunStream(runId);

  // Refetch on phase change: gradcam data lands after the tab first opens.
  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    fetch(`/api/runs/${runId}/gradcam`)
      .then((r) => r.json())
      .then((d) => { setItems(d); setLoading(false); })
      .catch(() => { setItems([]); setLoading(false); });
  }, [runId, status.phase, status.status]);

  const classNames = [...new Set(items.map((i: any) => i.class_name))];

  const filtered = items.filter((i: any) =>
    filterClass === '__all__' || i.class_name === filterClass
  );

  const gridItems = filtered.map((item: any) => ({
    img: `/static/${(showOverlay ? item.img_overlay : item.img_original ?? item.img_overlay ?? '').replace(/\\/g, '/')}`,
    title: item.class_name,
    subtitle: `Method: ${item.method}${item.fallback ? ' (approx)' : ''}`,
    metadata: {
      'true class': item.class_name,
      method: item.method,
      index: item.idx,
    },
  }));

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">Grad-CAM</p>
          <h1 className="page-header-title">Activation Maps</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {runLabel || runId || 'no run selected'}</p>
        </div>
      </div>

      <div className="reticus-card p-3 flex flex-wrap items-center gap-3">
        <Select value={filterClass} onValueChange={setFilterClass}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="All classes" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All classes</SelectItem>
            {classNames.map((c) => <SelectItem key={c as string} value={c as string}>{c as string}</SelectItem>)}
          </SelectContent>
        </Select>

        <Button
          variant={showOverlay ? 'default' : 'outline'}
          size="sm"
          onClick={() => setShowOverlay((v) => !v)}
        >
          {showOverlay ? 'Overlay' : 'Original'}
        </Button>

        <span className="text-xs text-muted-foreground ml-auto">{filtered.length} images</span>
      </div>

      {loading ? (
        <div className="reticus-card flex items-center justify-center gap-2 p-12 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Loading…
        </div>
      ) : gridItems.length > 0 ? (
        <div className="reticus-card p-4">
          <ImageGrid items={gridItems} cols={4} />
        </div>
      ) : (
        <div className="reticus-card p-8 text-center">
          <p className="text-muted-foreground text-sm">
            No Grad-CAM data yet. Call <code className="text-teal">cv.log_gradcam()</code> during training.
          </p>
        </div>
      )}
    </section>
  );
}
