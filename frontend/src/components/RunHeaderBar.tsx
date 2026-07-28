import { Badge } from '@/components/ui/badge';
import { cn, fmt3 } from '@/lib/utils';

type Props = {
  runId: string;
  status?: string;
  phase?: string;
  /** headline metric label + value, e.g. "Macro F1" / 0.82 */
  metricLabel?: string;
  metricValue?: number | null;
  /** right-aligned context, e.g. "epoch 7 / 10" or "12m remaining" */
  context?: string;
};

const dotColor = (s?: string) =>
  s === 'running' ? 'bg-emerald' : s === 'completed' ? 'bg-teal' : 'bg-slate';

const phaseVariant = (p?: string): 'success' | 'secondary' | 'warning' | 'default' => {
  if (p === 'completed') return 'success';
  if (p === 'preprocessing') return 'secondary';
  if (p === 'evaluation') return 'warning';
  return 'default';
};

const cap = (s?: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : '');

export default function RunHeaderBar({ runId, status, phase, metricLabel, metricValue, context }: Props) {
  return (
    <div className="sticky top-0 z-10 -mx-1 mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-border bg-card/85 px-4 py-2.5 shadow-sm backdrop-blur">
      <div className="flex items-center gap-2">
        <span className={cn('h-2 w-2 rounded-full', dotColor(status))} />
        <span className="font-mono text-sm font-medium text-foreground">{runId || 'no run'}</span>
      </div>

      {phase && <Badge variant={phaseVariant(phase)}>{cap(phase)}</Badge>}

      {metricLabel && metricValue != null && isFinite(metricValue) && (
        <div className="flex items-baseline gap-1.5">
          <span className="reticus-label">{metricLabel}</span>
          <span className="text-sm font-semibold tabular-nums text-foreground">{fmt3(metricValue)}</span>
        </div>
      )}

      {context && <span className="ml-auto text-xs text-muted-foreground">{context}</span>}
    </div>
  );
}
