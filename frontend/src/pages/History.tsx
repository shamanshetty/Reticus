import { useMemo, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Check, Pencil, X } from 'lucide-react';
import type { PageKey } from '@/lib/pipeline';

type RunRow = {
  run_id: string;
  display_name?: string;
  status: string;
  started_at?: string | null;
  ended_at?: string | null;
  config: Record<string, any>;
  metrics: Record<string, any>;
};

type Props = {
  project: string;
  runs: RunRow[];
  onOpenRun: (runId: string, page: PageKey) => void;
  onRenamed: (runId: string, displayName: string) => void;
};

const dotColor = (s: string) => (s === 'running' ? 'bg-emerald' : s === 'completed' ? 'bg-teal' : 'bg-slate');

function fmtDateTime(iso?: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function fmtDuration(start?: string | null, end?: string | null): string {
  if (!start) return '—';
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  if (isNaN(s) || isNaN(e) || e < s) return '—';
  const sec = Math.round((e - s) / 1000);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

export default function History({ project, runs, onOpenRun, onRenamed }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);

  const sorted = useMemo(
    () => [...runs].sort((a, b) => (b.started_at ?? '').localeCompare(a.started_at ?? '')),
    [runs]
  );

  const startRename = (row: RunRow) => {
    setEditing(row.run_id);
    setDraft(row.display_name || row.run_id);
  };

  const cancelRename = () => {
    setEditing(null);
    setDraft('');
  };

  const saveRename = async (runId: string) => {
    const name = draft.trim();
    if (!name) return;
    setSaving(true);
    try {
      const res = await fetch(`/api/runs/${runId}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: name }),
      });
      if (res.ok) onRenamed(runId, name);
    } finally {
      setSaving(false);
      setEditing(null);
    }
  };

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">History</p>
          <h1 className="page-header-title">All Runs</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {sorted.length} run{sorted.length === 1 ? '' : 's'}</p>
        </div>
      </div>

      {sorted.length === 0 ? (
        <div className="reticus-card p-12 text-center text-sm text-muted-foreground">
          No runs yet. Call <code className="text-teal">r.start_run(...)</code> to begin one.
        </div>
      ) : (
        <div className="reticus-card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-4 py-3 font-medium">Run</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Task</th>
                <th className="px-4 py-3 font-medium">Started</th>
                <th className="px-4 py-3 font-medium">Duration</th>
                <th className="px-4 py-3 font-medium">Headline metric</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => {
                const task = row.config?.task ?? '—';
                const overall = row.metrics?.overall ?? {};
                const metricVal = task === 'od' ? overall.map50 : overall.accuracy;
                const metricLabel = task === 'od' ? 'mAP@50' : 'Accuracy';
                const isEditing = editing === row.run_id;

                return (
                  <tr
                    key={row.run_id}
                    className="border-b border-border/60 last:border-0 transition-colors hover:bg-secondary/40"
                  >
                    <td className="px-4 py-3">
                      {isEditing ? (
                        <div className="flex items-center gap-1.5">
                          <Input
                            autoFocus
                            value={draft}
                            onChange={(e) => setDraft(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveRename(row.run_id);
                              if (e.key === 'Escape') cancelRename();
                            }}
                            className="h-7 w-44 text-sm"
                            maxLength={200}
                          />
                          <Button size="icon" variant="ghost" className="h-7 w-7" disabled={saving} onClick={() => saveRename(row.run_id)}>
                            <Check className="h-3.5 w-3.5" />
                          </Button>
                          <Button size="icon" variant="ghost" className="h-7 w-7" disabled={saving} onClick={cancelRename}>
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          className="group flex items-center gap-1.5 text-left font-medium text-foreground hover:text-teal"
                          onClick={() => onOpenRun(row.run_id, task === 'od' || task === 'classification' ? 'results' : 'results')}
                        >
                          {row.display_name || row.run_id}
                          <Pencil
                            className="h-3 w-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                            onClick={(e) => { e.stopPropagation(); startRename(row); }}
                          />
                        </button>
                      )}
                      {row.display_name && row.display_name !== row.run_id && (
                        <p className="mt-0.5 text-[11px] text-muted-foreground">{row.run_id}</p>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-1.5">
                        <span className={`h-2 w-2 rounded-full ${dotColor(row.status)}`} />
                        <Badge variant={row.status === 'completed' ? 'success' : row.status === 'running' ? 'default' : 'secondary'}>
                          {row.status}
                        </Badge>
                      </span>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{task}</td>
                    <td className="px-4 py-3 text-muted-foreground">{fmtDateTime(row.started_at)}</td>
                    <td className="px-4 py-3 text-muted-foreground">{fmtDuration(row.started_at, row.ended_at)}</td>
                    <td className="px-4 py-3 tabular-nums">
                      {metricVal != null ? `${metricLabel}: ${(Number(metricVal) * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Button size="sm" variant="outline" onClick={() => onOpenRun(row.run_id, 'results')}>
                        Open
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
