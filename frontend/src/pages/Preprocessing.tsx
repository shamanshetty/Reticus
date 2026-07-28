import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Wand2 } from 'lucide-react';

type Step = { name: string; params: Record<string, unknown> };
type Sample = { idx: number; raw: string; processed?: string };
type Pipeline = { steps: Step[]; samples: Sample[] };

type Props = { project: string; runId: string; runLabel?: string };

export default function Preprocessing({ project, runId, runLabel }: Props) {
  const [pipeline, setPipeline] = useState<Pipeline>({ steps: [], samples: [] });

  useEffect(() => {
    fetch('/api/project/preprocessing_pipeline')
      .then((r) => r.json())
      .then(setPipeline)
      .catch(() => setPipeline({ steps: [], samples: [] }));
  }, []);

  const { steps, samples } = pipeline;
  const hasData = steps.length > 0 || samples.length > 0;
  const staticUrl = (p: string) => `/static/${p.replace(/\\/g, '/')}`;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">Preprocessing</p>
          <h1 className="page-header-title">Transform Pipeline</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {runLabel || runId || 'no run selected'}</p>
        </div>
      </div>

      {!hasData && (
        <div className="reticus-card p-5">
          <div className="flex items-start gap-3">
            <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-amber" />
            <div className="space-y-2 text-sm">
              <p className="font-medium text-foreground">No preprocessing pipeline logged yet.</p>
              <p className="text-muted-foreground">
                Call this once, before training starts, to describe your transform pipeline:
              </p>
              <pre className="overflow-x-auto rounded-lg border border-border bg-secondary p-3 text-xs leading-relaxed text-foreground">
{`r.log_preprocessing(
    steps=[
        {"name": "Resize", "params": {"size": 224}},
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406]}},
    ],
    samples=raw_images,            # optional — before/after gallery
    processed_samples=transformed_images,
)`}
              </pre>
            </div>
          </div>
        </div>
      )}

      {steps.length > 0 && (
        <div className="reticus-card p-4">
          <p className="reticus-label mb-3">Pipeline ({steps.length} step{steps.length === 1 ? '' : 's'})</p>
          <ol className="space-y-2">
            {steps.map((step, i) => (
              <motion.li
                key={i}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: i * 0.04 }}
                className="flex items-start gap-3 rounded-lg border border-border/60 bg-secondary/30 px-3 py-2.5"
              >
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-teal">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <span className="inline-flex items-center gap-1.5 text-sm font-medium text-foreground">
                    <Wand2 className="h-3.5 w-3.5 text-teal" /> {step.name}
                  </span>
                  {Object.keys(step.params ?? {}).length > 0 && (
                    <p className="mt-1 break-words text-[11px] text-muted-foreground">
                      {Object.entries(step.params).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  ·  ')}
                    </p>
                  )}
                </div>
              </motion.li>
            ))}
          </ol>
        </div>
      )}

      {samples.length > 0 && (
        <div className="reticus-card p-4">
          <div className="mb-3 flex items-center justify-between">
            <p className="reticus-label">Before / After ({samples.length} sample{samples.length === 1 ? '' : 's'})</p>
            {samples.every((s) => !s.processed) && (
              <span className="text-[11px] text-muted-foreground">
                Pass <code className="text-teal">processed_samples=...</code> to <code className="text-teal">log_preprocessing</code> to fill the "after" column.
              </span>
            )}
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {samples.map((s) => (
              <div key={s.idx} className="overflow-hidden rounded-lg border border-border bg-card">
                <div className="grid grid-cols-2 divide-x divide-border">
                  <div>
                    <p className="bg-secondary/60 px-2 py-1 text-center text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Before</p>
                    <div className="aspect-square overflow-hidden bg-secondary">
                      <img src={staticUrl(s.raw)} alt="raw" className="h-full w-full object-cover" loading="lazy" />
                    </div>
                  </div>
                  <div>
                    <p className="bg-secondary/60 px-2 py-1 text-center text-[10px] font-medium uppercase tracking-wide text-muted-foreground">After</p>
                    <div className="aspect-square overflow-hidden bg-secondary">
                      {s.processed ? (
                        <img src={staticUrl(s.processed)} alt="processed" className="h-full w-full object-cover" loading="lazy" />
                      ) : (
                        <div className="flex h-full items-center justify-center px-2 text-center text-[10px] text-muted-foreground">
                          not provided
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
