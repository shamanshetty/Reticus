import { useCallback, useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { plotLayout, fmtPct } from '@/lib/utils';
import { useTheme } from '@/lib/theme';
import { Upload, Download, ImageOff, Loader2, Sparkles, AlertTriangle } from 'lucide-react';

type Props = { project: string; runId: string; runLabel?: string };

type ModelInfo = {
  live_predictor: boolean;
  downloadable: boolean;
  format?: string;
  class_names: string[];
  ready: boolean;
  used_default_preprocess: boolean;
  model_name?: string | null;
  source?: string | null;
  registered_at?: string | null;
  registered_run_id?: string | null;
};

type InferResult = {
  pred_class_name: string;
  confidence: number;
  top_k: Array<{ class_name: string; prob: number }>;
  width: number;
  height: number;
};

export default function Inference({ project, runId, runLabel }: Props) {
  useTheme();
  const PLOT_LAYOUT = plotLayout();
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<InferResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const loadInfo = useCallback(() => {
    fetch('/api/model/info').then((r) => r.json()).then(setInfo).catch(() => setInfo(null));
  }, []);

  useEffect(() => {
    loadInfo();
    const t = window.setInterval(loadInfo, 5000);
    return () => window.clearInterval(t);
  }, [loadInfo]);

  const runInference = useCallback((file: File) => {
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    const form = new FormData();
    form.append('file', file);
    fetch('/api/inference?top_k=10', { method: 'POST', body: form })
      .then(async (r) => {
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail || `Request failed (${r.status})`);
        }
        return r.json();
      })
      .then((d: InferResult) => setResult(d))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const onFile = (f: File | undefined) => { if (f) runInference(f); };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    onFile(e.dataTransfer.files?.[0]);
  };

  const topk = result?.top_k ?? [];
  const barData = [{
    type: 'bar' as const,
    orientation: 'h' as const,
    x: [...topk].reverse().map((t) => t.prob),
    y: [...topk].reverse().map((t) => t.class_name),
    marker: { color: [...topk].reverse().map((_, i, a) => (i === a.length - 1 ? '#40B5AD' : 'rgba(112,128,144,0.5)')) },
    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
  }];

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="page-header-eyebrow">Inference</p>
          <h1 className="page-header-title">Try the Model</h1>
          <p className="mt-1 text-sm text-muted-foreground">{project} · {runLabel || runId || 'no run selected'}</p>
        </div>
        <div className="flex items-center gap-2">
          {info?.live_predictor ? (
            <Badge variant="success">Predictor live</Badge>
          ) : (
            <Badge variant="secondary">No live predictor</Badge>
          )}
          {info?.downloadable && (
            <Button variant="outline" size="sm" asChild>
              <a href="/api/model/download">
                <Download className="mr-1.5 h-4 w-4" /> Download model
              </a>
            </Button>
          )}
        </div>
      </div>

      {/* Model identity card — confirms exactly which model/run is currently loaded */}
      {info?.live_predictor && (
        <div className="reticus-card grid grid-cols-2 gap-4 p-4 sm:grid-cols-4">
          <div>
            <p className="reticus-label">Model</p>
            <p className="mt-1 truncate text-sm font-medium text-foreground" title={info.model_name ?? undefined}>
              {info.model_name || 'Unknown'}
            </p>
          </div>
          <div>
            <p className="reticus-label">Registered via</p>
            <p className="mt-1 truncate text-sm font-medium text-foreground">
              {info.source === 'register_torch_classifier' ? 'register_torch_classifier' : info.source === 'register_predictor' ? 'register_predictor' : '—'}
            </p>
          </div>
          <div>
            <p className="reticus-label">From run</p>
            <p className="mt-1 truncate text-sm font-medium text-foreground">
              {info.registered_run_id || '—'}
              {runId && info.registered_run_id && info.registered_run_id !== runId && (
                <span className="ml-1.5 text-[11px] font-normal text-amber">(≠ selected run)</span>
              )}
            </p>
          </div>
          <div>
            <p className="reticus-label">Registered at</p>
            <p className="mt-1 truncate text-sm font-medium text-foreground">
              {info.registered_at ? new Date(info.registered_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—'}
            </p>
          </div>
        </div>
      )}

      {/* Loaded model belongs to a different run than the one selected in the nav bar */}
      {info?.live_predictor && runId && info.registered_run_id && info.registered_run_id !== runId && (
        <div className="reticus-card border-amber/40 bg-amber/5 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber" />
            <div className="space-y-1 text-sm">
              <p className="font-medium text-foreground">Loaded model is from a different run</p>
              <p className="text-muted-foreground">
                The live predictor was registered during <code className="text-teal">{info.registered_run_id}</code>,
                but you're currently viewing <code className="text-teal">{runId}</code>. Predictions below come
                from whatever model is loaded in the running process — not necessarily this run's model.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Live but using the generic fallback preprocess → may not match production */}
      {info?.live_predictor && info.used_default_preprocess && (
        <div className="reticus-card border-amber/40 bg-amber/5 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber" />
            <div className="space-y-1 text-sm">
              <p className="font-medium text-foreground">Using a generic preprocess, not necessarily your model's</p>
              <p className="text-muted-foreground">
                <code className="text-teal">register_torch_classifier</code> was called without a{' '}
                <code className="text-teal">preprocess</code> argument, so a default resize→ImageNet-normalize
                pipeline is being applied to uploads. If your training pipeline used different resizing,
                cropping, or normalization, predictions here will not match a production deployment of this
                model. Pass your own <code className="text-teal">preprocess(pil_image) -&gt; tensor</code> to fix this.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* No predictor → guidance. Distinguish "never registered" from "was
          registered by a process that has since exited" (e.g. viewing via
          `reticus <project>` after the training script ended) — the model
          identity is still known from disk even though nothing is live. */}
      {info && !info.live_predictor && (
        <div className="reticus-card p-5">
          <div className="flex items-start gap-3">
            <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-amber" />
            <div className="space-y-2 text-sm">
              {info.model_name ? (
                <p className="font-medium text-foreground">
                  A predictor ({info.model_name}) was registered previously, but isn't loaded in this process.
                </p>
              ) : (
                <p className="font-medium text-foreground">No predictor is registered yet.</p>
              )}
              <p className="text-muted-foreground">
                Inference runs your model live, in the process that started Reticus
                {info.model_name ? ' — reopen or rerun that script to make it live again, or' : '. While your script is running,'} register a predictor:
              </p>
              <pre className="overflow-x-auto rounded-lg border border-border bg-secondary p-3 text-xs leading-relaxed text-foreground">
{`# simplest — any callable that maps a PIL image to class probabilities
r.register_predictor(lambda img: model(transform(img)), class_names=CLASSES)

# or, for a torch classifier (handles softmax + preprocessing):
r.register_torch_classifier(model, class_names=CLASSES)`}
              </pre>
              <p className="text-muted-foreground">
                Keep the script alive (e.g. <code className="text-teal">input("press enter to exit")</code>)
                to keep using this tab after training finishes.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-12">
        {/* Upload / preview */}
        <div className="space-y-4 lg:col-span-5">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            className={[
              'reticus-card flex cursor-pointer flex-col items-center justify-center gap-2 border-2 border-dashed p-8 text-center transition-colors',
              dragging ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/40',
            ].join(' ')}
          >
            <Upload className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">Drop an image or click to upload</p>
            <p className="text-xs text-muted-foreground">PNG, JPG, WebP</p>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
          </div>

          {preview && (
            <div className="reticus-card overflow-hidden">
              <img src={preview} alt="upload preview" className="max-h-[360px] w-full object-contain bg-secondary" />
            </div>
          )}
        </div>

        {/* Result */}
        <div className="space-y-4 lg:col-span-7">
          {loading && (
            <div className="reticus-card flex items-center justify-center gap-2 p-12 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" /> Running inference…
            </div>
          )}

          {error && (
            <div className="reticus-card flex items-start gap-2 p-4 text-sm text-crimson">
              <ImageOff className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div className="reticus-card relative overflow-hidden px-5 py-4">
                <span className="absolute inset-y-0 left-0 w-1 bg-primary" />
                <p className="reticus-label">Prediction</p>
                <div className="mt-1 flex items-baseline gap-3">
                  <span className="text-2xl font-semibold tracking-tight text-foreground">{result.pred_class_name}</span>
                  <span className="text-sm font-medium tabular-nums text-teal">{fmtPct(result.confidence)}</span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{result.width}×{result.height}px</p>
              </div>

              <div className="reticus-card p-4">
                <p className="reticus-label mb-2">Top-{topk.length} Probabilities</p>
                <Plot
                  data={barData as any}
                  layout={{
                    ...PLOT_LAYOUT,
                    height: Math.max(180, topk.length * 28 + 60),
                    margin: { l: 110, r: 24, t: 8, b: 32 },
                    xaxis: { ...PLOT_LAYOUT.xaxis, range: [0, 1] },
                  } as any}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </motion.div>
          )}

          {!result && !loading && !error && (
            <div className="reticus-card p-12 text-center text-sm text-muted-foreground">
              Upload an image to see predictions.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
