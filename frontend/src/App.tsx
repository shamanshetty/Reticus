import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Moon, Sun } from 'lucide-react';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { useTheme } from '@/lib/theme';
import SoftAurora from '@/components/SoftAurora';
import PillNav from '@/components/PillNav';
import ReticusLogo from '@/components/ReticusLogo';
import GradientText from '@/components/GradientText';
import { PIPELINE, type PageKey } from '@/lib/pipeline';
import DataAnalysis from '@/pages/DataAnalysis';
import Preprocessing from '@/pages/Preprocessing';
import Training from '@/pages/Training';
import Results from '@/pages/Results';
import Inference from '@/pages/Inference';
import GradCAM from '@/pages/GradCAM';
import Compare from '@/pages/Compare';
import History from '@/pages/History';

type RunSummary = {
  run_id: string;
  display_name?: string;
  status: string;
  started_at?: string | null;
  ended_at?: string | null;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
};

type ProjectPayload = {
  project: string;
  runs: RunSummary[];
  latest_run: string | null;
  dataset: Record<string, unknown>;
  dataset_flags: unknown[];
};

const dotColor = (s: string) => (s === 'running' ? 'bg-emerald' : s === 'completed' ? 'bg-teal' : 'bg-slate');

export default function App() {
  const { theme, toggle } = useTheme();
  const dark = theme === 'dark';

  const navItems = useMemo(
    () => PIPELINE.map((p) => ({ key: p.key, label: p.label })),
    []
  );
  const [data, setData] = useState<ProjectPayload | null>(null);
  const [selectedRun, setSelectedRun] = useState<string>('');
  // null = on the navigator (sphere) surface; otherwise a page is open.
  const [activePage, setActivePage] = useState<PageKey | null>(null);

  useEffect(() => {
    const label = PIPELINE.find((p) => p.key === activePage)?.label;
    document.title = label ? `Reticus - ${label}` : 'Reticus';
  }, [activePage]);

  useEffect(() => {
    fetch('/api/project')
      .then((r) => r.json())
      .then((payload: ProjectPayload) => {
        setData(payload);
        const initial = payload.latest_run ?? payload.runs[0]?.run_id ?? '';
        setSelectedRun(initial);
      })
      .catch(() => {
        setData({ project: 'reticus', runs: [], latest_run: null, dataset: {}, dataset_flags: [] });
      });
  }, []);

  const runStatus = useMemo(() => {
    if (!data || !selectedRun) return 'idle';
    return data.runs.find((r) => r.run_id === selectedRun)?.status ?? 'idle';
  }, [data, selectedRun]);

  const runId = selectedRun || data?.latest_run || '';
  const project = data?.project ?? 'reticus';
  // The friendly name a user set via History's rename - falls back to the
  // raw run_id when no display_name was ever set. Every page should show
  // this instead of the raw id, or renaming a run only "works" in two places.
  const runLabel = data?.runs.find((r) => r.run_id === runId)?.display_name || runId;

  const page = (() => {
    switch (activePage) {
      case 'analysis':
        return (
          <DataAnalysis
            project={project}
            runId={runId}
            runLabel={runLabel}
            dataset={data?.dataset ?? {}}
            datasetFlags={(data?.dataset_flags ?? []) as any[]}
          />
        );
      case 'preprocessing':
        return <Preprocessing project={project} runId={runId} runLabel={runLabel} />;
      case 'training':
        return <Training project={project} runId={runId} runLabel={runLabel} runs={data?.runs ?? []} />;
      case 'results':
        return <Results project={project} runId={runId} runLabel={runLabel} />;
      case 'inference':
        return <Inference project={project} runId={runId} runLabel={runLabel} />;
      case 'gradcam':
        return <GradCAM project={project} runId={runId} runLabel={runLabel} />;
      case 'compare':
        return <Compare project={project} runs={data?.runs ?? []} />;
      case 'history':
        return (
          <History
            project={project}
            runs={data?.runs ?? []}
            onOpenRun={(rid, page) => { setSelectedRun(rid); setActivePage(page); }}
            onRenamed={(rid, displayName) => {
              setData((prev) => prev && {
                ...prev,
                runs: prev.runs.map((r) => (r.run_id === rid ? { ...r, display_name: displayName } : r)),
              });
            }}
          />
        );
      default:
        return null;
    }
  })();

  return (
    <TooltipProvider delayDuration={200}>
      <div className="relative flex min-h-screen flex-col bg-background text-foreground">
        {/* Ambient backdrop — quiet teal/prussian wash, fixed behind everything. */}
        <div
          className={cn('pointer-events-none fixed inset-0 z-0', dark ? 'opacity-60' : 'opacity-40')}
          aria-hidden
        >
          <SoftAurora
            key={theme}
            enableMouseInteraction={false}
            brightness={dark ? 0.7 : 1.1}
            speed={0.3}
            color1={dark ? '#40B5AD' : '#2f8f88'}
            color2={dark ? '#0a4d68' : '#1f6f86'}
          />
        </div>
        {/* Vignette — edges only, so the aurora stays visible mid-screen. */}
        <div
          className="pointer-events-none fixed inset-0 z-0 bg-gradient-to-b from-background/70 via-transparent to-background/85"
          aria-hidden
        />

        {/* Top bar — logo, pill nav, run selector, theme toggle. */}
        <header className="relative z-20 flex items-center gap-4 border-b border-soft bg-background/50 px-5 py-2.5 backdrop-blur-md lg:px-8">
          <PillNav
            items={navItems}
            activeKey={activePage ?? undefined}
            onSelect={(k) => setActivePage(k as PageKey)}
            onLogoClick={() => setActivePage(null)}
            logo={<ReticusLogo size={30} wordmarkClassName="text-[17px]" />}
            baseColor="#40B5AD"
            pillTextColor={dark ? '#e7ecf3' : '#1d2735'}
            hoveredPillTextColor={dark ? '#0c1116' : '#102330'}
          />

          <div className="ml-auto flex items-center gap-2.5">
            <div className="hidden w-[220px] sm:block">
              <Select value={selectedRun} onValueChange={setSelectedRun}>
                <SelectTrigger className="w-full">
                  <span className="flex items-center gap-2 truncate">
                    {selectedRun && <span className={cn('h-2 w-2 shrink-0 rounded-full', dotColor(runStatus))} />}
                    <SelectValue placeholder="No runs yet" />
                  </span>
                </SelectTrigger>
                <SelectContent>
                  {(data?.runs ?? []).length === 0 && (
                    <SelectItem value="__none__" disabled>
                      No runs yet
                    </SelectItem>
                  )}
                  {(data?.runs ?? []).map((run) => (
                    <SelectItem key={run.run_id} value={run.run_id}>
                      <span className="flex items-center gap-2">
                        <span className={cn('h-2 w-2 shrink-0 rounded-full', dotColor(run.status))} />
                        {run.display_name || run.run_id}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <button
              type="button"
              onClick={toggle}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-soft text-muted-foreground transition-colors hover:text-foreground"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            </button>
          </div>
        </header>

        {/* Main surface */}
        <main className="relative z-10 flex-1">
          <AnimatePresence mode="wait">
            {activePage === null ? (
              <motion.div
                key="navigator"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="mx-auto w-full max-w-[1100px] px-5 py-10 lg:px-8 lg:py-16"
              >
                <div className="text-center">
                  <p className="page-header-eyebrow">Local CV pipeline</p>
                  <h1 className="font-display mx-auto mt-2 max-w-3xl text-[34px] font-semibold leading-[1.05] lg:text-[48px]">
                    Inspect every stage of your run with{' '}
                    <GradientText animationSpeed={7}>Reticus</GradientText>
                  </h1>
                  <p className="mx-auto mt-3 max-w-lg text-[14px] leading-relaxed text-muted-foreground">
                    Dataset, training, results, inference — on one machine. No account, no cloud;
                    data never leaves this box.
                  </p>
                </div>

                <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {PIPELINE.map((p, i) => (
                    <motion.button
                      key={p.key}
                      type="button"
                      onClick={() => setActivePage(p.key)}
                      initial={{ opacity: 0, y: 14 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.4, delay: i * 0.05, ease: [0.22, 1, 0.36, 1] }}
                      className="reticus-card reticus-card-hover group relative overflow-hidden p-5 text-left"
                    >
                      <span
                        className="absolute inset-y-0 left-0 w-[3px]"
                        style={{ background: p.color }}
                        aria-hidden
                      />
                      <p className="font-mono text-[11px] uppercase tracking-[0.16em]" style={{ color: p.color }}>
                        {p.tag}
                      </p>
                      <h3 className="font-display mt-2 text-[20px] font-semibold tracking-tight">{p.label}</h3>
                      <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">{p.blurb}</p>
                      <span className="mt-4 inline-flex items-center gap-1 text-[12px] font-medium text-teal opacity-0 transition-opacity group-hover:opacity-100">
                        Open →
                      </span>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key={activePage}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.24, ease: 'easeOut' }}
                className="mx-auto w-full max-w-[1200px] px-5 py-6 lg:px-8"
              >
                {page}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </TooltipProvider>
  );
}
