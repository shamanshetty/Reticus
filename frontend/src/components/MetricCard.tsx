import { motion } from 'framer-motion';
import { cn, fmt3 } from '@/lib/utils';
import { useCountUp } from '@/hooks/useCountUp';
import Sparkline from '@/components/Sparkline';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Info } from 'lucide-react';

type Tone = 'default' | 'accent' | 'good' | 'warn' | 'bad';

type Props = {
  label: string;
  value: string | number | undefined | null;
  sublabel?: string;
  tone?: Tone;
  /** optional trend series rendered as an inline sparkline */
  spark?: Array<number | null | undefined>;
  /** optional explanation shown via a hover tooltip on the label */
  info?: string;
  /** stagger index for mount animation */
  index?: number;
};

const toneText: Record<Tone, string> = {
  default: 'text-foreground',
  accent: 'text-teal',
  good: 'text-emerald',
  warn: 'text-amber',
  bad: 'text-crimson',
};

const toneBar: Record<Tone, string> = {
  default: 'bg-slate/40',
  accent: 'bg-teal',
  good: 'bg-emerald',
  warn: 'bg-amber',
  bad: 'bg-crimson',
};

const toneSpark: Record<Tone, string> = {
  default: '#708090',
  accent: '#40B5AD',
  good: '#50C878',
  warn: '#FFB703',
  bad: '#DC143C',
};

/** Animated number — counts up when the value is numeric. */
function AnimatedNumber({ value }: { value: number }) {
  const live = useCountUp(value);
  return <>{fmt3(live)}</>;
}

export default function MetricCard({ label, value, sublabel, tone = 'default', spark, info, index = 0 }: Props) {
  const isNum = typeof value === 'number' && isFinite(value);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.04, ease: 'easeOut' }}
      className="reticus-card reticus-card-hover relative overflow-hidden px-4 py-3.5"
    >
      <span className={cn('absolute inset-y-0 left-0 w-1', toneBar[tone])} aria-hidden />
      <div className="flex items-start justify-between gap-2 pl-1.5">
        <div className="flex min-w-0 flex-col gap-1.5">
          <span className="reticus-label inline-flex items-center gap-1">
            {label}
            {info && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button type="button" className="text-muted-foreground/60 transition-colors hover:text-foreground">
                    <Info className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>{info}</TooltipContent>
              </Tooltip>
            )}
          </span>
          <span className={cn('text-[28px] font-bold tabular-nums leading-none tracking-tight', toneText[tone])}>
            {isNum ? <AnimatedNumber value={value as number} /> : (value ?? '—')}
          </span>
          {sublabel && <span className="text-xs text-muted-foreground">{sublabel}</span>}
        </div>
        {spark && spark.filter((v) => typeof v === 'number').length >= 2 && (
          <div className="shrink-0 self-end">
            <Sparkline data={spark} color={toneSpark[tone]} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
