import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { useCountUp } from '@/hooks/useCountUp';
import Sparkline from '@/components/Sparkline';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Info, type LucideIcon } from 'lucide-react';

export type Kpi = {
  label: string;
  value: number | string | null | undefined;
  /** format hint for numeric values */
  format?: 'num' | 'pct' | 'int' | 'raw';
  spark?: Array<number | null | undefined>;
  icon?: LucideIcon;
  info?: string;
  /** highlight as the primary hero stat (gradient number) */
  hero?: boolean;
};

function fmt(v: number, format: Kpi['format']): string {
  if (format === 'pct') return `${(v * 100).toFixed(1)}%`;
  if (format === 'int') return Math.round(v).toLocaleString();
  if (format === 'raw') return String(v);
  return v.toFixed(3);
}

function HeroNumber({ value, format, hero }: { value: number; format: Kpi['format']; hero?: boolean }) {
  const live = useCountUp(value);
  return (
    <span
      className={cn(
        'font-semibold tabular-nums leading-none tracking-tight',
        hero ? 'text-[40px] text-foreground' : 'text-[32px] text-foreground'
      )}
    >
      {fmt(live, format)}
    </span>
  );
}

export default function KpiHero({ items }: { items: Kpi[] }) {
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {items.map((k, i) => {
        const isNum = typeof k.value === 'number' && isFinite(k.value);
        const Icon = k.icon;
        return (
          <motion.div
            key={k.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.05, ease: [0.22, 1, 0.36, 1] }}
            className="reticus-card reticus-card-hover relative overflow-hidden p-5"
          >
            {/* left accent rule — amber for hero, teal otherwise */}
            <span
              className={cn('absolute inset-y-0 left-0 w-[3px]', k.hero ? 'bg-amber' : 'bg-teal/60')}
              aria-hidden
            />
            <div className="relative flex items-start justify-between gap-2 pl-1">
              <span className="reticus-label inline-flex items-center gap-1.5">
                {Icon && <Icon className={cn('h-3.5 w-3.5', k.hero ? 'text-amber' : 'text-teal')} />}
                {k.label}
                {k.info && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button type="button" className="text-muted-foreground/60 transition-colors hover:text-foreground">
                        <Info className="h-3 w-3" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>{k.info}</TooltipContent>
                  </Tooltip>
                )}
              </span>
              {k.spark && k.spark.filter((v) => typeof v === 'number').length >= 2 && (
                <Sparkline data={k.spark} color={k.hero ? '#FFB703' : '#40B5AD'} width={64} height={26} />
              )}
            </div>
            <div className="relative mt-3 pl-1">
              {isNum ? (
                <HeroNumber value={k.value as number} format={k.format} hero={k.hero} />
              ) : (
                <span className={cn('font-semibold leading-none', k.hero ? 'text-[36px] text-foreground' : 'text-[30px] text-foreground')}>
                  {k.value ?? '—'}
                </span>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
