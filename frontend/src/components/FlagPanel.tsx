import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { cn, severityBorder } from '@/lib/utils';
import { AlertOctagon, AlertTriangle, Info, CheckCircle2 } from 'lucide-react';

type Flag = {
  flag_id: string;
  severity: string;
  message: string;
  suggestion: string;
  stage: string;
};

type Props = {
  flags: Flag[];
  title?: string;
};

const severityVariant = (s: string): 'destructive' | 'warning' | 'default' => {
  if (s === 'CRITICAL') return 'destructive';
  if (s === 'WARNING') return 'warning';
  return 'default';
};

const SeverityIcon = ({ severity }: { severity: string }) => {
  const cls = 'mt-0.5 h-4 w-4 shrink-0';
  if (severity === 'CRITICAL') return <AlertOctagon className={cn(cls, 'text-crimson')} />;
  if (severity === 'WARNING') return <AlertTriangle className={cn(cls, 'text-amber')} />;
  return <Info className={cn(cls, 'text-teal')} />;
};

export default function FlagPanel({ flags, title = 'Flags' }: Props) {
  if (!flags || flags.length === 0) {
    return (
      <div className="reticus-card p-4">
        <p className="reticus-label mb-3">{title}</p>
        <div className="flex items-center gap-2 rounded-lg border border-emerald/20 bg-emerald/10 px-3 py-2.5 text-sm text-emerald">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>No flags raised.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="reticus-card p-4">
      <p className="reticus-label mb-3">{title} ({flags.length})</p>
      <div className="space-y-2">
        {flags.map((flag, idx) => (
          <motion.div
            key={`${flag.flag_id}-${idx}`}
            initial={{ opacity: 0, x: 6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25, delay: Math.min(idx, 10) * 0.04, ease: 'easeOut' }}
            className={cn(
              'rounded-lg border border-border border-l-[3px] bg-card px-3 py-2.5 shadow-sm',
              severityBorder(flag.severity)
            )}
          >
            <div className="flex items-start gap-2">
              <SeverityIcon severity={flag.severity} />
              <div className="min-w-0">
                <div className="mb-1 flex items-center gap-2">
                  <Badge variant={severityVariant(flag.severity)} className="text-[10px]">
                    {flag.severity}
                  </Badge>
                </div>
                <p className="text-xs font-medium leading-snug text-foreground">{flag.message}</p>
                {flag.suggestion && (
                  <p className="mt-1 text-xs leading-snug text-muted-foreground">{flag.suggestion}</p>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
