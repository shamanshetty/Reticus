import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function fmt3(value: number | string | undefined | null): string {
  if (value == null || value === '') return '—';
  const n = typeof value === 'string' ? parseFloat(value) : value;
  if (!isFinite(n)) return '—';
  return n.toFixed(3);
}

export function fmtPct(value: number | undefined | null): string {
  if (value == null || !isFinite(value)) return '—';
  return (value * 100).toFixed(1) + '%';
}

export function f1Color(f1: number): string {
  if (f1 >= 0.75) return 'text-emerald';
  if (f1 >= 0.5) return 'text-amber';
  return 'text-crimson';
}

export function f1BadgeVariant(f1: number): 'success' | 'warning' | 'destructive' {
  if (f1 >= 0.75) return 'success';
  if (f1 >= 0.5) return 'warning';
  return 'destructive';
}

export function severityColor(severity: string): string {
  switch (severity) {
    case 'CRITICAL': return 'text-crimson';
    case 'WARNING': return 'text-amber';
    case 'INFO': return 'text-teal';
    default: return 'text-muted-foreground';
  }
}

export function severityBorder(severity: string): string {
  switch (severity) {
    case 'CRITICAL': return 'border-l-crimson';
    case 'WARNING': return 'border-l-amber';
    case 'INFO': return 'border-l-teal';
    default: return 'border-l-border';
  }
}

/** Reticus chart palette — teal, amber, emerald, crimson, slate, prussian. */
export const PLOT_COLORWAY = ['#40B5AD', '#FFB703', '#50C878', '#DC143C', '#708090', '#2E6F8E', '#B5879B', '#C9A227'];

function isLight(): boolean {
  return typeof document !== 'undefined' && document.documentElement.classList.contains('light');
}

/**
 * Theme-aware Plotly base layout. Call inside render so it recomputes when the
 * theme toggles. One source of truth for all charts.
 */
export function plotLayout(): Record<string, any> {
  const light = isLight();
  const grid = light ? '#EAECF0' : 'rgba(255,255,255,0.07)';
  const zero = light ? '#D0D5DD' : 'rgba(255,255,255,0.14)';
  const axis = light ? '#667085' : '#A1A1AA';
  const text = light ? '#475467' : '#A1A1AA';
  return {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: text, family: 'Inter, system-ui, sans-serif', size: 12 },
    margin: { l: 52, r: 16, t: 16, b: 44 },
    xaxis: { gridcolor: grid, zerolinecolor: zero, linecolor: grid, color: axis },
    yaxis: { gridcolor: grid, zerolinecolor: zero, linecolor: grid, color: axis },
    legend: { bgcolor: 'transparent', font: { color: text } },
    colorway: PLOT_COLORWAY,
  };
}

/** Back-compat alias — a fresh themed layout snapshot. */
export const PLOT_LAYOUT = plotLayout();
