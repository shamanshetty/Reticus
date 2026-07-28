import { useMemo } from 'react';

type Props = {
  data: Array<number | null | undefined>;
  color?: string;
  width?: number;
  height?: number;
  /** draw a soft area fill under the line */
  fill?: boolean;
  className?: string;
};

/**
 * Tiny inline trend line rendered as SVG (no Plotly). Auto-scales to its
 * own min/max. Renders nothing for <2 valid points.
 */
export default function Sparkline({
  data,
  color = '#40B5AD',
  width = 72,
  height = 28,
  fill = true,
  className,
}: Props) {
  const { line, area, dot } = useMemo(() => {
    const pts = data.map((d) => (typeof d === 'number' && isFinite(d) ? d : null));
    const valid = pts.filter((v): v is number => v !== null);
    if (valid.length < 2) return { line: '', area: '', dot: null as null | [number, number] };

    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const span = max - min || 1;
    const pad = 2;
    const innerW = width - pad * 2;
    const innerH = height - pad * 2;
    const n = pts.length;

    const coords: Array<[number, number]> = [];
    pts.forEach((v, i) => {
      if (v === null) return;
      const x = pad + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
      const y = pad + innerH - ((v - min) / span) * innerH;
      coords.push([x, y]);
    });

    const linePath = coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    const last = coords[coords.length - 1];
    const first = coords[0];
    const areaPath = `${linePath} L${last[0].toFixed(1)},${(height - pad).toFixed(1)} L${first[0].toFixed(1)},${(height - pad).toFixed(1)} Z`;

    return { line: linePath, area: areaPath, dot: last };
  }, [data, width, height]);

  if (!line) return null;

  const gradId = `spark-${color.replace('#', '')}`;

  return (
    <svg width={width} height={height} className={className} aria-hidden>
      {fill && (
        <>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.18" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={area} fill={`url(#${gradId})`} stroke="none" />
        </>
      )}
      <path d={line} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      {dot && <circle cx={dot[0]} cy={dot[1]} r={1.8} fill={color} />}
    </svg>
  );
}
