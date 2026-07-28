import GradientText from '@/components/GradientText';

/**
 * Reticus mark — a reticular graph (nodes joined by edges, like a reticulum /
 * mesh). Hexagonal node ring around a central hub, evoking a connected pipeline.
 * Pure SVG, theme-agnostic (uses currentColor + accent).
 */

type ReticusLogoProps = {
  size?: number;
  showWordmark?: boolean;
  className?: string;
  /** scale the wordmark text */
  wordmarkClassName?: string;
};

const TEAL = '#40B5AD';
const NODES = [
  // central hub
  { x: 24, y: 24, r: 3.2, hub: true },
  // outer reticular ring (hexagon)
  { x: 24, y: 8, r: 2.2 },
  { x: 38, y: 16, r: 2.2 },
  { x: 38, y: 32, r: 2.2 },
  { x: 24, y: 40, r: 2.2 },
  { x: 10, y: 32, r: 2.2 },
  { x: 10, y: 16, r: 2.2 },
];

// edges: hub->each outer, plus ring between neighbours (the "net")
const EDGES: Array<[number, number]> = [
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
  [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1],
];

export default function ReticusLogo({
  size = 32,
  showWordmark = true,
  className,
  wordmarkClassName,
}: ReticusLogoProps) {
  return (
    <span className={className} style={{ display: 'inline-flex', alignItems: 'center', gap: showWordmark ? 10 : 0 }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 48 48"
        fill="none"
        aria-label="Reticus"
        role="img"
        style={{ flexShrink: 0 }}
      >
        <g stroke={TEAL} strokeOpacity="0.45" strokeWidth="1.1" strokeLinecap="round">
          {EDGES.map(([a, b], i) => (
            <line key={i} x1={NODES[a].x} y1={NODES[a].y} x2={NODES[b].x} y2={NODES[b].y} />
          ))}
        </g>
        {NODES.map((n, i) => (
          <circle
            key={i}
            cx={n.x}
            cy={n.y}
            r={n.r}
            fill={n.hub ? TEAL : 'var(--background, #181b20)'}
            stroke={TEAL}
            strokeWidth={n.hub ? 0 : 1.3}
          >
            {n.hub && (
              <animate attributeName="opacity" values="1;0.55;1" dur="3s" repeatCount="indefinite" />
            )}
          </circle>
        ))}
      </svg>
      {showWordmark && (
        <GradientText
          animationSpeed={9}
          className={`font-display font-semibold tracking-tight ${wordmarkClassName ?? 'text-[18px]'}`}
        >
          Reticus
        </GradientText>
      )}
    </span>
  );
}
