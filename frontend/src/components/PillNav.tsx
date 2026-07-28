import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import './PillNav.css';

/**
 * Pill navigation, ported from React Bits. Router stripped — items fire an
 * onClick(key) callback to drive the app's state-based page switching.
 */

export type PillNavItem = {
  key: string;
  label: string;
  ariaLabel?: string;
};

type PillNavProps = {
  items: PillNavItem[];
  activeKey?: string;
  onSelect?: (key: string) => void;
  logo?: React.ReactNode;
  onLogoClick?: () => void;
  className?: string;
  ease?: string;
  baseColor?: string;
  pillColor?: string;
  hoveredPillTextColor?: string;
  pillTextColor?: string;
  initialLoadAnimation?: boolean;
};

export default function PillNav({
  items,
  activeKey,
  onSelect,
  logo,
  onLogoClick,
  className = '',
  ease = 'power3.easeOut',
  baseColor = '#40B5AD',
  pillColor = 'rgba(255,255,255,0.04)',
  hoveredPillTextColor = '#0c1116',
  pillTextColor,
  initialLoadAnimation = true,
}: PillNavProps) {
  const resolvedPillTextColor = pillTextColor ?? '#e7ecf3';
  const circleRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const tlRefs = useRef<Array<gsap.core.Timeline | null>>([]);
  const activeTweenRefs = useRef<Array<gsap.core.Tween | null>>([]);
  const navItemsRef = useRef<HTMLDivElement>(null);
  const [, force] = useState(0);

  useEffect(() => {
    const layout = () => {
      circleRefs.current.forEach((circle) => {
        if (!circle?.parentElement) return;
        const pill = circle.parentElement;
        const rect = pill.getBoundingClientRect();
        const { width: w, height: h } = rect;
        const R = ((w * w) / 4 + h * h) / (2 * h);
        const D = Math.ceil(2 * R) + 2;
        const delta = Math.ceil(R - Math.sqrt(Math.max(0, R * R - (w * w) / 4))) + 1;
        const originY = D - delta;
        circle.style.width = `${D}px`;
        circle.style.height = `${D}px`;
        circle.style.bottom = `-${delta}px`;
        gsap.set(circle, { xPercent: -50, scale: 0, transformOrigin: `50% ${originY}px` });

        const label = pill.querySelector<HTMLElement>('.pill-label');
        const white = pill.querySelector<HTMLElement>('.pill-label-hover');
        if (label) gsap.set(label, { y: 0 });
        if (white) gsap.set(white, { y: h + 12, opacity: 0 });

        const index = circleRefs.current.indexOf(circle);
        if (index === -1) return;
        tlRefs.current[index]?.kill();
        const tl = gsap.timeline({ paused: true });
        tl.to(circle, { scale: 1.2, xPercent: -50, duration: 2, ease, overwrite: 'auto' }, 0);
        if (label) tl.to(label, { y: -(h + 8), duration: 2, ease, overwrite: 'auto' }, 0);
        if (white) {
          gsap.set(white, { y: Math.ceil(h + 100), opacity: 0 });
          tl.to(white, { y: 0, opacity: 1, duration: 2, ease, overwrite: 'auto' }, 0);
        }
        tlRefs.current[index] = tl;
      });
    };

    layout();
    const onResize = () => layout();
    window.addEventListener('resize', onResize);
    if (document.fonts?.ready) document.fonts.ready.then(layout).catch(() => undefined);

    if (initialLoadAnimation && navItemsRef.current) {
      const navItems = navItemsRef.current;
      gsap.set(navItems, { width: 0, overflow: 'hidden' });
      gsap.to(navItems, { width: 'auto', duration: 0.6, ease });
    }

    return () => window.removeEventListener('resize', onResize);
    // force re-run layout if item set changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, ease, initialLoadAnimation]);

  const handleEnter = (i: number) => {
    const tl = tlRefs.current[i];
    if (!tl) return;
    activeTweenRefs.current[i]?.kill();
    activeTweenRefs.current[i] = tl.tweenTo(tl.duration(), { duration: 0.3, ease, overwrite: 'auto' });
  };
  const handleLeave = (i: number) => {
    const tl = tlRefs.current[i];
    if (!tl) return;
    activeTweenRefs.current[i]?.kill();
    activeTweenRefs.current[i] = tl.tweenTo(0, { duration: 0.2, ease, overwrite: 'auto' });
  };

  // keep referenced so eslint/force isn't flagged; harmless
  void force;

  const cssVars = {
    ['--base' as string]: baseColor,
    ['--pill-bg' as string]: pillColor,
    ['--hover-text' as string]: hoveredPillTextColor,
    ['--pill-text' as string]: resolvedPillTextColor,
  } as React.CSSProperties;

  return (
    <nav className={`pill-nav ${className}`} aria-label="Primary" style={cssVars}>
      {logo && (
        <button type="button" className="pill-logo-btn" aria-label="Home" onClick={onLogoClick}>
          {logo}
        </button>
      )}
      <div className="pill-nav-items" ref={navItemsRef}>
        <ul className="pill-list" role="menubar">
          {items.map((item, i) => (
            <li key={item.key} role="none">
              <button
                type="button"
                role="menuitem"
                className={`pill${activeKey === item.key ? ' is-active' : ''}`}
                aria-label={item.ariaLabel || item.label}
                onMouseEnter={() => handleEnter(i)}
                onMouseLeave={() => handleLeave(i)}
                onClick={() => onSelect?.(item.key)}
              >
                <span
                  className="hover-circle"
                  aria-hidden="true"
                  ref={(el) => {
                    circleRefs.current[i] = el;
                  }}
                />
                <span className="label-stack">
                  <span className="pill-label">{item.label}</span>
                  <span className="pill-label-hover" aria-hidden="true">
                    {item.label}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
