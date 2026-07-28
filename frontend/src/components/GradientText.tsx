import { useEffect, useRef, useState, type ReactNode } from 'react';
import { motion, useMotionValue, useAnimationFrame, useTransform } from 'framer-motion';

/**
 * Animated gradient text for headings. Uses framer-motion (the project's
 * existing animation lib) — same hooks the React Bits `motion/react` version used.
 */

type Direction = 'horizontal' | 'vertical' | 'diagonal';

type GradientTextProps = {
  children: ReactNode;
  className?: string;
  colors?: string[];
  animationSpeed?: number;
  direction?: Direction;
  pauseOnHover?: boolean;
  yoyo?: boolean;
};

export default function GradientText({
  children,
  className = '',
  // Reticus palette: teal -> prussian -> emerald, looping back to teal.
  colors = ['#40B5AD', '#2f8f88', '#50C878', '#0a4d68', '#40B5AD'],
  animationSpeed = 8,
  direction = 'horizontal',
  pauseOnHover = false,
  yoyo = true,
}: GradientTextProps) {
  const [isPaused, setIsPaused] = useState(false);
  const progress = useMotionValue(0);
  const elapsedRef = useRef(0);
  const lastTimeRef = useRef<number | null>(null);
  const animationDuration = animationSpeed * 1000;

  useAnimationFrame((time) => {
    if (isPaused) {
      lastTimeRef.current = null;
      return;
    }
    if (lastTimeRef.current === null) {
      lastTimeRef.current = time;
      return;
    }
    const deltaTime = time - lastTimeRef.current;
    lastTimeRef.current = time;
    elapsedRef.current += deltaTime;

    if (yoyo) {
      const fullCycle = animationDuration * 2;
      const cycleTime = elapsedRef.current % fullCycle;
      if (cycleTime < animationDuration) progress.set((cycleTime / animationDuration) * 100);
      else progress.set(100 - ((cycleTime - animationDuration) / animationDuration) * 100);
    } else {
      progress.set((elapsedRef.current / animationDuration) * 100);
    }
  });

  useEffect(() => {
    elapsedRef.current = 0;
    progress.set(0);
  }, [animationSpeed, progress, yoyo]);

  const backgroundPosition = useTransform(progress, (p) =>
    direction === 'vertical' ? `50% ${p}%` : `${p}% 50%`
  );

  const gradientAngle =
    direction === 'horizontal' ? 'to right' : direction === 'vertical' ? 'to bottom' : 'to bottom right';
  const gradientColors = [...colors, colors[0]].join(', ');

  const gradientStyle = {
    backgroundImage: `linear-gradient(${gradientAngle}, ${gradientColors})`,
    backgroundSize:
      direction === 'horizontal' ? '300% 100%' : direction === 'vertical' ? '100% 300%' : '300% 300%',
    backgroundRepeat: 'repeat' as const,
    WebkitBackgroundClip: 'text' as const,
    backgroundClip: 'text' as const,
    color: 'transparent',
  };

  return (
    <motion.span
      className={className}
      style={{ display: 'inline-block' }}
      onMouseEnter={() => pauseOnHover && setIsPaused(true)}
      onMouseLeave={() => pauseOnHover && setIsPaused(false)}
    >
      <motion.span style={{ ...gradientStyle, backgroundPosition, display: 'inline-block' }}>
        {children}
      </motion.span>
    </motion.span>
  );
}
