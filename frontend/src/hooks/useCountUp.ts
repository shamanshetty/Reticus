import { useEffect, useState } from 'react';
import { animate, useReducedMotion } from 'framer-motion';

/**
 * Animate a number from its previous value to `target` on change.
 * Returns the live display value. Respects prefers-reduced-motion.
 */
export function useCountUp(target: number, duration = 0.7): number {
  const reduce = useReducedMotion();
  const [value, setValue] = useState(target);

  useEffect(() => {
    if (!isFinite(target)) {
      setValue(target);
      return;
    }
    if (reduce) {
      setValue(target);
      return;
    }
    const controls = animate(value, target, {
      duration,
      ease: 'easeOut',
      onUpdate: (v) => setValue(v),
    });
    return () => controls.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, reduce]);

  return value;
}
