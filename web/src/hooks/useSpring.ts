import { useEffect, useRef, useState } from 'react';

/**
 * Animate a number toward `target` with a damped spring. When `disabled`
 * (reduced motion), the target is returned immediately with no animation.
 */
export function useSpring(target: number, disabled: boolean, stiffness = 90, damping = 14): number {
  const [value, setValue] = useState(target);
  const state = useRef({ x: target, v: 0 });

  useEffect(() => {
    if (disabled) {
      state.current = { x: target, v: 0 };
      return undefined;
    }
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.04, (now - last) / 1000);
      last = now;
      const s = state.current;
      const force = -stiffness * (s.x - target) - damping * s.v;
      s.v += force * dt;
      s.x += s.v * dt;
      if (Math.abs(s.x - target) < 1e-3 && Math.abs(s.v) < 1e-3) {
        s.x = target;
        s.v = 0;
        setValue(target);
        return;
      }
      setValue(s.x);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, disabled, stiffness, damping]);

  return disabled ? target : value;
}
