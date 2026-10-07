import { useCallback, useEffect, useMemo, useState } from 'react';
import { advanceWalk, isSink, startWalk, type Metagraph, type WalkPolicy, type WalkState } from '../model';

export type WalkSpeed = 'slow' | 'normal' | 'fast';

export const SPEED_MS: Record<WalkSpeed, number> = { slow: 1500, normal: 900, fast: 450 };

interface UseWalkOptions {
  readonly metagraph: Metagraph;
  readonly gamma: number;
  readonly policy: WalkPolicy;
  readonly seed: number;
  readonly start: string;
  readonly speed: WalkSpeed;
}

export interface WalkController {
  readonly state: WalkState;
  readonly playing: boolean;
  /** Predicted partitions after the current one (the policies are deterministic). */
  readonly preview: readonly string[];
  /** True right after γ, the rule, the seed or the start changed mid-walk. */
  readonly restarted: boolean;
  readonly intervalMs: number;
  readonly step: () => void;
  readonly play: () => void;
  readonly pause: () => void;
  readonly reset: () => void;
}

interface Stored {
  readonly key: string;
  readonly state: WalkState;
}

export function useWalk({ metagraph, gamma, policy, seed, start, speed }: UseWalkOptions): WalkController {
  const key = `${gamma}|${policy}|${seed}|${start}`;
  const [stored, setStored] = useState<Stored>(() => ({ key, state: startWalk(start, seed) }));
  const [playing, setPlaying] = useState(false);

  // A configuration change restarts the walk: derive it instead of syncing state in an effect.
  const fresh = stored.key !== key;
  const state = useMemo<WalkState>(() => {
    const current = fresh ? startWalk(start, seed) : stored.state;
    // A start that is already stable is a sink before any step is taken.
    return current.status === 'running' && isSink(metagraph, current.current, gamma) ? { ...current, status: 'sink' } : current;
  }, [fresh, start, seed, stored.state, metagraph, gamma]);
  const restarted = fresh && stored.state.steps.length > 0;

  const step = useCallback(() => {
    const base = stored.key === key ? stored.state : startWalk(start, seed);
    const next = advanceWalk(metagraph, base, gamma, policy);
    setStored({ key, state: next });
    if (next.status !== 'running') setPlaying(false);
  }, [stored, key, start, seed, metagraph, gamma, policy]);

  const reset = useCallback(() => {
    setPlaying(false);
    setStored({ key, state: startWalk(start, seed) });
  }, [key, start, seed]);

  const intervalMs = SPEED_MS[speed];

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setTimeout(step, stored.state.steps.length === 0 && !fresh ? 250 : intervalMs);
    return () => window.clearTimeout(timer);
  }, [playing, step, intervalMs, stored, fresh]);

  const preview = useMemo(() => {
    let predicted = state.status === 'running' ? advanceWalk(metagraph, state, gamma, policy) : state;
    while (predicted.status === 'running') predicted = advanceWalk(metagraph, predicted, gamma, policy);
    return predicted.steps.slice(state.steps.length).map((s) => s.to);
  }, [state, metagraph, gamma, policy]);

  return {
    state,
    playing,
    preview,
    restarted,
    intervalMs,
    step,
    play: () => {
      if (state.status === 'running') setPlaying(true);
    },
    pause: () => setPlaying(false),
    reset,
  };
}
