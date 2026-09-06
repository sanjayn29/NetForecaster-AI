import React from 'react';
import { Play, Pause, RotateCcw, FastForward, SkipForward, Layers } from 'lucide-react';
import { ReplayStatus, SegmentMeta } from '../types';

interface ReplayControlsProps {
  status: ReplayStatus | null;
  segments: Record<string, SegmentMeta>;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onStep: () => void;
  onSpeedChange: (speed: number) => void;
  onSelectSegment: (segmentId: string) => void;
}

export const ReplayControls: React.FC<ReplayControlsProps> = ({
  status,
  segments,
  onStart,
  onPause,
  onReset,
  onStep,
  onSpeedChange,
  onSelectSegment,
}) => {
  const isRunning = status?.is_running ?? false;
  const currentStep = status?.current_step ?? 20;
  const totalSteps = status?.total_steps ?? 150;
  const progressPct = status?.progress_pct ?? 0;
  const currentSpeed = status?.speed ?? 1.0;
  const activeSegment = status?.active_segment ?? 'infiltration_attack';

  return (
    <div className="cyber-card p-4 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
      {/* Segment Selector & Info */}
      <div className="flex items-center gap-3 flex-1 min-w-[280px]">
        <div className="flex items-center gap-2 text-xs font-mono font-medium text-slate-400">
          <Layers className="h-4 w-4 text-cyan-400" />
          <span>SCENARIO:</span>
        </div>
        <select
          value={activeSegment}
          onChange={(e) => onSelectSegment(e.target.value)}
          className="bg-[#0a0d14] border border-[#1f293d] hover:border-cyan-500/50 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-medium focus:outline-none focus:border-cyan-400 transition-colors w-full max-w-md"
        >
          {Object.entries(segments).map(([id, meta]) => (
            <option key={id} value={id}>
              {meta.name}
            </option>
          ))}
        </select>
      </div>

      {/* Progress & Flow Counter */}
      <div className="flex-1 min-w-[220px] flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span>PROGRESS</span>
          <span className="text-cyan-300 font-semibold">
            Flow {currentStep} / {totalSteps} ({progressPct}%)
          </span>
        </div>
        <div className="w-full h-2 bg-[#0a0d14] rounded-full overflow-hidden border border-[#1f293d]">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-teal-400 to-emerald-400 transition-all duration-200"
            style={{ width: `${Math.min(100, Math.max(0, progressPct))}%` }}
          />
        </div>
      </div>

      {/* Control Action Buttons */}
      <div className="flex items-center gap-2 justify-end">
        {/* Play / Pause */}
        {isRunning ? (
          <button
            onClick={onPause}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/50 text-amber-300 text-xs font-semibold tracking-wider transition-all shadow-[0_0_10px_rgba(245,158,11,0.2)]"
          >
            <Pause className="h-3.5 w-3.5 fill-current" />
            <span>PAUSE</span>
          </button>
        ) : (
          <button
            onClick={onStart}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/50 text-cyan-300 text-xs font-semibold tracking-wider transition-all shadow-[0_0_10px_rgba(0,240,255,0.2)]"
          >
            <Play className="h-3.5 w-3.5 fill-current" />
            <span>PLAY REPLAY</span>
          </button>
        )}

        {/* Step Forward */}
        <button
          onClick={onStep}
          disabled={isRunning}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#161c2d] hover:bg-cyan-950/40 border border-[#1f293d] hover:border-cyan-500/50 disabled:opacity-40 disabled:hover:border-[#1f293d] text-slate-300 text-xs font-medium transition-colors"
          title="Advance 1 Flow Step"
        >
          <SkipForward className="h-3.5 w-3.5" />
          <span>STEP</span>
        </button>

        {/* Reset */}
        <button
          onClick={onReset}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#161c2d] hover:bg-rose-950/40 border border-[#1f293d] hover:border-rose-500/50 text-slate-300 hover:text-rose-300 text-xs font-medium transition-colors"
          title="Reset Playback to Step 20"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>RESET</span>
        </button>

        {/* Speed Toggles */}
        <div className="flex items-center border border-[#1f293d] rounded-lg bg-[#0a0d14] p-0.5 ml-1">
          {[0.5, 1.0, 2.0, 5.0].map((s) => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`px-2 py-1 text-[11px] font-mono font-medium rounded transition-colors ${
                currentSpeed === s
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
