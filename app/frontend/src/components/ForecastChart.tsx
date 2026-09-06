import React from 'react';
import { TrendingUp, CheckCircle2, XCircle } from 'lucide-react';
import { GroundTruthData } from '../types';

interface ForecastChartProps {
  forecastHorizons: Record<string, number> | null;
  groundTruth: GroundTruthData | null;
}

export const ForecastChart: React.FC<ForecastChartProps> = ({
  forecastHorizons,
  groundTruth,
}) => {
  const horizons = ['T+1', 'T+2', 'T+3', 'T+4', 'T+5'];

  return (
    <div className="cyber-card p-5">
      <div className="flex flex-wrap items-center justify-between border-b border-[#1f293d] pb-3 mb-4 gap-2">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-cyan-400" />
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
            Recursive 5-Step Forecast vs Ground-Truth State
          </h2>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-gradient-to-r from-cyan-400 to-teal-300" />
            <span className="text-cyan-300 font-medium">Predicted Attack Probability</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
            <span className="text-rose-300 font-medium">Actual Attack State</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
            <span className="text-emerald-300 font-medium">Actual Benign State</span>
          </div>
        </div>
      </div>

      {/* Horizon Bars Grid */}
      <div className="grid grid-cols-5 gap-3 sm:gap-4 pt-2">
        {horizons.map((h, idx) => {
          const prob = forecastHorizons ? forecastHorizons[h] ?? 0.0 : 0.0;
          const probPct = Math.round(prob * 100);
          const actualBinary = groundTruth?.binary ? groundTruth.binary[h] : undefined;
          const actualLabel = groundTruth?.labels ? groundTruth.labels[h] : 'Pending';

          const isAttack = actualBinary === 1;
          const isBenign = actualBinary === 0;

          // Horizon step decay label
          const horizonWeights = ['30%', '25%', '20%', '15%', '10%'];

          return (
            <div
              key={h}
              className="bg-[#0a0d14] border border-[#1f293d] hover:border-cyan-500/40 rounded-xl p-3 flex flex-col items-center justify-between transition-all"
            >
              {/* Header: Horizon name & weight */}
              <div className="text-center w-full border-b border-[#1f293d] pb-1.5 mb-2">
                <div className="text-xs font-bold font-mono text-slate-200">{h}</div>
                <div className="text-[10px] font-mono text-slate-500">
                  Weight: {horizonWeights[idx]}
                </div>
              </div>

              {/* Vertical Probability Bar */}
              <div className="w-full flex flex-col items-center gap-1 my-2">
                <div className="h-28 w-10 sm:w-12 bg-[#121826] rounded-lg relative overflow-hidden flex flex-col justify-end p-1 border border-[#1f293d]">
                  {/* Fill */}
                  <div
                    className="w-full rounded bg-gradient-to-t from-cyan-600 via-teal-400 to-cyan-300 transition-all duration-300 shadow-[0_0_12px_rgba(0,240,255,0.3)]"
                    style={{ height: `${Math.max(4, Math.min(100, probPct))}%` }}
                  />
                </div>
                <span className="text-xs font-mono font-bold text-cyan-300">
                  {probPct}%
                </span>
                <span className="text-[9px] uppercase font-mono text-slate-400">
                  P(Attack)
                </span>
              </div>

              {/* Ground Truth Status Marker */}
              <div className="w-full mt-2 pt-2 border-t border-[#1f293d] flex flex-col items-center">
                <span className="text-[9px] font-mono text-slate-400 mb-1">ACTUAL TRUTH</span>
                {isAttack && (
                  <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/20 border border-rose-500/50 text-rose-300 text-[10px] font-mono font-bold">
                    <XCircle className="h-3 w-3" />
                    <span>ATTACK</span>
                  </div>
                )}
                {isBenign && (
                  <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/50 text-emerald-300 text-[10px] font-mono font-bold">
                    <CheckCircle2 className="h-3 w-3" />
                    <span>BENIGN</span>
                  </div>
                )}
                {actualBinary === undefined && (
                  <div className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px] font-mono">
                    AWAITING
                  </div>
                )}
                <span className="text-[9px] text-slate-500 truncate max-w-[90px] mt-1 text-center font-mono">
                  {actualLabel}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
