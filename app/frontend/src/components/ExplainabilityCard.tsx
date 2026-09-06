import React from 'react';
import { HelpCircle, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { ExplanatoryFeature } from '../types';

interface ExplainabilityCardProps {
  features: ExplanatoryFeature[];
}

export const ExplainabilityCard: React.FC<ExplainabilityCardProps> = ({ features }) => {
  return (
    <div className="cyber-card p-5">
      <div className="flex items-center justify-between border-b border-[#1f293d] pb-3 mb-4">
        <div className="flex items-center gap-2">
          <HelpCircle className="h-4 w-4 text-cyan-400" />
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
            Why is the Threat Risk Elevated? (Integrated Gradients)
          </h2>
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          Top-5 Key Flow Drivers
        </span>
      </div>

      {/* Top Features List */}
      <div className="flex flex-col gap-2.5">
        {features && features.length > 0 ? (
          features.map((f, i) => {
            const isIncrease = f.direction === 'increases_attack_risk';
            const pct = Math.round(f.importance * 100);

            return (
              <div
                key={i}
                className="bg-[#0a0d14] border border-[#1f293d] rounded-lg p-2.5 flex items-center justify-between gap-3 hover:border-cyan-500/40 transition-colors"
              >
                {/* Feature Name & Direction */}
                <div className="flex items-center gap-2.5 flex-1 min-w-0">
                  <div
                    className={`h-7 w-7 rounded-md flex items-center justify-center flex-shrink-0 ${
                      isIncrease
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    }`}
                  >
                    {isIncrease ? (
                      <ArrowUpRight className="h-4 w-4" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4" />
                    )}
                  </div>
                  <div className="truncate">
                    <div className="text-xs font-mono font-semibold text-slate-200 truncate">
                      {f.feature}
                    </div>
                    <div className="text-[10px] font-mono text-slate-400">
                      {isIncrease ? '↑ Increases Attack Risk' : '↓ Lowers Attack Risk'}
                    </div>
                  </div>
                </div>

                {/* Relative Attribution Bar */}
                <div className="flex items-center gap-3 w-36 flex-shrink-0">
                  <div className="flex-1 h-2 bg-[#161c2d] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        isIncrease ? 'bg-rose-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(10, pct))}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono font-bold text-slate-300 w-9 text-right">
                    {pct}%
                  </span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="p-4 text-center text-xs font-mono text-slate-500">
            Awaiting feature attribution telemetry...
          </div>
        )}
      </div>
    </div>
  );
};
