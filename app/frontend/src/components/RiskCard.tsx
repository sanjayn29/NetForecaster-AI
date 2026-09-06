import React from 'react';
import { AlertTriangle, ShieldCheck, Zap, Gauge } from 'lucide-react';
import { ForecastData } from '../types';

interface RiskCardProps {
  forecast: ForecastData | null;
}

export const RiskCard: React.FC<RiskCardProps> = ({ forecast }) => {
  const riskScore = forecast ? Math.round(forecast.risk_score * 10) / 10 : 0;
  const riskLevel = forecast ? forecast.risk_level : 'LOW';
  const confidence = forecast ? Math.round(forecast.forecast_certainty * 100) : 0;
  const decisionMargin = forecast ? Math.round(forecast.decision_margin * 100) : 0;
  const latentNorm = forecast ? forecast.latent_state_norm : 5.45;

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'CRITICAL':
        return {
          text: 'text-rose-400',
          bg: 'bg-rose-500/10',
          border: 'border-rose-500/40',
          badge: 'bg-rose-500 text-black',
          glow: 'shadow-[0_0_25px_rgba(244,63,94,0.25)]',
          stroke: '#f43f5e',
        };
      case 'HIGH':
        return {
          text: 'text-amber-400',
          bg: 'bg-amber-500/10',
          border: 'border-amber-500/40',
          badge: 'bg-amber-500 text-black',
          glow: 'shadow-[0_0_25px_rgba(245,158,11,0.25)]',
          stroke: '#f59e0b',
        };
      case 'MEDIUM':
        return {
          text: 'text-yellow-300',
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/40',
          badge: 'bg-yellow-400 text-black',
          glow: 'shadow-[0_0_25px_rgba(234,179,8,0.2)]',
          stroke: '#eab308',
        };
      default:
        return {
          text: 'text-emerald-400',
          bg: 'bg-emerald-500/10',
          border: 'border-emerald-500/40',
          badge: 'bg-emerald-500 text-black',
          glow: 'shadow-[0_0_25px_rgba(16,185,129,0.2)]',
          stroke: '#10b981',
        };
    }
  };

  const style = getLevelColor(riskLevel);

  // SVG Circular Gauge calculation
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (Math.min(100, Math.max(0, riskScore)) / 100) * circumference;

  return (
    <div className={`cyber-card p-5 relative overflow-hidden ${style.border} ${style.glow}`}>
      {/* Background Subtle Gradient */}
      <div className={`absolute inset-0 ${style.bg} opacity-30 pointer-events-none`} />

      <div className="flex items-center justify-between border-b border-[#1f293d] pb-3 mb-4">
        <div className="flex items-center gap-2">
          <Gauge className={`h-4 w-4 ${style.text}`} />
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold">
            Current Threat Risk Score
          </h2>
        </div>
        <span className={`text-[10px] font-bold font-mono px-2.5 py-0.5 rounded-full ${style.badge}`}>
          {riskLevel} ALERT
        </span>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-around gap-6">
        {/* Radial Gauge */}
        <div className="relative flex items-center justify-center">
          <svg className="w-36 h-36 transform -rotate-90">
            {/* Background Track */}
            <circle
              cx="72"
              cy="72"
              r={radius}
              stroke="#1a2234"
              strokeWidth="10"
              fill="transparent"
            />
            {/* Value Track */}
            <circle
              cx="72"
              cy="72"
              r={radius}
              stroke={style.stroke}
              strokeWidth="10"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              className="transition-all duration-300 ease-out"
            />
          </svg>

          {/* Center Numerical Score */}
          <div className="absolute flex flex-col items-center justify-center text-center">
            <span className={`text-3xl font-bold font-mono tracking-tight ${style.text}`}>
              {riskScore}
            </span>
            <span className="text-[10px] uppercase font-mono text-slate-400 font-medium">
              / 100
            </span>
          </div>
        </div>

        {/* Risk Metrics Breakdown */}
        <div className="flex flex-col gap-3 w-full sm:w-auto">
          {/* Status Verdict */}
          <div className="flex items-center gap-2">
            {riskLevel === 'LOW' ? (
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
            ) : (
              <AlertTriangle className={`h-4 w-4 ${style.text} animate-bounce`} />
            )}
            <span className="text-xs font-semibold text-slate-200">
              {forecast?.current_state || 'Normal Network Traffic'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <div className="bg-[#0a0d14] border border-[#1f293d] rounded-lg p-2.5">
              <div className="text-slate-400 text-[10px]">FORECAST CERTAINTY</div>
              <div className="text-cyan-300 font-bold text-sm mt-0.5">{confidence}%</div>
            </div>

            <div className="bg-[#0a0d14] border border-[#1f293d] rounded-lg p-2.5">
              <div className="text-slate-400 text-[10px]">DECISION MARGIN</div>
              <div className="text-slate-200 font-bold text-sm mt-0.5">{decisionMargin}%</div>
            </div>

            <div className="bg-[#0a0d14] border border-[#1f293d] rounded-lg p-2.5 col-span-2 flex items-center justify-between">
              <div>
                <div className="text-slate-400 text-[10px]">LATENT STATE NORM ||S_t||</div>
                <div className="text-purple-300 font-bold text-xs mt-0.5">{latentNorm}</div>
              </div>
              <Zap className="h-4 w-4 text-purple-400" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
