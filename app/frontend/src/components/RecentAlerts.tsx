import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { AlertLogItem } from '../types';

interface RecentAlertsProps {
  alerts: AlertLogItem[];
}

export const RecentAlerts: React.FC<RecentAlertsProps> = ({ alerts }) => {
  return (
    <div className="cyber-card p-5">
      <div className="flex items-center justify-between border-b border-[#1f293d] pb-3 mb-3">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-cyan-400" />
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
            Recent SOC Incident & Forecast Telemetry
          </h2>
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          Last {alerts.length} Replay Frames
        </span>
      </div>

      <div className="overflow-x-auto max-h-56">
        <table className="w-full text-left text-xs font-mono">
          <thead className="text-[10px] uppercase text-slate-400 border-b border-[#1f293d] sticky top-0 bg-[#101522]">
            <tr>
              <th className="py-2 px-3">Timestamp</th>
              <th className="py-2 px-3">Risk Score</th>
              <th className="py-2 px-3">Alert Level</th>
              <th className="py-2 px-3">Threat Stage</th>
              <th className="py-2 px-3">T+1 Forecast</th>
              <th className="py-2 px-3">Actual Future Ground Truth</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#182030] text-slate-300">
            {alerts && alerts.length > 0 ? (
              alerts.map((item) => {
                const isCrit = item.risk_level === 'CRITICAL';
                const isHigh = item.risk_level === 'HIGH';
                const isMed = item.risk_level === 'MEDIUM';

                return (
                  <tr key={item.id} className="hover:bg-[#161c2d]/60 transition-colors">
                    <td className="py-2 px-3 text-slate-400 whitespace-nowrap">{item.timestamp}</td>
                    <td className="py-2 px-3 font-bold text-slate-100">{item.risk_score}</td>
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isCrit
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : isHigh
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : isMed
                            ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        }`}
                      >
                        {item.risk_level}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-200">{item.stage_name}</td>
                    <td className="py-2 px-3 text-cyan-300 font-semibold">{Math.round(item.t1_prob * 100)}%</td>
                    <td className="py-2 px-3">
                      <span
                        className={`font-semibold ${
                          item.actual_label.toLowerCase() === 'benign'
                            ? 'text-emerald-400'
                            : 'text-rose-400'
                        }`}
                      >
                        {item.actual_label}
                      </span>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={6} className="py-6 text-center text-slate-500 text-xs">
                  Replay stream initializing. Incident telemetry will appear here in real-time.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
