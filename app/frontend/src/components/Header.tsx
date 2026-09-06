import React, { useState, useEffect } from 'react';
import { Shield, Radio, Activity, Info, BarChart3, Database } from 'lucide-react';

interface HeaderProps {
  onOpenModelInfo: () => void;
  onOpenBenchmarks: () => void;
  connected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenModelInfo,
  onOpenBenchmarks,
  connected,
}) => {
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-[#1f293d] bg-[#0d111a]/80 backdrop-blur-md sticky top-0 z-40 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
      {/* Title & Branding */}
      <div className="flex items-center gap-3.5">
        <div className="h-10 w-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
          <Shield className="h-5 w-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold tracking-wider text-slate-100 uppercase">
              NetForecaster AI
            </h1>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/60 text-cyan-300">
              SIH26153
            </span>
          </div>
          <p className="text-xs text-slate-400 tracking-tight">
            AI-Based Multi-Horizon Network Attack Forecasting Operations
          </p>
        </div>
      </div>

      {/* Operational Status Badges */}
      <div className="flex items-center gap-3">
        {/* System Connection Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#101522] border border-[#1f293d]">
          <span className={`h-2.5 w-2.5 rounded-full ${connected ? 'bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]' : 'bg-rose-500'}`} />
          <span className="text-xs font-mono font-medium text-slate-300">
            {connected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
          </span>
        </div>

        {/* Mode Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#101522] border border-[#1f293d]">
          <Radio className="h-3.5 w-3.5 text-cyan-400 animate-pulse" />
          <span className="text-xs font-mono text-cyan-300 font-semibold tracking-wide">
            OFFLINE TRAFFIC REPLAY
          </span>
        </div>

        {/* Dataset */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#101522] border border-[#1f293d]">
          <Database className="h-3.5 w-3.5 text-slate-400" />
          <span className="text-xs font-mono text-slate-300">
            CSE-CIC-IDS2018 (Feb 28 - Mar 02)
          </span>
        </div>

        {/* Live Clock */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#101522] border border-[#1f293d]">
          <Activity className="h-3.5 w-3.5 text-slate-400" />
          <span className="text-xs font-mono text-slate-300">
            {time}
          </span>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 ml-2">
          <button
            onClick={onOpenBenchmarks}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#161c2d] hover:bg-cyan-950/40 border border-[#1f293d] hover:border-cyan-500/50 text-slate-200 hover:text-cyan-300 transition-all duration-150"
            title="Model Benchmarks & Comparison"
          >
            <BarChart3 className="h-3.5 w-3.5 text-cyan-400" />
            <span>Benchmarks</span>
          </button>

          <button
            onClick={onOpenModelInfo}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#161c2d] hover:bg-cyan-950/40 border border-[#1f293d] hover:border-cyan-500/50 text-slate-200 hover:text-cyan-300 transition-all duration-150"
            title="Architecture & Methodology Details"
          >
            <Info className="h-3.5 w-3.5 text-cyan-400" />
            <span>Architecture</span>
          </button>
        </div>
      </div>
    </header>
  );
};
