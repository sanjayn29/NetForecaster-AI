import React from 'react';
import { Target, AlertCircle } from 'lucide-react';
import { ThreatStageInfo } from '../types';

interface ThreatStageCardProps {
  stage: ThreatStageInfo | null;
  predictedAttackType: string;
}

const STAGES = [
  { id: 0, name: 'Normal', code: 'Stage 0' },
  { id: 1, name: 'Init Access', code: 'Stage 1' },
  { id: 2, name: 'Web Exploit', code: 'Stage 2' },
  { id: 3, name: 'DoS/Impact', code: 'Stage 3' },
  { id: 4, name: 'C2 Beacon', code: 'Stage 4' },
  { id: 5, name: 'Lateral Mvmt', code: 'Stage 5' },
];

export const ThreatStageCard: React.FC<ThreatStageCardProps> = ({
  stage,
  predictedAttackType,
}) => {
  const currentStageId = stage?.stage_id ?? 0;
  const stageName = stage?.stage_name ?? 'Normal Baseline';
  const description = stage?.description ?? 'Normal baseline network traffic operations.';

  return (
    <div className="cyber-card p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between border-b border-[#1f293d] pb-3 mb-3">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-cyan-400" />
            <h2 className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold">
              Threat Lifecycle Stage
            </h2>
          </div>
          <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-[#161c2d] text-cyan-300 border border-cyan-800/50">
            STAGE {currentStageId} OF 5
          </span>
        </div>

        {/* Current Active Stage Badge */}
        <div className="bg-[#0a0d14] border border-[#1f293d] rounded-xl p-3 mb-3 flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono text-slate-400 uppercase">IDENTIFIED TAXONOMY STAGE</div>
            <div className="text-sm font-bold text-slate-100 mt-0.5">{stageName}</div>
            <div className="text-xs text-slate-400 mt-1">{description}</div>
          </div>
          <div className="text-right">
            <div className="text-[10px] font-mono text-slate-400 uppercase">ATTACK CLASSIFICATION</div>
            <div className="text-xs font-mono font-bold text-amber-300 mt-0.5">{predictedAttackType}</div>
          </div>
        </div>

        {/* Stage Breadcrumb Progression */}
        <div className="grid grid-cols-6 gap-1.5 mb-3">
          {STAGES.map((s) => {
            const isActive = s.id === currentStageId;
            const isPast = s.id < currentStageId && currentStageId > 0;
            return (
              <div
                key={s.id}
                className={`flex flex-col items-center justify-center p-1.5 rounded-lg border text-center transition-all ${
                  isActive
                    ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 font-bold shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                    : isPast
                    ? 'bg-[#161c2d]/80 border-slate-700 text-slate-400'
                    : 'bg-[#0a0d14] border-[#1f293d] text-slate-600'
                }`}
              >
                <span className="text-[10px] font-mono font-bold">{s.id}</span>
                <span className="text-[9px] truncate max-w-full font-sans leading-tight mt-0.5">
                  {s.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Heuristic Disclaimer */}
      <div className="flex items-start gap-1.5 p-2 rounded-lg bg-[#0a0d14] border border-[#1f293d] text-[10px] text-slate-400 leading-tight">
        <AlertCircle className="h-3.5 w-3.5 text-slate-500 flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-slate-300 font-medium">Taxonomy Note:</strong> Threat stages are heuristic operational translations mapped from the CIC-IDS2018 dataset and do not represent guaranteed ground-truth kill chain events.
        </span>
      </div>
    </div>
  );
};
