import React from 'react';
import { X, Info, ShieldCheck, Cpu, GitCommit } from 'lucide-react';
import { ModelInfo } from '../types';

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  info: ModelInfo | null;
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({
  isOpen,
  onClose,
  info,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="cyber-card w-full max-w-3xl max-h-[90vh] overflow-y-auto p-6 border-cyan-500/40 shadow-[0_0_30px_rgba(0,240,255,0.15)] relative">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1f293d] pb-4 mb-4">
          <div className="flex items-center gap-2.5">
            <Info className="h-5 w-5 text-cyan-400" />
            <h2 className="text-sm font-bold font-mono tracking-wider text-slate-100 uppercase">
              System Architecture & Methodology Specifications
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Specifications Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5 text-xs font-mono">
          {/* Input Windowing */}
          <div className="bg-[#0a0d14] border border-[#1f293d] rounded-xl p-4">
            <div className="flex items-center gap-2 text-cyan-400 font-bold mb-2">
              <Cpu className="h-4 w-4" />
              <span>INPUT SPECIFICATION</span>
            </div>
            <ul className="space-y-1.5 text-slate-300">
              <li>• <strong>Window Size (W):</strong> 20 consecutive flow records (not seconds)</li>
              <li>• <strong>Features per Flow:</strong> 68 numerical traffic features</li>
              <li>• <strong>Input Tensor Shape:</strong> (Batch, 20, 68)</li>
              <li>• <strong>Preprocessing:</strong> RobustScaler (fitted strictly on training data)</li>
              <li>• <strong>Zero Identifier Leakage:</strong> IP & Port columns stripped from feature space</li>
            </ul>
          </div>

          {/* World Model */}
          <div className="bg-[#0a0d14] border border-[#1f293d] rounded-xl p-4">
            <div className="flex items-center gap-2 text-purple-400 font-bold mb-2">
              <GitCommit className="h-4 w-4" />
              <span>LATENT WORLD MODEL</span>
            </div>
            <ul className="space-y-1.5 text-slate-300">
              <li>• <strong>Latent Dimension:</strong> 64-dimensional continuous state vector S_t</li>
              <li>• <strong>Temporal Encoder:</strong> Causal Temporal Convolutional Network (TCN)</li>
              <li>• <strong>State Transition:</strong> Residual MLP: S_(t+1) = S_t + f(S_t)</li>
              <li>• <strong>Forecast Horizon (K):</strong> 5 future flow steps (T+1 ... T+5)</li>
              <li>• <strong>Rollout:</strong> Pure autoregressive latent rollout (zero teacher forcing)</li>
            </ul>
          </div>

          {/* Explainability & Risk */}
          <div className="bg-[#0a0d14] border border-[#1f293d] rounded-xl p-4 md:col-span-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold mb-2">
              <ShieldCheck className="h-4 w-4" />
              <span>EXPLAINABILITY & RISK FORMULATION</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-slate-300">
              <div>
                <strong className="text-slate-200">Integrated Gradients:</strong>
                <p className="text-slate-400 mt-1">Linear path interpolation across 20 historical flows isolating directional risk drivers (increases vs decreases risk).</p>
              </div>
              <div>
                <strong className="text-slate-200">Horizon-Decayed Risk Score:</strong>
                <p className="text-slate-400 mt-1">Simplex weights [0.30, 0.25, 0.20, 0.15, 0.10] weighting near-term attack emergence higher than distant predictions.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Methodology Notes & Disclaimers */}
        <div className="p-4 rounded-xl bg-[#0a0d14] border border-[#1f293d] text-xs text-slate-400">
          <strong className="text-slate-200 font-mono">Formal Methodology Notes & Scientific Integrity:</strong>
          <ul className="mt-2 space-y-1 text-slate-400 list-disc list-inside">
            {info?.methodology_disclaimers.map((d, i) => (
              <li key={i}>{d}</li>
            )) || (
              <>
                <li>Operating on offline CSE-CIC-IDS2018 chronological replay partition.</li>
                <li>Forecast horizons represent flow sequence steps, not physical clock seconds.</li>
                <li>Threat stages are operational taxonomy translations, not endpoint kill-chain telemetry.</li>
                <li>No fake IP graph or host relationships are fabricated.</li>
              </>
            )}
          </ul>
        </div>

        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-[#161c2d] hover:bg-[#1f293d] border border-[#1f293d] text-xs font-mono text-slate-200 transition-colors"
          >
            Close Information
          </button>
        </div>
      </div>
    </div>
  );
};
