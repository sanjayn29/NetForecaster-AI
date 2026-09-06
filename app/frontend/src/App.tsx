import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { ReplayControls } from './components/ReplayControls';
import { RiskCard } from './components/RiskCard';
import { ThreatStageCard } from './components/ThreatStageCard';
import { ForecastChart } from './components/ForecastChart';
import { ExplainabilityCard } from './components/ExplainabilityCard';
import { TemporalTimeline } from './components/TemporalTimeline';
import { RecentAlerts } from './components/RecentAlerts';
import { BenchmarkModal } from './components/BenchmarkModal';
import { ModelInfoModal } from './components/ModelInfoModal';
import {
  ForecastData,
  GroundTruthData,
  ReplayPayload,
  ReplayStatus,
  SegmentMeta,
  ModelBenchmarkItem,
  ModelInfo,
  AlertLogItem,
} from './types';

export const App: React.FC = () => {
  // State
  const [connected, setConnected] = useState<boolean>(false);
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [segments, setSegments] = useState<Record<string, SegmentMeta>>({});
  const [currentForecast, setCurrentForecast] = useState<ForecastData | null>(null);
  const [groundTruth, setGroundTruth] = useState<GroundTruthData | null>(null);
  const [alerts, setAlerts] = useState<AlertLogItem[]>([]);
  const [benchmarks, setBenchmarks] = useState<ModelBenchmarkItem[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Modals
  const [isBenchmarksOpen, setIsBenchmarksOpen] = useState<boolean>(false);
  const [isModelInfoOpen, setIsModelInfoOpen] = useState<boolean>(false);

  // WebSocket & State Refs to stabilize callbacks
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const segmentsRef = useRef<Record<string, SegmentMeta>>({});

  // Keep segmentsRef in sync
  useEffect(() => {
    segmentsRef.current = segments;
  }, [segments]);

  // Fetch initial REST metadata strictly once
  const fetchMetadata = useCallback(async () => {
    try {
      const [segRes, benchRes, infoRes] = await Promise.all([
        fetch('/api/replay/segments'),
        fetch('/api/benchmarks'),
        fetch('/api/model-info'),
      ]);

      if (segRes.ok) {
        const segData = await segRes.json();
        setSegments(segData.segments || {});
        segmentsRef.current = segData.segments || {};
      }
      if (benchRes.ok) {
        const bData = await benchRes.json();
        setBenchmarks(bData.models || []);
      }
      if (infoRes.ok) {
        const iData = await infoRes.json();
        setModelInfo(iData);
      }
    } catch (e) {
      console.warn('Could not load initial REST metadata:', e);
    }
  }, []);

  // Process inbound Replay Frame
  const handleReplayFrame = useCallback((payload: ReplayPayload) => {
    if (payload.forecast) {
      setCurrentForecast(payload.forecast);
      setGroundTruth(payload.actual_ground_truth);

      // Update status progress
      setStatus((prev) => ({
        active_segment: payload.segment_id,
        segment_meta: segmentsRef.current[payload.segment_id] || prev?.segment_meta || {
          name: payload.segment_id,
          description: '',
          start_row: 0,
          length: payload.total_steps,
        },
        current_step: payload.step,
        total_steps: payload.total_steps,
        is_running: prev?.is_running ?? false,
        speed: prev?.speed ?? 1.0,
        progress_pct: payload.progress_pct,
        active_connections: 1,
      }));

      // Add to rolling alert log
      const newAlert: AlertLogItem = {
        id: `${payload.step}-${Date.now()}`,
        timestamp: payload.forecast.timestamp,
        risk_score: payload.forecast.risk_score,
        risk_level: payload.forecast.risk_level,
        stage_name: payload.forecast.threat_stage.stage_name,
        t1_prob: payload.forecast.forecast_horizons['T+1'] || 0.0,
        actual_label: payload.actual_ground_truth?.labels?.['T+1'] || 'Benign',
      };

      setAlerts((prev) => [newAlert, ...prev.slice(0, 14)]);
    }
  }, []);

  // WebSocket connection manager
  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/replay`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'replay_update') {
          handleReplayFrame(data);
        }
      } catch (e) {
        console.error('Error parsing WS frame:', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Try to reconnect every 3s
      if (!reconnectTimeoutRef.current) {
        reconnectTimeoutRef.current = window.setTimeout(connectWebSocket, 3000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [handleReplayFrame]);

  useEffect(() => {
    fetchMetadata();
    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [fetchMetadata, connectWebSocket]);

  // Control Actions
  const handleStart = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'start' }));
    } else {
      await fetch('/api/replay/start', { method: 'POST' });
    }
    setStatus((prev) => (prev ? { ...prev, is_running: true } : null));
  };

  const handlePause = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'pause' }));
    } else {
      await fetch('/api/replay/pause', { method: 'POST' });
    }
    setStatus((prev) => (prev ? { ...prev, is_running: false } : null));
  };

  const handleReset = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    } else {
      await fetch('/api/replay/reset', { method: 'POST' });
    }
    setStatus((prev) => (prev ? { ...prev, is_running: false, current_step: 20, progress_pct: 0 } : null));
  };

  const handleStep = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'step' }));
    } else {
      const res = await fetch('/api/replay/step', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        handleReplayFrame(data);
      }
    }
  };

  const handleSpeedChange = async (speed: number) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'speed', speed }));
    } else {
      await fetch('/api/replay/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed }),
      });
    }
    setStatus((prev) => (prev ? { ...prev, speed } : null));
  };

  const handleSelectSegment = async (segmentId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'select_segment', segment_id: segmentId }));
    } else {
      await fetch('/api/replay/select-segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segment_id: segmentId }),
      });
    }
    setStatus((prev) =>
      prev
        ? {
            ...prev,
            active_segment: segmentId,
            is_running: false,
            current_step: 20,
            progress_pct: 0,
          }
        : null
    );
  };

  return (
    <div className="min-h-screen bg-[#0a0d14] text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <Header
        onOpenModelInfo={() => setIsModelInfoOpen(true)}
        onOpenBenchmarks={() => setIsBenchmarksOpen(true)}
        connected={connected}
      />

      {/* Main Dashboard Container */}
      <main className="flex-1 p-4 md:p-6 max-w-[1700px] w-full mx-auto space-y-5">
        {/* Replay Controls Toolbar */}
        <ReplayControls
          status={status}
          segments={segments}
          onStart={handleStart}
          onPause={handlePause}
          onReset={handleReset}
          onStep={handleStep}
          onSpeedChange={handleSpeedChange}
          onSelectSegment={handleSelectSegment}
        />

        {/* Top Operational Row: Risk Card + Threat Stage Card */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <RiskCard forecast={currentForecast} />
          <ThreatStageCard
            stage={currentForecast?.threat_stage || null}
            predictedAttackType={currentForecast?.predicted_attack_type || 'Benign'}
          />
        </div>

        {/* Center: Multi-Horizon 5-Step Forecast vs Ground Truth */}
        <ForecastChart
          forecastHorizons={currentForecast?.forecast_horizons || null}
          groundTruth={groundTruth}
        />

        {/* Bottom Row: Explainability + Temporal Saliency */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <ExplainabilityCard
            features={currentForecast?.top_explanatory_features || []}
          />
          <TemporalTimeline
            importance={currentForecast?.temporal_flow_importance || []}
            mostInfluentialStep={currentForecast?.most_influential_timestep || 't-0'}
          />
        </div>

        {/* Recent Incident Feed Table */}
        <RecentAlerts alerts={alerts} />
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1f293d] bg-[#0d111a] px-6 py-4 text-center text-xs font-mono text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div>
          NetForecaster AI • Smart India Hackathon (SIH26153) • Offline SOC Defense Suite
        </div>
        <div>
          Recursive Latent World Model Rollout (W=20, K=5) • Zero Cloud Dependencies
        </div>
      </footer>

      {/* Modals */}
      <BenchmarkModal
        isOpen={isBenchmarksOpen}
        onClose={() => setIsBenchmarksOpen(false)}
        benchmarks={benchmarks}
      />
      <ModelInfoModal
        isOpen={isModelInfoOpen}
        onClose={() => setIsModelInfoOpen(false)}
        info={modelInfo}
      />
    </div>
  );
};

export default App;
