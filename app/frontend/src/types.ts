export interface ThreatStageInfo {
  stage_id: number;
  stage_name: string;
  description: string;
  disclaimer: string;
}

export interface ExplanatoryFeature {
  feature: string;
  importance: number;
  raw_attribution: number;
  direction: 'increases_attack_risk' | 'decreases_attack_risk' | string;
}

export interface ForecastData {
  timestamp: string;
  current_state: string;
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  forecast_horizons: Record<string, number>;
  predicted_attack_type: string;
  threat_stage: ThreatStageInfo;
  top_explanatory_features: ExplanatoryFeature[];
  temporal_flow_importance: number[];
  most_influential_timestep: string;
  forecast_certainty: number;
  decision_margin: number;
  latent_state_norm: number;
  window_size: number;
  forecast_horizon_steps: number;
}

export interface GroundTruthData {
  labels: Record<string, string>;
  binary: Record<string, number>;
}

export interface ReplayPayload {
  type: string;
  segment_id: string;
  step: number;
  total_steps: number;
  progress_pct: number;
  current_flow_label: string;
  forecast: ForecastData;
  actual_ground_truth: GroundTruthData;
}

export interface SegmentMeta {
  name: string;
  description: string;
  start_row: number;
  length: number;
}

export interface ReplayStatus {
  active_segment: string;
  segment_meta: SegmentMeta;
  current_step: number;
  total_steps: number;
  is_running: boolean;
  speed: number;
  progress_pct: number;
  active_connections: number;
}

export interface ModelBenchmarkItem {
  name: string;
  type: string;
  t1_roc_auc: number;
  t1_pr_auc: number;
  t1_macro_f1: number;
  t5_roc_auc: number;
  t5_pr_auc: number;
  t5_macro_f1: number;
  brier_score: number;
  fpr: number;
  note: string;
}

export interface ModelInfo {
  dataset: string;
  input_specification: {
    window_size: string;
    features_per_flow: number;
    scaler: string;
  };
  forecasting_architecture: {
    type: string;
    latent_dim: number;
    encoder: string;
    transition: string;
    rollout_mechanism: string;
  };
  explainability: {
    method: string;
    attribution_targets: string;
  };
  risk_scoring: {
    method: string;
    alert_levels: string[];
  };
  methodology_disclaimers: string[];
}

export interface AlertLogItem {
  id: string;
  timestamp: string;
  risk_score: number;
  risk_level: string;
  stage_name: string;
  t1_prob: number;
  actual_label: string;
}
