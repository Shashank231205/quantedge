/**
 * API client.
 *
 * Every screen reads through this module, so there is exactly one place where
 * a request could be faked — and it isn't. If the backend is down or a run has
 * not been created yet, screens show an explicit empty state rather than
 * placeholder numbers.
 */

const API_KEY = import.meta.env.VITE_API_KEY ?? 'quantedge-dev-key'
const BASE = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* response body was not JSON */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  // --- Portfolio Dashboard -------------------------------------------
  portfolioSummary: () => request<PortfolioSummary>('/v1/portfolio/summary'),
  equityCurve: (maxPoints = 800) =>
    request<EquityCurve>(`/v1/portfolio/equity-curve?max_points=${maxPoints}`),
  drawdown: () => request<DrawdownSeries>('/v1/portfolio/drawdown'),
  signals: (topN = 20) => request<SignalsResponse>(`/v1/portfolio/signals?top_n=${topN}`),
  portfolioLogs: (limit = 20) => request<LogsResponse>(`/v1/portfolio/logs?limit=${limit}`),

  // --- Factor Explorer -------------------------------------------------
  factorTable: (params: { limit?: number; sortBy?: string; search?: string } = {}) => {
    const q = new URLSearchParams()
    if (params.limit) q.set('limit', String(params.limit))
    if (params.sortBy) q.set('sort_by', params.sortBy)
    if (params.search) q.set('search', params.search)
    return request<FactorTable>(`/v1/factors/table?${q}`)
  },
  factorCorrelation: () => request<CorrelationMatrix>('/v1/factors/correlation'),
  factorIC: () => request<FactorICResponse>('/v1/factors/ic'),
  factorList: () => request<FactorListResponse>('/v1/factors/list'),
  tickerDetail: (ticker: string) =>
    request<TickerDetail>(`/v1/factors/${ticker}/detail?lookback=252`),

  // --- Backtest Analysis -------------------------------------------------
  backtestRuns: () => request<{ runs: BacktestRunSummary[] }>('/v1/backtest/runs'),
  backtestMetrics: (runId?: number) =>
    request<BacktestMetrics>(`/v1/backtest/metrics${runId ? `?run_id=${runId}` : ''}`),
  backtestEquity: (runId?: number) =>
    request<EquityCurve>(`/v1/backtest/equity-curve${runId ? `?run_id=${runId}` : ''}`),
  backtestFolds: (runId?: number) =>
    request<FoldsResponse>(`/v1/backtest/folds${runId ? `?run_id=${runId}` : ''}`),
  trades: (params: { page?: number; pageSize?: number; outcome?: string; ticker?: string } = {}) => {
    const q = new URLSearchParams()
    q.set('page', String(params.page ?? 1))
    q.set('page_size', String(params.pageSize ?? 25))
    q.set('outcome', params.outcome ?? 'ALL')
    if (params.ticker) q.set('ticker', params.ticker)
    return request<TradesResponse>(`/v1/backtest/trades?${q}`)
  },
  strategySpec: () => request<StrategySpecResponse>('/v1/backtest/strategy'),
  runBacktest: (body: Record<string, unknown>) =>
    request<{ run_id: number; status: string }>('/v1/backtest/run', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  // --- Risk Monitor ---------------------------------------------------------
  riskSummary: () => request<RiskSummary>('/v1/risk/summary'),
  riskExposure: () => request<ExposureResponse>('/v1/risk/exposure'),
  riskPositions: (limit = 40) => request<PositionsResponse>(`/v1/risk/positions?limit=${limit}`),
  riskBreaches: () => request<BreachesResponse>('/v1/risk/breaches'),
  riskLogs: (limit = 20) => request<LogsResponse>(`/v1/risk/logs?limit=${limit}`),

  // --- Analyst ---------------------------------------------------------------
  analystStatus: () => request<AnalystStatus>('/v1/analyst/status'),
  analystReport: (params: { universeSize?: number; refresh?: boolean } = {}) => {
    const q = new URLSearchParams()
    q.set('universe_size', String(params.universeSize ?? 50))
    if (params.refresh) q.set('refresh', 'true')
    return request<AnalystReport>(`/v1/analyst/report?${q}`)
  },

  // --- System Health ---------------------------------------------------------
  systemStatus: () => request<SystemStatus>('/v1/system/status'),
  systemJobs: () => request<JobsResponse>('/v1/system/jobs'),
  ingestionStats: () => request<IngestionStats>('/v1/system/ingestion'),
  apiMetrics: () => request<ApiMetrics>('/v1/system/api-metrics'),
  systemLogs: (limit = 60, level = 'ALL') =>
    request<LogsResponse>(`/v1/system/logs?limit=${limit}&level=${level}`),
  systemInfo: () => request<SystemInfo>('/v1/system/info'),
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PortfolioSummary {
  run_id: number
  run_name: string
  is_out_of_sample: boolean
  sharpe_ratio: number | null
  sortino_ratio: number | null
  calmar_ratio: number | null
  annualized_return: number | null
  total_return: number | null
  volatility: number | null
  max_drawdown: number | null
  period: { start: string; end: string }
  pipeline: {
    uptime_pct: number | null
    total_runs: number
    failed_runs: number
    status: string
  }
}

export interface EquityPoint {
  date: string
  equity: number
  returns?: number | null
  drawdown?: number | null
  is_oos?: boolean
}

export interface EquityCurve {
  run_id: number
  n_points?: number
  is_out_of_sample: boolean
  series: EquityPoint[]
  benchmark?: { date: string; value: number }[]
}

export interface DrawdownSeries {
  run_id: number
  current_drawdown: number
  max_drawdown: number
  series: { date: string; drawdown: number }[]
}

export interface Signal {
  ticker: string
  composite_score: number
  bias: string
  [key: string]: string | number
}

export interface SignalsResponse {
  as_of: string | null
  n_signals: number
  signals: Signal[]
  note: string
}

export interface LogEntry {
  timestamp: string
  level: string
  source: string
  message: string
}

export interface LogsResponse {
  logs: LogEntry[]
  filter?: string
  n_entries?: number
  note?: string
}

export interface FactorRow {
  ticker: string
  sector: string | null
  composite: number
  [key: string]: string | number | null
}

export interface FactorTable {
  as_of: string
  n_total: number
  universe_size?: number
  factors: string[]
  rows: FactorRow[]
}

export interface CorrelationMatrix {
  factors: string[]
  matrix: Record<string, Record<string, number>>
  note: string
}

export interface ICRecord {
  factor: string
  horizon_days: number
  mean_ic: number
  ic_ir: number
  hit_rate: number
  t_stat: number
  p_value: number
  n_periods: number
}

export interface FactorICResponse {
  horizons: number[]
  ic_by_factor: Record<string, ICRecord[]>
  note: string
}

export interface FactorListResponse {
  core_factors: string[]
  production_weights: Record<string, number>
  production_orientations: Record<string, number>
  available: { name: string; doc: string }[]
  note: string
}

export interface TickerDetail {
  ticker: string
  sector: string | null
  as_of: string | null
  current: Record<string, number | null>
  composite: number | null
  price_history: { date: string; close: number }[]
  factor_history: Record<string, (number | null)[] | string[]>
}

export interface BacktestRunSummary {
  id: number
  name: string
  engine_type: string
  is_walk_forward: boolean
  start_date: string
  end_date: string
  sharpe: number | null
  sharpe_oos: number | null
  max_drawdown: number | null
  total_return: number | null
  n_trades: number | null
  runtime_ms: number | null
  created_at: string | null
}

export interface BacktestMetrics {
  run_id: number
  name: string
  is_walk_forward: boolean
  engine: string
  period: { start: string; end: string }
  runtime_ms: number | null
  config: Record<string, unknown>
  metrics: Record<string, any>
  headline: {
    sharpe: number | null
    max_drawdown: number | null
    total_return: number | null
    win_rate: number | null
    n_trades: number | null
  }
  validation_note: string
}

export interface Fold {
  fold: number
  train: { start: string; end: string }
  test: { start: string; end: string }
  sharpe_is: number | null
  sharpe_oos: number | null
  return_oos: number | null
  max_drawdown_oos: number | null
  selected_params: Record<string, unknown> | null
}

export interface FoldsResponse {
  run_id: number
  n_folds: number
  positive_folds: number
  folds: Fold[]
}

export interface TradeRow {
  id: number
  ticker: string
  side: string
  entry_date: string
  exit_date: string | null
  entry_price: number | null
  exit_price: number | null
  pnl_pct: number | null
  pnl_abs: number | null
  holding_days: number | null
  status: string
}

export interface TradesResponse {
  run_id: number
  page: number
  page_size: number
  total: number
  total_pages: number
  filter: string
  trades: TradeRow[]
}

export interface StrategySpecResponse {
  spec: Record<string, any>
  rationale: Record<string, string>
}

export interface RiskSummary {
  circuit_breaker: {
    current_drawdown: number
    threshold: number
    remaining_pct: number
    status: string
    peak_equity?: number
    current_equity?: number
  }
  exposure: {
    gross_exposure: number
    net_exposure: number
    long_exposure: number
    short_exposure: number
    n_positions: number
    n_long: number
    n_short: number
    largest_position: number
    concentration_hhi: number
  }
  var: Record<string, any>
  portfolio_volatility: number
  as_of: string
}

export interface SectorExposure {
  sector: string
  long: number
  short: number
  net: number
  gross: number
  n_positions: number
}

export interface ExposureResponse {
  as_of: string
  summary: RiskSummary['exposure']
  sectors: SectorExposure[]
  limits: { max_position_weight: number; max_sector_weight: number }
}

export interface PositionRow {
  ticker: string
  sector?: string | null
  current_weight: number
  vol_adj_target: number
  volatility: number
  drift: number
  risk_status: string
}

export interface PositionsResponse {
  as_of: string
  n_positions?: number
  positions: PositionRow[]
}

export interface Breach {
  type: string
  severity: string
  subject: string
  message: string
  value: number
  limit: number
}

export interface BreachesResponse {
  as_of: string
  active_breaches: Breach[]
  n_active: number
  history: { id: number; type: string; severity: string; message: string }[]
}

export interface SystemStatus {
  system_integrity: string
  uptime_pct: number | null
  uptime_window_days: number
  total_job_runs: number
  failed_job_runs: number
  active_jobs: number
  configured_jobs: number
  last_sync_seconds_ago: number | null
  last_sync_job: string | null
  coverage: {
    total_rows: number
    n_tickers: number
    n_securities: number
    first_date: string | null
    last_date: string | null
    years_of_history: number
  }
}

export interface JobStream {
  job_name: string
  last_run: string | null
  last_status: string
  last_duration_ms: number | null
  last_rows_written: number
  uptime_pct: number | null
  total_runs: number
}

export interface JobsResponse {
  streams: JobStream[]
  recent_runs: {
    id: number
    job_name: string
    status: string
    started_at: string | null
    duration_ms: number | null
    rows_written: number
    error: string | null
  }[]
}

export interface IngestionStats {
  tickers_updated: number
  rows_written_last_run: number
  bytes_processed_last_run: number
  duration_ms: number | null
  fetch_success_rate: number | null
  source: string
  n_securities: number
  latest_data_date: string | null
  coverage: SystemStatus['coverage']
  cleaning: Record<string, number | null>
  failed_tickers: number
  note: string
}

export interface ApiMetrics {
  live: {
    n_requests: number
    p50_ms: number | null
    p95_ms: number | null
    p99_ms: number | null
    mean_ms: number | null
    max_ms?: number | null
    error_rate: number
  }
  total_requests_logged: number
  target_p95_ms: number
  meets_target: boolean | null
  by_endpoint: { endpoint: string; requests: number; mean_ms: number; max_ms: number }[]
}

export interface AnalystStatus {
  active_provider: string
  is_template: boolean
  configured_chain: string[]
  available: string[]
  cache_ttl_seconds: number
  note: string
}

export interface AnalystCitation {
  title: string
  description: string
  points: string[]
  source: string
}

export interface AnalystMetric {
  key: string
  label: string
  score: number
  band: 'STRONG' | 'ADEQUATE' | 'WEAK' | 'FAILING'
  value: number | null
  reasons: string
  gaps: string
  citations: AnalystCitation[]
}

export interface AnalystReport {
  run_id: number
  run_name: string
  overall_score: number
  verdict: string
  summary: string
  metrics: AnalystMetric[]
  provider: string
  model: string
  generated_at: string
  universe_size: number
  is_template: boolean
  notes: string[]
  cached: boolean
}

export interface SystemInfo {
  python: string
  platform: string
  data_source: string
  benchmark: string
  history_years: number
  trading_days_per_year: number
  costs: { commission_bps: number; slippage_bps: number }
  risk_limits: Record<string, number>
}
