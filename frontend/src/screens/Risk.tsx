/**
 * Screen 4 — Risk Monitor.
 *
 * Drawdown circuit-breaker gauge, sector concentration, VaR, and the position
 * sizing matrix showing current weight against the volatility-adjusted target.
 */

import { api } from '../lib/api'
import { useApi } from '../hooks/useApi'
import { DASH, clockTime, levelTone, money, num, pct, statusTone } from '../lib/format'
import {
  Cell,
  Empty,
  ErrorState,
  Loading,
  Panel,
  Pill,
  Provenance,
  StatTile,
  Th,
} from '../components/Primitives'

const SECTOR_COLOURS = ['#7EE3B0', '#9BA9F0', '#E8B84B', '#8FA8E8', '#4FBF89', '#F0736A', '#6B6B6B']

function CircuitGauge() {
  const { data, loading, error } = useApi(() => api.riskSummary(), [], 60_000)

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No risk data" />

  const cb = data.circuit_breaker
  const remaining = Math.max(0, Math.min(100, cb.remaining_pct))
  const radius = 52
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - remaining / 100)
  const tone = remaining > 50 ? '#7EE3B0' : remaining > 25 ? '#E8B84B' : '#F0736A'

  return (
    <div className="h-full flex flex-col items-center justify-center gap-2 p-3">
      <div className="relative">
        <svg width="140" height="140" className="-rotate-90">
          <circle cx="70" cy="70" r={radius} fill="none" stroke="#2A2A2A" strokeWidth="8" />
          <circle
            cx="70"
            cy="70"
            r={radius}
            fill="none"
            stroke={tone}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="butt"
            className="transition-all duration-700"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-2xl" style={{ color: tone }}>
            {remaining.toFixed(1)}%
          </span>
          <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
            Remaining
          </span>
        </div>
      </div>
      <div className="w-full space-y-1 text-2xs font-mono">
        <div className="flex justify-between">
          <span className="text-ink-faint">Threshold</span>
          <span className="text-ink-muted">{pct(cb.threshold, 2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-faint">Current DD</span>
          <span className="text-danger">{pct(cb.current_drawdown, 2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-faint">Status</span>
          <span className={statusTone(cb.status)}>{cb.status}</span>
        </div>
      </div>
    </div>
  )
}

function SectorTreemap() {
  const { data, loading, error } = useApi(() => api.riskExposure(), [], 60_000)

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.sectors.length)
    return <Empty message="No sector exposure" hint="The book holds no positions." />

  const maxGross = Math.max(
    ...data.sectors.map((s) => Math.abs(s.gross)),
    data.limits.max_sector_weight,
  )

  return (
    <div className="h-full flex flex-col">
      {/* A plain bar layout rather than a Recharts Treemap: with only a
          handful of sectors it reads more clearly, labels never collide, and
          the gross-vs-cap comparison is directly visible. */}
      <div className="flex-1 min-h-0 overflow-auto p-3 space-y-1.5">
        {data.sectors.map((s, i) => {
          const colour = SECTOR_COLOURS[i % SECTOR_COLOURS.length]
          const width = Math.min(100, (s.gross / maxGross) * 100)
          const overCap = s.gross > data.limits.max_sector_weight
          return (
            <div key={s.sector} className="space-y-0.5">
              <div className="flex items-baseline justify-between gap-2 text-2xs font-mono">
                <span className="text-ink-muted truncate">{s.sector}</span>
                <span className={overCap ? 'text-danger' : 'text-ink'}>
                  {pct(s.gross, 1)}
                  <span className="text-ink-faint ml-1.5">{s.n_positions}n</span>
                </span>
              </div>
              <div className="h-3 bg-base-deep relative">
                <div
                  className="h-full transition-all duration-500"
                  style={{ width: `${width}%`, backgroundColor: colour, opacity: 0.55 }}
                />
                {/* Cap marker, so a breach is visible rather than inferred. */}
                <div
                  className="absolute top-0 h-full w-px bg-danger/70"
                  style={{
                    left: `${Math.min(100, (data.limits.max_sector_weight / maxGross) * 100)}%`,
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
      <Provenance>
        Sector cap {pct(data.limits.max_sector_weight, 0)} · position cap{' '}
        {pct(data.limits.max_position_weight, 0)}
      </Provenance>
    </div>
  )
}

function BreachBanner() {
  const { data } = useApi(() => api.riskBreaches(), [], 30_000)
  if (!data || data.n_active === 0) return null

  return (
    <div className="border border-danger/40 bg-danger/10 px-3 py-2 space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-danger">⚠</span>
        <span className="text-sm text-danger font-medium">
          {data.n_active} Limit Breach{data.n_active > 1 ? 'es' : ''} Detected
        </span>
      </div>
      {data.active_breaches.slice(0, 3).map((b, i) => (
        <p key={i} className="text-2xs font-mono text-ink-muted pl-5">
          {b.message}
        </p>
      ))}
    </div>
  )
}

function PositionMatrix() {
  const { data, loading } = useApi(() => api.riskPositions(40), [], 60_000)

  if (loading) return <Loading />
  if (!data?.positions.length) return <Empty message="No open positions" />

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-base-panel">
            <tr>
              <Th>Ticker</Th>
              <Th>Sector</Th>
              <Th align="right">Weight</Th>
              <Th align="right">Vol-Adj Target</Th>
              <Th align="right">Drift</Th>
              <Th align="center">Risk</Th>
            </tr>
          </thead>
          <tbody>
            {data.positions.map((p) => (
              <tr key={p.ticker} className="border-b border-edge/40 hover:bg-base-raised">
                <Cell tone="text-ink">{p.ticker}</Cell>
                <Cell tone="text-ink-faint">
                  <span className="text-2xs">{p.sector ?? DASH}</span>
                </Cell>
                <Cell align="right">{pct(p.current_weight, 2)}</Cell>
                <Cell align="right" tone="text-ink-muted">
                  {pct(p.vol_adj_target, 2)}
                </Cell>
                <Cell align="right" tone={p.drift > 0 ? 'text-mint' : 'text-danger'}>
                  {pct(p.drift, 2, true)}
                </Cell>
                <Cell align="center">
                  <span
                    className={`inline-block h-1.5 w-1.5 rounded-full ${
                      p.risk_status === 'HIGH'
                        ? 'bg-danger'
                        : p.risk_status === 'MEDIUM'
                        ? 'bg-warn'
                        : 'bg-mint'
                    }`}
                  />
                </Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>
        Target weights are inverse-volatility scaled, so each holding contributes comparable risk
        rather than comparable dollars.
      </Provenance>
    </div>
  )
}

function RiskLog() {
  const { data } = useApi(() => api.riskLogs(30), [], 15_000)
  if (!data?.logs.length) return <Empty message="No risk-engine output" />

  return (
    <div className="h-full overflow-auto px-3 py-2 space-y-0.5 font-mono text-2xs">
      {data.logs.map((log, i) => (
        <div key={i} className="flex gap-2">
          <span className="text-ink-faint shrink-0">{clockTime(log.timestamp)}</span>
          <span className={`shrink-0 ${levelTone(log.level)}`}>[{log.level.slice(0, 4)}]</span>
          <span className="text-ink-muted break-all">{log.message}</span>
        </div>
      ))}
    </div>
  )
}

export default function Risk() {
  const { data } = useApi(() => api.riskSummary(), [], 60_000)
  const var95 = data?.var?.var_95 ?? {}

  return (
    <div className="space-y-3">
      <BreachBanner />

      <div>
        <div className="text-2xs font-mono uppercase tracking-wider text-mint">Risk Perimeter</div>
        <h1 className="text-2xl font-mono tracking-wide text-ink">
          Systemic Exposure &amp; <span className="text-accent">Capital Preservation</span>
        </h1>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 border border-edge bg-base-panel divide-x divide-edge">
        <StatTile
          label="Gross Exposure"
          value={num(data?.exposure.gross_exposure, 2)}
          sub={`${data?.exposure.n_positions ?? 0} positions`}
        />
        <StatTile label="Net Exposure" value={num(data?.exposure.net_exposure, 2)} />
        <StatTile
          label="Portfolio Vol"
          value={pct(data?.portfolio_volatility, 1)}
          sub="60-day realised"
        />
        <StatTile
          label="VaR (95%)"
          value={money(var95.historical_usd)}
          sub="per $1M / day"
          tone="text-warn"
        />
        <StatTile
          label="CVaR (95%)"
          value={money(var95.cvar_usd)}
          sub="expected shortfall"
          tone="text-danger"
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-3">
        <Panel title="Max Drawdown Circuit" badge="ACTIVE" className="h-[300px]">
          <CircuitGauge />
        </Panel>
        <Panel title="Sector Concentration" className="xl:col-span-2 h-[300px]">
          <SectorTreemap />
        </Panel>
        <Panel title="Risk Engine Log" badge="LIVE" className="h-[300px]">
          <RiskLog />
        </Panel>
      </div>

      <Panel title="Position Sizing Matrix" className="h-[360px]">
        <PositionMatrix />
      </Panel>
    </div>
  )
}
