/**
 * Screen 1 — Portfolio Dashboard.
 *
 * Every figure comes from the walk-forward run, which the header labels as
 * out-of-sample so it cannot be mistaken for an in-sample result.
 */

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../lib/api'
import { useApi } from '../hooks/useApi'
import {
  DASH,
  clockTime,
  levelTone,
  num,
  pct,
  shortDate,
} from '../lib/format'
import {
  Dot,
  Empty,
  ErrorState,
  Loading,
  Panel,
  Pill,
  Provenance,
  StatTile,
} from '../components/Primitives'

const AXIS = { stroke: '#6B6B6B', fontSize: 10, fontFamily: 'ui-monospace, monospace' }

const tooltipStyle = {
  contentStyle: {
    background: '#232323',
    border: '1px solid #333',
    borderRadius: 0,
    fontSize: 11,
    fontFamily: 'ui-monospace, monospace',
  },
  labelStyle: { color: '#9A9A9A' },
}

function CumulativeReturns() {
  const { data, loading, error, refetch } = useApi(() => api.equityCurve(800), [])

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} onRetry={refetch} />
  if (!data?.series.length)
    return <Empty message="No equity curve yet" hint="Run `make backtest` to generate one." />

  const base = data.series[0].equity
  const benchmark = new Map((data.benchmark ?? []).map((b) => [b.date, b.value]))

  const points = data.series.map((p) => ({
    date: p.date,
    strategy: (p.equity / base - 1) * 100,
    benchmark: benchmark.has(p.date) ? (benchmark.get(p.date)! - 1) * 100 : null,
  }))

  const last = points[points.length - 1]

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-4 px-3 pt-2 text-2xs font-mono">
        <span className="flex items-center gap-1.5 text-ink-muted">
          <span className="w-3 h-px bg-accent" /> STRATEGY {pct(last.strategy / 100, 1, true)}
        </span>
        <span className="flex items-center gap-1.5 text-ink-faint">
          <span className="w-3 h-px bg-ink-faint" /> S&P 500{' '}
          {last.benchmark != null ? pct(last.benchmark / 100, 1, true) : DASH}
        </span>
      </div>
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickFormatter={shortDate} minTickGap={60} />
            <YAxis tick={AXIS} tickFormatter={(v) => `${v.toFixed(0)}%`} />
            <Tooltip
              {...tooltipStyle}
              formatter={(v: number, n: string) => [`${v?.toFixed(2)}%`, n]}
              labelFormatter={shortDate}
            />
            <Line isAnimationActive={false}
              type="monotone"
              dataKey="strategy"
              stroke="#9BA9F0"
              strokeWidth={1.5}
              dot={false}
              name="Strategy"
            />
            <Line isAnimationActive={false}
              type="monotone"
              dataKey="benchmark"
              stroke="#6B6B6B"
              strokeWidth={1}
              dot={false}
              name="S&P 500"
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function DrawdownChart() {
  const { data, loading, error } = useApi(() => api.drawdown(), [])

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.series.length) return <Empty message="No drawdown data" />

  const points = data.series.map((p) => ({ date: p.date, dd: p.drawdown * 100 }))

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-3 pt-2">
        <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
          Portfolio Drawdown
        </span>
        <span className="text-2xs font-mono text-danger">
          {pct(data.current_drawdown, 2)} current · {pct(data.max_drawdown, 2)} worst
        </span>
      </div>
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#F0736A" stopOpacity={0.05} />
                <stop offset="100%" stopColor="#F0736A" stopOpacity={0.35} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickFormatter={shortDate} minTickGap={60} />
            <YAxis tick={AXIS} tickFormatter={(v) => `${v.toFixed(0)}%`} />
            <Tooltip
              {...tooltipStyle}
              formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
              labelFormatter={shortDate}
            />
            <Area isAnimationActive={false}
              type="monotone"
              dataKey="dd"
              stroke="#F0736A"
              strokeWidth={1}
              fill="url(#ddFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function LiveSignals() {
  const { data, loading, error } = useApi(() => api.signals(12), [], 60_000)

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.signals.length) return <Empty message="No signals" />

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-base-panel">
            <tr>
              <th className="px-2.5 py-1.5 text-left text-2xs font-mono uppercase tracking-wider font-normal text-ink-faint border-b border-edge">
                Asset
              </th>
              <th className="px-2.5 py-1.5 text-right text-2xs font-mono uppercase tracking-wider font-normal text-ink-faint border-b border-edge">
                Score
              </th>
              <th className="px-2.5 py-1.5 text-right text-2xs font-mono uppercase tracking-wider font-normal text-ink-faint border-b border-edge">
                Bias
              </th>
            </tr>
          </thead>
          <tbody>
            {data.signals.map((s) => (
              <tr key={s.ticker} className="border-b border-edge/40 hover:bg-base-raised">
                <td className="px-2.5 py-1.5">
                  <div className="font-mono text-ink">{s.ticker}</div>
                  <div className="text-2xs font-mono text-ink-faint">
                    MOM {num(s.momentum_risk_adj as number, 2)}
                  </div>
                </td>
                <td className="px-2.5 py-1.5 text-right font-mono tabular-nums text-ink">
                  {num(s.composite_score, 3)}
                </td>
                <td className="px-2.5 py-1.5 text-right">
                  <Pill text={s.bias} tone={s.bias === 'LONG' ? 'mint' : 'muted'} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>
        Ranking as of {data.as_of ?? DASH}. Scores are lagged one bar, so they reflect only
        information available before the trade date.
      </Provenance>
    </div>
  )
}

function SystemLog() {
  const { data, loading } = useApi(() => api.portfolioLogs(40), [], 15_000)

  if (loading) return <Loading />
  if (!data?.logs.length) return <Empty message="No log records" />

  return (
    <div className="h-full overflow-auto px-3 py-2 space-y-0.5 font-mono text-2xs leading-relaxed">
      {data.logs.map((log, i) => (
        <div key={i} className="flex gap-2">
          <span className="text-ink-faint shrink-0">{clockTime(log.timestamp)}</span>
          <span className={`shrink-0 ${levelTone(log.level)}`}>
            [{log.level.slice(0, 4)}]
          </span>
          <span className="text-ink-muted break-all">{log.message}</span>
        </div>
      ))}
      <div className="text-mint">_</div>
    </div>
  )
}

export default function Dashboard() {
  const { data } = useApi(() => api.portfolioSummary(), [], 30_000)

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 lg:grid-cols-4 border border-edge bg-base-panel divide-x divide-edge">
        <StatTile
          label="Sharpe Ratio (OOS)"
          value={num(data?.sharpe_ratio, 2)}
          sub={`Sortino ${num(data?.sortino_ratio, 2)}`}
          tone="text-mint"
          large
        />
        <StatTile
          label="Annualised Return"
          value={pct(data?.annualized_return, 2, true)}
          sub={`Total ${pct(data?.total_return, 1, true)}`}
          tone="text-mint"
          large
        />
        <StatTile
          label="Volatility (σ)"
          value={pct(data?.volatility, 1)}
          sub={`Calmar ${num(data?.calmar_ratio, 2)}`}
          large
        />
        <StatTile
          label="Max Drawdown"
          value={pct(data?.max_drawdown, 2)}
          sub={data ? `${data.period.start} → ${data.period.end}` : undefined}
          tone="text-danger"
          large
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          title="Cumulative Returns"
          badge={data?.is_out_of_sample ? 'OUT-OF-SAMPLE' : 'IN-SAMPLE'}
          badgeTone={data?.is_out_of_sample ? 'mint' : 'warn'}
          className="xl:col-span-2 h-[340px]"
        >
          <CumulativeReturns />
        </Panel>

        <Panel
          title="Live Signals"
          badge="TOP RANKED"
          className="h-[340px]"
          bodyClass="flex flex-col min-h-0"
        >
          <LiveSignals />
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel title="Drawdown" className="xl:col-span-2 h-[240px]">
          <DrawdownChart />
        </Panel>

        <Panel
          title="System Log"
          badge="LIVE"
          className="h-[240px]"
          actions={<Dot pulse />}
        >
          <SystemLog />
        </Panel>
      </div>
    </div>
  )
}
