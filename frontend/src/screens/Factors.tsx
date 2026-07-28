/**
 * Screen 2 — Factor Explorer.
 *
 * Sortable factor table, ticker detail pane with an empty state, cross-factor
 * correlation matrix, and the IC decay curve that justified the strategy's
 * rebalancing frequency.
 */

import { useState } from 'react'
import {
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
import { DASH, clockTime, levelTone, num, shortDate } from '../lib/format'
import {
  Cell,
  Empty,
  ErrorState,
  Loading,
  Panel,
  Provenance,
  Th,
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

/** Blue for below-average, mint for above — mirrors the Figma heatmap. */
function heatTone(value: number): string {
  if (value >= 0.99) return 'bg-mint/25 text-mint'
  if (value > 0.15) return 'bg-mint/15 text-mint'
  if (value < -0.15) return 'bg-accent/20 text-accent'
  return 'bg-base-raised text-ink-muted'
}

function CorrelationMatrix() {
  const { data, loading, error } = useApi(() => api.factorCorrelation(), [])

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No correlation data" />

  const short = (f: string) =>
    f.replace('momentum_risk_adj', 'MOM').replace('mean_reversion', 'REV')
      .replace('volatility', 'VOL').replace('momentum', 'MOM').toUpperCase().slice(0, 4)

  return (
    <div className="p-3">
      <table className="w-full text-2xs font-mono">
        <thead>
          <tr>
            <th />
            {data.factors.map((f) => (
              <th key={f} className="px-1 py-1 text-ink-faint font-normal text-center">
                {short(f)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.factors.map((a) => (
            <tr key={a}>
              <td className="px-1 py-1 text-ink-faint text-right pr-2">{short(a)}</td>
              {data.factors.map((b) => {
                const v = data.matrix[a]?.[b] ?? 0
                return (
                  <td key={b} className="px-0.5 py-0.5">
                    <div className={`py-1.5 text-center tabular-nums ${heatTone(v)}`}>
                      {v.toFixed(2)}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-2xs font-mono text-ink-faint mt-2 leading-relaxed">
        Off-diagonal values near zero mean the factors carry independent information, which is
        what makes blending them worthwhile.
      </p>
    </div>
  )
}

function ICDecay() {
  const { data, loading, error } = useApi(() => api.factorIC(), [])

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No IC data" />

  const names = Object.keys(data.ic_by_factor)
  const points = data.horizons.map((h, i) => {
    const row: Record<string, number | string> = { horizon: `${h}d` }
    names.forEach((n) => {
      row[n] = data.ic_by_factor[n][i]?.mean_ic ?? 0
    })
    return row
  })

  const colours: Record<string, string> = {
    momentum_risk_adj: '#7EE3B0',
    momentum: '#7EE3B0',
    volatility: '#9BA9F0',
    mean_reversion: '#E8B84B',
    composite: '#E8E8E8',
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="horizon" tick={AXIS} />
            <YAxis tick={AXIS} tickFormatter={(v) => v.toFixed(3)} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
            {names.map((n) => (
              <Line isAnimationActive={false}
                key={n}
                type="monotone"
                dataKey={n}
                stroke={colours[n] ?? '#6B6B6B'}
                strokeWidth={n === 'composite' ? 2 : 1.25}
                dot={{ r: 2 }}
                name={n}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Provenance>
        Spearman IC of the one-bar-lagged signal against forward returns. Momentum strengthens
        with horizon, which is why the strategy rebalances monthly rather than daily.
      </Provenance>
    </div>
  )
}

function TickerDetail({ ticker }: { ticker: string | null }) {
  const { data, loading } = useApi(
    () => (ticker ? api.tickerDetail(ticker) : Promise.resolve(null as any)),
    [ticker],
  )

  if (!ticker)
    return (
      <Empty
        message="NO TICKER SELECTED"
        hint="Click a row to inspect factor history and price."
      />
    )
  if (loading) return <Loading />
  if (!data) return <Empty message="No detail available" />

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b border-edge">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-lg text-ink">{data.ticker}</span>
          <span className="text-2xs font-mono text-ink-faint">{data.sector ?? DASH}</span>
        </div>
        <div className="flex gap-3 mt-1 flex-wrap">
          {Object.entries(data.current).map(([k, v]) => (
            <span key={k} className="text-2xs font-mono text-ink-muted">
              {k.replace('momentum_risk_adj', 'MOM').replace('mean_reversion', 'REV')
                .replace('volatility', 'VOL').toUpperCase()}{' '}
              <span className="text-ink">{num(v as number | null, 3)}</span>
            </span>
          ))}
        </div>
      </div>
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data.price_history}
            margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
          >
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickFormatter={shortDate} minTickGap={50} />
            <YAxis tick={AXIS} domain={['auto', 'auto']} />
            <Tooltip {...tooltipStyle} labelFormatter={shortDate} />
            <Line isAnimationActive={false} type="monotone" dataKey="close" stroke="#7EE3B0" strokeWidth={1.25} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function DiagnosticLogs() {
  const { data } = useApi(() => api.systemLogs(30), [], 15_000)
  if (!data?.logs.length) return <Empty message="No diagnostic output" />

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

export default function Factors() {
  const [sortBy, setSortBy] = useState('composite')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<string | null>(null)

  const { data, loading, error, refetch } = useApi(
    () => api.factorTable({ limit: 150, sortBy, search: search || undefined }),
    [sortBy, search],
  )

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="text-2xs font-mono uppercase tracking-wider text-mint">
            Universe Analysis
          </div>
          <h1 className="text-2xl font-mono tracking-wide text-ink">FACTOR EXPLORER</h1>
        </div>
        <div className="flex items-center gap-4 text-right">
          <div>
            <div className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
              Universe Size
            </div>
            <div className="font-mono text-ink">{data?.universe_size ?? DASH}</div>
          </div>
          <div>
            <div className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
              As Of
            </div>
            <div className="font-mono text-ink">{data?.as_of ?? DASH}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          className="xl:col-span-2 h-[420px]"
          title={
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value.toUpperCase())}
              placeholder="FILTER TICKER..."
              className="bg-base-deep border border-edge px-2 py-1 text-2xs font-mono uppercase tracking-wider text-ink placeholder:text-ink-faint focus:outline-none focus:border-mint w-44"
            />
          }
          badge={`${data?.n_total ?? 0} NAMES`}
          bodyClass="overflow-auto"
        >
          {loading ? (
            <Loading />
          ) : error ? (
            <ErrorState error={error} onRetry={refetch} />
          ) : !data?.rows.length ? (
            <Empty message="No matching tickers" />
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-base-panel z-10">
                <tr>
                  <Th onClick={() => setSortBy('ticker')} active={sortBy === 'ticker'}>
                    Ticker
                  </Th>
                  {data.factors.map((f) => (
                    <Th key={f} align="right" onClick={() => setSortBy(f)} active={sortBy === f}>
                      {f.replace('momentum_risk_adj', 'MOMENTUM')
                        .replace('mean_reversion', 'MEAN-REV')
                        .replace('volatility', 'VOLATILITY')}
                    </Th>
                  ))}
                  <Th align="right" onClick={() => setSortBy('composite')} active={sortBy === 'composite'}>
                    Composite
                  </Th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr
                    key={row.ticker}
                    onClick={() => setSelected(row.ticker)}
                    className={`border-b border-edge/40 cursor-pointer hover:bg-base-raised ${
                      selected === row.ticker ? 'bg-mint/10' : ''
                    }`}
                  >
                    <Cell tone="text-ink">
                      <span className="flex items-center gap-1.5">
                        <span className="h-1 w-1 rounded-full bg-mint" />
                        {row.ticker}
                      </span>
                    </Cell>
                    {data.factors.map((f) => {
                      const v = row[f] as number | null
                      return (
                        <Cell
                          key={f}
                          align="right"
                          tone={
                            v == null ? 'text-ink-faint'
                              : v > 0.66 ? 'text-mint'
                              : v < 0.33 ? 'text-danger'
                              : 'text-ink-muted'
                          }
                        >
                          {num(v, 3)}
                        </Cell>
                      )
                    })}
                    <Cell align="right" tone="text-accent">
                      {num(row.composite, 3)}
                    </Cell>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        <Panel title="Ticker Detail" className="h-[420px]">
          <TickerDetail ticker={selected} />
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel title="Cross-Factor Correlation" className="h-[280px]" bodyClass="overflow-auto">
          <CorrelationMatrix />
        </Panel>
        <Panel title="Information Coefficient Decay" className="h-[280px]">
          <ICDecay />
        </Panel>
        <Panel title="Pipeline Diagnostics" badge="LIVE" className="h-[280px]">
          <DiagnosticLogs />
        </Panel>
      </div>
    </div>
  )
}
