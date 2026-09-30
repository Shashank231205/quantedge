/**
 * Screen 8 — Microstructure.
 *
 * Short-horizon research on Binance futures top-of-book and trade data: how
 * far order-book features predict the next minute, where that edge lives, and
 * whether it survives the cost of executing on it. Every figure is read from
 * the published study; out-of-sample results are labelled as such.
 */

import { useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type MicroExecution, type MicroPaired } from '../lib/api'
import { useApi } from '../hooks/useApi'
import { compact, num, pct, shortDate } from '../lib/format'
import {
  Cell,
  Empty,
  ErrorState,
  Loading,
  Panel,
  Provenance,
  StatTile,
  Tabs,
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
const FEATURE_COLOURS: Record<string, string> = {
  qi: '#7EE3B0',
  ofi_1: '#9BA9F0',
  ofi_5: '#8FA8E8',
  ofi_30: '#5F6FB0',
  tfi_5: '#E8B84B',
  tfi_30: '#A8853A',
  past_ret_5: '#F0736A',
}

function bps(value: number | null | undefined, digits = 2): string {
  return value == null ? '—' : `${num(value, digits)} bps`
}

function significance(p: MicroPaired): string {
  if (p.t_stat == null) return 'identical'
  return `t=${num(p.t_stat, 1)}`
}

function IcDecay({ symbol }: { symbol: string }) {
  const { data, loading, error } = useApi(() => api.microStudy(symbol), [symbol])
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No study" />

  const horizons = [...new Set(data.ic_by_horizon.map((r) => r.horizon_s))]
  const points = horizons.map((h) => {
    const row: Record<string, number> = { horizon: h }
    data.ic_by_horizon.filter((r) => r.horizon_s === h).forEach((r) => (row[r.feature] = r.ic))
    return row
  })

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="horizon" tick={AXIS} tickFormatter={(h) => `${h}s`} />
            <YAxis tick={AXIS} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} labelFormatter={(h) => `${h}s ahead`} />
            {data.features.map((f) => (
              <Line
                isAnimationActive={false}
                key={f}
                type="monotone"
                dataKey={f}
                stroke={FEATURE_COLOURS[f] ?? '#6B6B6B'}
                strokeWidth={f === 'qi' ? 2 : 1.25}
                dot={{ r: 2 }}
                name={f}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Provenance>
        Spearman IC against the forward mid return, sampled every h seconds so no two labels
        overlap. Queue imbalance (qi) leads at every horizon and decays with it.
      </Provenance>
    </div>
  )
}

function ExplainVsPredict({ symbol }: { symbol: string }) {
  const { data, loading, error } = useApi(() => api.microStudy(symbol), [symbol])
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No study" />

  return (
    <div className="h-full flex flex-col">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <Th>Interval</Th>
            <Th align="right">Same-interval R²</Th>
            <Th align="right">Next-interval R²</Th>
          </tr>
        </thead>
        <tbody>
          {data.contemporaneous_vs_predictive.map((r) => (
            <tr key={r.interval_s} className="border-b border-edge/50">
              <Cell>{r.interval_s}s</Cell>
              <Cell align="right" tone="text-mint">{pct(r.r2_contemporaneous, 1)}</Cell>
              <Cell align="right" tone="text-warn">{pct(r.r2_predictive, 2)}</Cell>
            </tr>
          ))}
        </tbody>
      </table>
      <Provenance>
        Order-flow imbalance explains much of the price change it coincides with, and almost none
        of the next one. The first column describes price formation; only the second is a signal.
      </Provenance>
    </div>
  )
}

function Regimes({ symbol }: { symbol: string }) {
  const { data, loading, error } = useApi(() => api.microStudy(symbol), [symbol])
  const [split, setSplit] = useState<'vol_regime' | 'session' | 'spread_regime'>('vol_regime')
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No study" />

  const labels = { vol_regime: 'Volatility', session: 'Session', spread_regime: 'Spread' }
  const byLabel = Object.fromEntries(Object.entries(labels).map(([k, v]) => [v, k])) as Record<
    string,
    typeof split
  >

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b border-edge">
        <Tabs
          options={Object.values(labels)}
          value={labels[split]}
          onChange={(v) => setSplit(byLabel[v])}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <Th>Regime</Th>
              <Th align="right">n</Th>
              <Th align="right">IC qi</Th>
              <Th align="right">IC ofi_5</Th>
              <Th align="right">IC tfi_5</Th>
            </tr>
          </thead>
          <tbody>
            {data.ic_by_regime[split].map((r) => (
              <tr key={r.regime} className="border-b border-edge/50">
                <Cell>{r.regime}</Cell>
                <Cell align="right" tone="text-ink-muted">{compact(r.n)}</Cell>
                <Cell align="right">{num(r.ic_qi, 3)}</Cell>
                <Cell align="right">{num(r.ic_ofi_5, 3)}</Cell>
                <Cell align="right">{num(r.ic_tfi_5, 3)}</Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>10-second horizon. Tercile cut points are descriptive, never traded on.</Provenance>
    </div>
  )
}

function Stability({ symbol }: { symbol: string }) {
  const { data, loading, error } = useApi(() => api.microStability(symbol), [symbol])
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.hourly.length) return <Empty message="No hourly series" />

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.hourly} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="hour" tick={AXIS} tickFormatter={shortDate} minTickGap={40} />
            <YAxis tick={AXIS} tickFormatter={(v) => v.toFixed(2)} />
            <ReferenceLine y={0} stroke="#F0736A" strokeDasharray="3 3" />
            <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
            <Line isAnimationActive={false} type="monotone" dataKey="ic" stroke="#7EE3B0" dot={false} strokeWidth={1} name="hourly IC" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Provenance>
        Queue-imbalance IC, 10s horizon, one point per hour: positive in{' '}
        {pct(data.pct_hours_positive, 0)} of {data.hours} hours, worst hour {num(data.worst_hour_ic, 3)}.
      </Provenance>
    </div>
  )
}

function ExecutionRow({ label, e }: { label: string; e: MicroExecution }) {
  return (
    <tr className="border-b border-edge/50">
      <Cell>{label}</Cell>
      <Cell align="right">{bps(e.policies.market.mean_cost_bps)}</Cell>
      <Cell align="right" tone="text-mint">{bps(e.policies.limit.mean_cost_bps)}</Cell>
      <Cell align="right">{bps(e.policies.signal.mean_cost_bps)}</Cell>
      <Cell align="right" tone="text-ink-muted">{pct(e.limit_detail.fill_rate, 0)}</Cell>
      <Cell align="right" tone="text-danger">{bps(e.limit_detail.fill_markout_bps)}</Cell>
      <Cell align="right" tone="text-ink-muted">{significance(e.signal_detail.vs_best_static)}</Cell>
    </tr>
  )
}

function Execution({ symbol }: { symbol: string }) {
  const { data, loading, error } = useApi(() => api.microStudy(symbol), [symbol])
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No study" />

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0 overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <Th>Wait · fill model</Th>
              <Th align="right">Market</Th>
              <Th align="right">Limit</Th>
              <Th align="right">Signal</Th>
              <Th align="right">Fill rate</Th>
              <Th align="right">Fill markout</Th>
              <Th align="right">Signal vs best</Th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.execution).flatMap(([wait, models]) =>
              Object.entries(models).map(([model, e]) => (
                <ExecutionRow key={`${wait}-${model}`} label={`${wait.replace('wait_', '')} · ${model}`} e={e} />
              )),
            )}
          </tbody>
        </table>
        <table className="w-full text-xs mt-2">
          <thead>
            <tr>
              <Th>Fee tier (maker / taker)</Th>
              <Th align="right">Market</Th>
              <Th align="right">Limit</Th>
              <Th align="right">Signal</Th>
              <Th align="right">Routed to market</Th>
              <Th align="right">Signal − best static</Th>
            </tr>
          </thead>
          <tbody>
            {data.fee_sensitivity.map((t) => (
              <tr key={t.tier} className="border-b border-edge/50">
                <Cell>{`${t.tier} (${t.maker_bps} / ${t.taker_bps})`}</Cell>
                <Cell align="right">{bps(t.market_mean_bps)}</Cell>
                <Cell align="right">{bps(t.limit_mean_bps)}</Cell>
                <Cell align="right">{bps(t.signal_mean_bps)}</Cell>
                <Cell align="right" tone="text-ink-muted">{pct(t.routed_market_pct, 0)}</Cell>
                <Cell
                  align="right"
                  tone={t.signal_vs_best_static.mean_diff_bps < 0 ? 'text-mint' : 'text-ink-muted'}
                >
                  {`${num(t.signal_vs_best_static.mean_diff_bps, 3, true)} (${significance(t.signal_vs_best_static)})`}
                </Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>
        Mean cost vs arrival mid, fees included; lower is better. Signal routing is fitted on day
        d−1 and scored on day d. Conservative fills require the whole price level to trade
        through; optimistic fills assume front of queue. A negative markout is adverse selection.
      </Provenance>
    </div>
  )
}

function CrossAsset() {
  const { data, loading, error } = useApi(() => api.microSummary(), [])
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.cross_asset.length) return <Empty message="No study published" />

  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <Th>Symbol</Th>
            <Th align="right">Book events</Th>
            <Th align="right">QI IC 10s</Th>
            <Th align="right">Days +</Th>
            <Th align="right">OOS IC</Th>
            <Th align="right">OOS folds +</Th>
            <Th align="right">Limit saves</Th>
          </tr>
        </thead>
        <tbody>
          {data.cross_asset.map((r) => (
            <tr key={r.symbol} className="border-b border-edge/50">
              <Cell>{r.symbol}</Cell>
              <Cell align="right" tone="text-ink-muted">{compact(r.book_events)}</Cell>
              <Cell align="right" tone="text-mint">{num(r.qi_ic_10s, 3)}</Cell>
              <Cell align="right">{r.qi_days_positive}</Cell>
              <Cell align="right" tone="text-mint">{num(r.oos_ic_10s, 3)}</Cell>
              <Cell align="right">{r.oos_folds_positive}</Cell>
              <Cell align="right">{bps(r.limit_saving_vs_market_bps)}</Cell>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Microstructure() {
  const summary = useApi(() => api.microSummary(), [])
  const symbols = summary.data?.symbols ?? []
  const [chosen, setChosen] = useState<string | null>(null)
  const symbol = chosen ?? symbols[0] ?? null
  const row = summary.data?.cross_asset.find((r) => r.symbol === symbol)
  const dataset = symbol ? summary.data?.datasets[symbol] : undefined

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-end justify-between gap-3 flex-wrap">
        <div>
          <div className="text-2xs font-mono uppercase tracking-wider text-mint">
            Order Book · {dataset ? `${dataset.venue}, ${dataset.start} to ${dataset.end}` : 'Binance futures'}
          </div>
          <h1 className="text-2xl font-mono tracking-wide text-ink">
            Microstructure &amp; <span className="text-accent">Execution</span>
          </h1>
        </div>
        {symbols.length > 0 && symbol && (
          <Tabs options={symbols} value={symbol} onChange={setChosen} />
        )}
      </div>

      {summary.loading ? (
        <Loading />
      ) : summary.error ? (
        <ErrorState error={summary.error} onRetry={summary.refetch} />
      ) : !symbol || !row ? (
        <Empty message="No microstructure study published" hint="Run: quantedge micro study" />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-5 border border-edge bg-base-panel divide-x divide-edge">
            <StatTile
              label="Quote updates"
              value={compact(row.book_events)}
              sub={`${dataset?.days ?? 0} days · ${compact(dataset?.trades)} trades`}
            />
            <StatTile
              label="QI IC · 10s"
              value={num(row.qi_ic_10s, 3)}
              sub={`positive ${row.qi_days_positive} days`}
              tone="text-mint"
            />
            <StatTile
              label="OFI R² same / next"
              value={`${pct(row.ofi_r2_contemporaneous_10s, 0)} / ${pct(row.ofi_r2_predictive_10s, 1)}`}
              sub="10s intervals"
            />
            <StatTile
              label="Walk-forward OOS IC"
              value={num(row.oos_ic_10s, 3)}
              sub={`${row.oos_folds_positive} test days positive`}
              tone="text-mint"
            />
            <StatTile
              label="Limit vs market"
              value={bps(row.limit_saving_vs_market_bps)}
              sub={`saved · fill rate ${pct(row.limit_fill_rate, 0)}`}
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
            <Panel title="IC by Horizon" badge="NON-OVERLAPPING" className="xl:col-span-2 h-[300px]">
              <IcDecay symbol={symbol} />
            </Panel>
            <Panel title="Explanation vs Prediction" className="h-[300px]">
              <ExplainVsPredict symbol={symbol} />
            </Panel>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
            <Panel title="Hourly Stability" className="xl:col-span-2 h-[280px]">
              <Stability symbol={symbol} />
            </Panel>
            <Panel title="Where It Works" className="h-[280px]">
              <Regimes symbol={symbol} />
            </Panel>
          </div>

          <Panel title="Market vs Limit Execution" badge="WALK-FORWARD OOS" className="h-[420px]">
            <Execution symbol={symbol} />
          </Panel>

          <Panel title="Across Assets" className="h-[180px]">
            <CrossAsset />
          </Panel>
        </>
      )}
    </div>
  )
}
