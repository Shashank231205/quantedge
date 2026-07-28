/**
 * Screen 3 — Backtest Analysis.
 *
 * Metric tiles, equity curve, per-fold walk-forward results and a paginated
 * trade log. The in-sample/out-of-sample distinction is shown explicitly
 * rather than left for the reader to infer.
 */

import { useState } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../lib/api'
import { useApi } from '../hooks/useApi'
import { DASH, money, num, pct, shortDate } from '../lib/format'
import {
  Cell,
  Empty,
  ErrorState,
  Loading,
  Panel,
  Pill,
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

function EquityCurve({ runId }: { runId?: number }) {
  const { data, loading, error } = useApi(() => api.backtestEquity(runId), [runId])

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.series.length) return <Empty message="No equity data" />

  const base = data.series[0].equity
  const points = data.series.map((p) => ({
    date: p.date,
    equity: (p.equity / base - 1) * 100,
  }))

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-3 px-3 pt-2 text-2xs font-mono">
        <span className="text-ink-muted">
          {data.series[0].date} — {data.series[data.series.length - 1].date}
        </span>
        {data.is_out_of_sample && <Pill text="Walk-Forward OOS" tone="mint" />}
      </div>
      <div className="flex-1 min-h-0 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#7EE3B0" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#7EE3B0" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickFormatter={shortDate} minTickGap={60} />
            <YAxis tick={AXIS} tickFormatter={(v) => `${v.toFixed(0)}%`} />
            <Tooltip
              {...tooltipStyle}
              formatter={(v: number) => [`${v.toFixed(2)}%`, 'Return']}
              labelFormatter={shortDate}
            />
            <Area isAnimationActive={false}
              type="monotone"
              dataKey="equity"
              stroke="#7EE3B0"
              strokeWidth={1.5}
              fill="url(#eqFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function FoldTable({ runId }: { runId?: number }) {
  const { data, loading } = useApi(() => api.backtestFolds(runId), [runId])

  if (loading) return <Loading />
  if (!data?.folds.length) return <Empty message="No walk-forward folds" />

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-base-panel">
            <tr>
              <Th>Fold</Th>
              <Th>Test Window</Th>
              <Th align="right">IS</Th>
              <Th align="right">OOS</Th>
              <Th align="right">Return</Th>
            </tr>
          </thead>
          <tbody>
            {data.folds.map((f) => (
              <tr key={f.fold} className="border-b border-edge/40 hover:bg-base-raised">
                <Cell tone="text-ink-muted">#{f.fold}</Cell>
                <Cell tone="text-ink-muted">
                  <span className="text-2xs">
                    {f.test.start} → {f.test.end}
                  </span>
                </Cell>
                <Cell align="right" tone="text-ink-muted">
                  {num(f.sharpe_is, 2)}
                </Cell>
                <Cell
                  align="right"
                  tone={(f.sharpe_oos ?? 0) > 0 ? 'text-mint' : 'text-danger'}
                >
                  {num(f.sharpe_oos, 2)}
                </Cell>
                <Cell
                  align="right"
                  tone={(f.return_oos ?? 0) > 0 ? 'text-mint' : 'text-danger'}
                >
                  {pct(f.return_oos, 1, true)}
                </Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>
        {data.positive_folds}/{data.n_folds} folds positive. Each fold trains on past data only,
        with an embargo gap before its test window.
      </Provenance>
    </div>
  )
}

function TradeLog({ runId }: { runId?: number }) {
  const [page, setPage] = useState(1)
  const [outcome, setOutcome] = useState('ALL')
  const [ticker, setTicker] = useState('')

  const { data, loading } = useApi(
    () => api.trades({ page, pageSize: 12, outcome, ticker: ticker || undefined }),
    [page, outcome, ticker, runId],
  )

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-edge shrink-0">
        <Tabs
          options={['ALL', 'WINS', 'LOSSES']}
          value={outcome}
          onChange={(v) => {
            setOutcome(v)
            setPage(1)
          }}
        />
        <input
          value={ticker}
          onChange={(e) => {
            setTicker(e.target.value.toUpperCase())
            setPage(1)
          }}
          placeholder="FILTER TICKER..."
          className="bg-base-deep border border-edge px-2 py-0.5 text-2xs font-mono uppercase tracking-wider text-ink placeholder:text-ink-faint focus:outline-none focus:border-mint w-36"
        />
      </div>

      <div className="flex-1 overflow-auto min-h-0">
        {loading ? (
          <Loading />
        ) : !data?.trades.length ? (
          <Empty message="No trades match this filter" />
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-base-panel">
              <tr>
                <Th>Entry</Th>
                <Th>Exit</Th>
                <Th>Ticker</Th>
                <Th>Side</Th>
                <Th align="right">P&L %</Th>
                <Th align="right">P&L $</Th>
                <Th align="right">Held</Th>
                <Th>Status</Th>
              </tr>
            </thead>
            <tbody>
              {data.trades.map((t) => (
                <tr key={t.id} className="border-b border-edge/40 hover:bg-base-raised">
                  <Cell tone="text-ink-muted">{t.entry_date}</Cell>
                  <Cell tone="text-ink-muted">{t.exit_date ?? DASH}</Cell>
                  <Cell tone="text-ink">{t.ticker}</Cell>
                  <Cell>
                    <Pill text={t.side} tone={t.side === 'LONG' ? 'mint' : 'danger'} />
                  </Cell>
                  <Cell align="right" tone={(t.pnl_pct ?? 0) >= 0 ? 'text-mint' : 'text-danger'}>
                    {pct(t.pnl_pct, 2, true)}
                  </Cell>
                  <Cell align="right" tone={(t.pnl_abs ?? 0) >= 0 ? 'text-mint' : 'text-danger'}>
                    {money(t.pnl_abs)}
                  </Cell>
                  <Cell align="right" tone="text-ink-muted">
                    {num(t.holding_days, 0)}d
                  </Cell>
                  <Cell>
                    <Pill text={t.status} tone={t.status === 'OPEN' ? 'warn' : 'muted'} />
                  </Cell>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {data && data.total > 0 && (
        <div className="flex items-center justify-between px-3 py-2 border-t border-edge text-2xs font-mono text-ink-faint shrink-0">
          <span>
            {(data.page - 1) * data.page_size + 1}–
            {Math.min(data.page * data.page_size, data.total)} of {data.total.toLocaleString()}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="px-2 py-0.5 border border-edge disabled:opacity-30 hover:border-mint hover:text-mint transition-colors"
            >
              ‹
            </button>
            <span className="px-2 text-ink">
              {data.page} / {data.total_pages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
              disabled={page >= data.total_pages}
              className="px-2 py-0.5 border border-edge disabled:opacity-30 hover:border-mint hover:text-mint transition-colors"
            >
              ›
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default function Backtest() {
  const { data: runs } = useApi(() => api.backtestRuns(), [])
  const [runId, setRunId] = useState<number | undefined>()
  const { data, loading, error } = useApi(() => api.backtestMetrics(runId), [runId])

  const m = data?.metrics ?? {}
  const risk = m.risk ?? {}
  const riskAdj = m.risk_adjusted ?? {}
  const trades = m.trades ?? {}
  const turnover = m.turnover ?? {}
  const deflated = m.deflated_sharpe ?? {}
  const comparison = m.comparison ?? {}

  if (loading) return <Loading label="LOADING BACKTEST" />
  if (error) return <ErrorState error={error} />

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <span className="text-2xs font-mono uppercase tracking-wider text-mint">
            Backtest Engine
          </span>
          <select
            value={runId ?? ''}
            onChange={(e) => setRunId(e.target.value ? Number(e.target.value) : undefined)}
            className="bg-base-panel border border-edge px-2 py-1 text-xs font-mono text-ink focus:outline-none focus:border-mint max-w-md"
          >
            <option value="">Latest walk-forward run</option>
            {runs?.runs.map((r) => (
              <option key={r.id} value={r.id}>
                #{r.id} · {r.name}
              </option>
            ))}
          </select>
        </div>
        <Pill
          text={data?.is_walk_forward ? '✓ WALK-FORWARD VERIFIED' : '⚠ IN-SAMPLE ONLY'}
          tone={data?.is_walk_forward ? 'mint' : 'warn'}
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-6 border border-edge bg-base-panel divide-x divide-edge">
        <StatTile label="Sharpe Ratio" value={num(riskAdj.sharpe_ratio, 2)} tone="text-mint" />
        <StatTile label="Sortino" value={num(riskAdj.sortino_ratio, 2)} tone="text-mint" />
        <StatTile label="Max Drawdown" value={pct(risk.max_drawdown, 2)} tone="text-danger" />
        <StatTile label="Win Rate" value={pct(trades.win_rate, 1)} />
        <StatTile label="Ann. Turnover" value={num(turnover.annual_turnover, 1)} />
        <StatTile label="Avg Hold" value={`${num(trades.avg_holding_days, 1)}d`} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          title="Equity Curve"
          badge={data?.is_walk_forward ? 'OUT-OF-SAMPLE' : 'IN-SAMPLE'}
          badgeTone={data?.is_walk_forward ? 'mint' : 'warn'}
          className="xl:col-span-2 h-[330px]"
        >
          <EquityCurve runId={runId} />
        </Panel>

        <Panel title="Validation" className="h-[330px]" bodyClass="overflow-auto">
          <div className="p-3 space-y-3 text-xs">
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <span className="text-ink-faint">In-sample Sharpe</span>
                <span className="font-mono text-ink">{num(comparison.sharpe_is, 3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-faint">Out-of-sample Sharpe</span>
                <span className="font-mono text-mint">{num(comparison.sharpe_oos, 3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-faint">Degradation</span>
                <span className="font-mono text-ink-muted">
                  {comparison.sharpe_degradation_pct != null
                    ? `${comparison.sharpe_degradation_pct}%`
                    : DASH}
                </span>
              </div>
            </div>

            {comparison.verdict && (
              <p className="text-2xs text-ink-muted leading-relaxed border-l-2 border-mint/40 pl-2">
                {comparison.verdict}
              </p>
            )}

            <div className="pt-2 border-t border-edge space-y-1.5">
              <div className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
                Multiple-testing check
              </div>
              <div className="flex justify-between">
                <span className="text-ink-faint">Configurations tested</span>
                <span className="font-mono text-ink">{deflated.n_trials ?? DASH}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-faint">Deflated Sharpe</span>
                <span
                  className={`font-mono ${deflated.is_significant ? 'text-mint' : 'text-warn'}`}
                >
                  {num(deflated.deflated_sharpe, 3)}
                </span>
              </div>
              {deflated.interpretation && (
                <p className="text-2xs text-ink-faint leading-relaxed">
                  {deflated.interpretation}
                </p>
              )}
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel title="Walk-Forward Folds" className="xl:col-span-1 h-[380px]">
          <FoldTable runId={runId} />
        </Panel>
        <Panel title="Execution Audit Log" className="xl:col-span-2 h-[380px]">
          <TradeLog runId={runId} />
        </Panel>
      </div>
    </div>
  )
}
