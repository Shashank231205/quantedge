/**
 * Application shell: fixed left navigation, global KPI header, content area.
 * Matches the Figma layout — the nav and header persist across all screens.
 */

import { NavLink, Outlet } from 'react-router-dom'
import { api } from '../lib/api'
import { useApi, useWebSocket } from '../hooks/useApi'
import { DASH, num, pct } from '../lib/format'
import { Dot } from '../components/Primitives'

const NAV = [
  { to: '/', label: 'Dashboard', icon: '▤', end: true },
  { to: '/factors', label: 'Factors', icon: '◈' },
  { to: '/backtest', label: 'Backtest', icon: '◱' },
  { to: '/risk', label: 'Risk', icon: '◉' },
  { to: '/pipeline', label: 'Pipeline', icon: '⛁' },
  { to: '/analyst', label: 'Analyst', icon: '☝' },
]

function KpiHeader() {
  // Poll every 30s; the underlying data changes at most daily, so anything
  // faster would be motion without information.
  const { data } = useApi(() => api.portfolioSummary(), [], 30_000)
  const { connected } = useWebSocket()

  const items = [
    {
      label: 'Portfolio Sharpe',
      value: num(data?.sharpe_ratio, 2),
      tone: (data?.sharpe_ratio ?? 0) > 1 ? 'text-mint' : 'text-ink',
    },
    {
      label: 'Ann. Return',
      value: pct(data?.annualized_return, 1, true),
      tone: (data?.annualized_return ?? 0) > 0 ? 'text-mint' : 'text-danger',
    },
    {
      label: 'Max Drawdown',
      value: pct(data?.max_drawdown, 1),
      tone: 'text-danger',
    },
    {
      label: 'Pipeline Status',
      value: data?.pipeline.status ?? DASH,
      tone: data?.pipeline.status === 'OPERATIONAL' ? 'text-mint' : 'text-warn',
      sub: data?.pipeline.uptime_pct != null ? `${data.pipeline.uptime_pct}%` : 'NO RUNS',
    },
  ]

  return (
    <header className="h-14 border-b border-edge bg-base-deep flex items-stretch shrink-0">
      <div className="flex-1 flex items-stretch divide-x divide-edge min-w-0">
        {items.map((item) => (
          <div key={item.label} className="px-4 flex flex-col justify-center min-w-0 flex-1">
            <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint truncate">
              {item.label}
            </span>
            <span className={`text-sm font-mono tabular-nums ${item.tone} truncate`}>
              {item.value}
              {item.sub && <span className="ml-1.5 text-2xs text-ink-faint">{item.sub}</span>}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3 px-4 border-l border-edge shrink-0">
        {data?.is_out_of_sample && (
          <span className="hidden lg:inline text-2xs font-mono uppercase tracking-wider px-1.5 py-0.5 border border-mint/30 bg-mint/10 text-mint">
            Walk-Forward OOS
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <Dot tone={connected ? 'mint' : 'faint'} pulse={connected} />
          <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
            {connected ? 'LIVE' : 'OFFLINE'}
          </span>
        </span>
      </div>
    </header>
  )
}

function Sidebar() {
  const { data } = useApi(() => api.systemStatus(), [], 60_000)

  return (
    <aside className="w-48 shrink-0 border-r border-edge bg-base-deep flex flex-col">
      <div className="h-14 flex items-center gap-2 px-4 border-b border-edge shrink-0">
        <span className="h-5 w-5 grid place-items-center border border-mint/50 text-mint text-2xs">
          ◈
        </span>
        <span className="font-mono text-sm tracking-[0.2em] text-ink">QUANTEDGE</span>
      </div>

      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-4 py-2 text-sm transition-colors border-l-2 ${
                isActive
                  ? 'border-mint bg-mint/10 text-mint'
                  : 'border-transparent text-ink-muted hover:text-ink hover:bg-base-panel'
              }`
            }
          >
            <span className="text-xs opacity-70">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-edge p-3 space-y-2 shrink-0">
        <div className="space-y-0.5">
          <div className="text-2xs font-mono uppercase tracking-wider text-ink-faint">
            Universe
          </div>
          <div className="text-2xs font-mono text-ink-muted">
            {data ? `${data.coverage.n_tickers} tickers` : DASH}
          </div>
          <div className="text-2xs font-mono text-ink-muted">
            {data ? `${data.coverage.years_of_history} yrs · ${(data.coverage.total_rows / 1000).toFixed(0)}k rows` : DASH}
          </div>
        </div>
        <div className="pt-2 border-t border-edge">
          <div className="text-2xs font-mono text-ink-faint">S. KASAMSHETTY</div>
          <div className="text-2xs font-mono text-ink-faint opacity-60">RESEARCH</div>
        </div>
      </div>
    </aside>
  )
}

export default function Shell() {
  return (
    <div className="h-screen flex bg-base text-ink overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <KpiHeader />
        <main className="flex-1 overflow-auto p-3">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
