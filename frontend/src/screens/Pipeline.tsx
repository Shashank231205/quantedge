/**
 * Screen 5 — System Health.
 *
 * The Figma for this screen showed a Binance crypto feed, a FIX order-flow
 * gateway, a sentiment NLP stream and 128 auto-scaling workers. This platform
 * has none of those, so every panel is bound to telemetry it genuinely
 * produces: the four scheduled jobs, real row counts, measured request
 * latency, and the application's own log tail.
 *
 * The success rate reads 92.5% rather than 100% because 46 index members were
 * delisted or acquired and the provider no longer serves their history. That
 * is the honest number, and the panel says why.
 */

import { useState } from 'react'
import { api } from '../lib/api'
import { useApi } from '../hooks/useApi'
import {
  DASH,
  ago,
  bytes,
  clockTime,
  compact,
  levelTone,
  ms,
  num,
  statusTone,
} from '../lib/format'
import {
  Cell,
  Dot,
  Empty,
  ErrorState,
  Loading,
  Panel,
  Provenance,
  StatTile,
  Tabs,
  Th,
} from '../components/Primitives'

function JobStreams() {
  const { data, loading, error } = useApi(() => api.systemJobs(), [], 20_000)

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data?.streams.length) return <Empty message="No jobs configured" />

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-base-panel">
            <tr>
              <Th>Job</Th>
              <Th>Last Run</Th>
              <Th align="right">Duration</Th>
              <Th align="right">Rows</Th>
              <Th align="right">Uptime</Th>
              <Th align="center">Status</Th>
            </tr>
          </thead>
          <tbody>
            {data.streams.map((s) => (
              <tr key={s.job_name} className="border-b border-edge/40 hover:bg-base-raised">
                <Cell tone="text-ink">{s.job_name.toUpperCase()}</Cell>
                <Cell tone="text-ink-faint">
                  <span className="text-2xs">
                    {s.last_run ? new Date(s.last_run).toLocaleString('en-US', { hour12: false }) : DASH}
                  </span>
                </Cell>
                <Cell align="right" tone="text-ink-muted">
                  {ms(s.last_duration_ms)}
                </Cell>
                <Cell align="right" tone="text-ink-muted">
                  {compact(s.last_rows_written)}
                </Cell>
                <Cell align="right" tone={(s.uptime_pct ?? 0) >= 99 ? 'text-mint' : 'text-warn'}>
                  {s.uptime_pct != null ? `${s.uptime_pct}%` : DASH}
                </Cell>
                <Cell align="center" tone={statusTone(s.last_status)}>
                  {s.last_status}
                </Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>
        These are the jobs this platform actually schedules. Uptime is successes divided by
        completed runs in the <code>job_runs</code> table — fail one and this number moves.
      </Provenance>
    </div>
  )
}

function IngestionEngine() {
  const { data, loading, error } = useApi(() => api.ingestionStats(), [], 30_000)

  if (loading) return <Loading />
  if (error) return <ErrorState error={error} />
  if (!data) return <Empty message="No ingestion data" />

  const c = data.coverage
  const cleaning = data.cleaning

  return (
    <div className="h-full flex flex-col overflow-auto">
      <div className="grid grid-cols-3 divide-x divide-edge border-b border-edge">
        <StatTile label="Tickers Updated" value={compact(data.tickers_updated)} />
        <StatTile label="Rows Last Run" value={compact(data.rows_written_last_run)} />
        <StatTile
          label="Fetch Success"
          value={data.fetch_success_rate != null ? `${data.fetch_success_rate}%` : DASH}
          tone={(data.fetch_success_rate ?? 0) >= 90 ? 'text-mint' : 'text-warn'}
        />
      </div>

      <div className="p-3 space-y-2 text-2xs font-mono">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <Row label="Source" value={data.source} />
          <Row label="Data volume" value={bytes(data.bytes_processed_last_run)} />
          <Row label="Ingest duration" value={ms(data.duration_ms)} />
          <Row label="Latest bar" value={data.latest_data_date ?? DASH} />
          <Row label="Total rows" value={compact(c.total_rows)} />
          <Row label="History" value={`${c.years_of_history} yrs`} />
          <Row label="Universe" value={`${c.n_tickers} priced / ${c.n_securities} known`} />
          <Row label="Failed tickers" value={String(data.failed_tickers)} tone="text-warn" />
        </div>

        <div className="pt-2 border-t border-edge grid grid-cols-2 gap-x-4 gap-y-1">
          <Row label="Rows in → out" value={`${compact(cleaning.rows_in)} → ${compact(cleaning.rows_out)}`} />
          <Row label="Retention" value={`${cleaning.retention_pct ?? DASH}%`} />
          <Row label="Bad OHLC dropped" value={String(cleaning.dropped_bad_ohlc ?? 0)} />
          <Row label="Ticks flagged" value={String(cleaning.flagged_extreme_return ?? 0)} />
        </div>

        <p className="text-ink-faint leading-relaxed pt-1">{data.note}</p>
      </div>
    </div>
  )
}

function Row({ label, value, tone = 'text-ink' }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-ink-faint">{label}</span>
      <span className={`${tone} truncate`}>{value}</span>
    </div>
  )
}

function ApiMesh() {
  const { data, loading } = useApi(() => api.apiMetrics(), [], 10_000)

  if (loading) return <Loading />
  if (!data) return <Empty message="No API metrics" />

  const live = data.live
  const meets = data.meets_target

  return (
    <div className="h-full flex flex-col overflow-auto">
      <div className="grid grid-cols-3 divide-x divide-edge border-b border-edge">
        <StatTile
          label="p95 Latency"
          value={live.p95_ms != null ? `${live.p95_ms}ms` : DASH}
          sub={`target <${data.target_p95_ms}ms`}
          tone={meets ? 'text-mint' : meets === false ? 'text-danger' : 'text-ink'}
        />
        <StatTile label="p50" value={live.p50_ms != null ? `${live.p50_ms}ms` : DASH} />
        <StatTile
          label="Requests"
          value={compact(live.n_requests)}
          sub={`${(live.error_rate * 100).toFixed(1)}% errors`}
        />
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-base-panel">
            <tr>
              <Th>Endpoint</Th>
              <Th align="right">Reqs</Th>
              <Th align="right">Mean</Th>
              <Th align="right">Max</Th>
            </tr>
          </thead>
          <tbody>
            {data.by_endpoint.map((e) => (
              <tr key={e.endpoint} className="border-b border-edge/40">
                <Cell tone="text-ink-muted">
                  <span className="text-2xs">{e.endpoint}</span>
                </Cell>
                <Cell align="right" tone="text-ink-faint">
                  {e.requests}
                </Cell>
                <Cell align="right" tone={e.mean_ms < 200 ? 'text-mint' : 'text-warn'}>
                  {e.mean_ms}ms
                </Cell>
                <Cell align="right" tone="text-ink-faint">
                  {e.max_ms}ms
                </Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Provenance>
        Measured by middleware on every request, not sampled or estimated.
      </Provenance>
    </div>
  )
}

function Syslog() {
  const [level, setLevel] = useState('ALL')
  const { data, loading } = useApi(() => api.systemLogs(120, level), [level], 8_000)

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex items-center justify-between px-3 py-2 border-b border-edge shrink-0">
        <Tabs options={['ALL', 'ERRORS', 'WARNINGS']} value={level} onChange={setLevel} />
        <span className="text-2xs font-mono text-ink-faint">
          {data?.n_entries ?? 0} entries
        </span>
      </div>
      <div className="flex-1 overflow-auto px-3 py-2 space-y-0.5 font-mono text-2xs min-h-0">
        {loading ? (
          <Loading />
        ) : !data?.logs.length ? (
          <Empty message={`No ${level.toLowerCase()} records`} />
        ) : (
          data.logs.map((log, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-ink-faint shrink-0">{clockTime(log.timestamp)}</span>
              <span className={`shrink-0 ${levelTone(log.level)}`}>
                [{log.level.slice(0, 4)}]
              </span>
              <span className="text-ink-faint shrink-0">{log.source}</span>
              <span className="text-ink-muted break-all">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default function Pipeline() {
  const { data } = useApi(() => api.systemStatus(), [], 15_000)
  const { data: info } = useApi(() => api.systemInfo(), [])

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 lg:grid-cols-4 border border-edge bg-base-panel divide-x divide-edge">
        <StatTile
          label="System Integrity"
          value={data?.system_integrity ?? DASH}
          tone={statusTone(data?.system_integrity ?? '')}
        />
        <StatTile
          label="Pipeline Uptime"
          value={data?.uptime_pct != null ? `${data.uptime_pct}%` : 'NO RUNS'}
          sub={data ? `${data.total_job_runs} runs · ${data.failed_job_runs} failed` : undefined}
          tone={(data?.uptime_pct ?? 0) >= 99 ? 'text-mint' : 'text-warn'}
          bar={data?.uptime_pct != null ? data.uptime_pct / 100 : undefined}
        />
        <StatTile
          label="Scheduled Jobs"
          value={`${data?.active_jobs ?? 0} / ${data?.configured_jobs ?? 0}`}
          sub="running / configured"
        />
        <StatTile
          label="Last Sync"
          value={ago(data?.last_sync_seconds_ago)}
          sub={data?.last_sync_job ?? undefined}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <Panel title="Data Ingestion Engine" badge="REAL COUNTS" className="h-[380px]">
          <IngestionEngine />
        </Panel>
        <Panel
          title="Syslog"
          badge="LIVE STREAM"
          className="h-[380px]"
          actions={<Dot pulse />}
        >
          <Syslog />
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <Panel title="Job Streams" className="h-[300px]">
          <JobStreams />
        </Panel>
        <Panel title="API Mesh Health" className="h-[300px]">
          <ApiMesh />
        </Panel>
      </div>

      {info && (
        <div className="border border-edge bg-base-panel px-3 py-2 flex items-center gap-4 flex-wrap text-2xs font-mono text-ink-faint">
          <span className="flex items-center gap-1.5">
            <Dot tone="mint" /> Python {info.python}
          </span>
          <span>SOURCE: {info.data_source.toUpperCase()}</span>
          <span>BENCHMARK: {info.benchmark}</span>
          <span>
            COSTS: {info.costs.commission_bps + info.costs.slippage_bps}bps/side
          </span>
          <span>
            LIMITS: DD {(info.risk_limits.max_drawdown_limit * 100).toFixed(0)}% · POS{' '}
            {(info.risk_limits.max_position_weight * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  )
}
