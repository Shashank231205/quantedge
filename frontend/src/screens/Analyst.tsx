/**
 * Analyst Mode (screen 6).
 *
 * Every score shown here is computed by the backend's rubric, not by the
 * language model — the model only writes the reasons and gaps around a verdict
 * that was already decided. The screen states which provider wrote the prose
 * so a reader can weigh it, and citations expand to the figures behind each
 * claim so nothing has to be taken on trust.
 */

import { useCallback, useEffect, useState } from 'react'

import {
  Empty,
  ErrorState,
  Label,
  Loading,
  Panel,
  Pill,
  Provenance,
} from '../components/Primitives'
import { api, type AnalystCitation, type AnalystMetric, type AnalystReport, type AnalystStatus } from '../lib/api'

const UNIVERSE_OPTIONS = [10, 25, 50, 100, 567]

/** Bands map to the same tones the rest of the app uses for good/bad. */
const BAND_TONE: Record<string, string> = {
  STRONG: 'mint',
  ADEQUATE: 'info',
  WEAK: 'warn',
  FAILING: 'danger',
}

const VERDICT_TONE: Record<string, string> = {
  VALIDATED: 'mint',
  PROMISING: 'info',
  UNPROVEN: 'warn',
  WEAK: 'warn',
  'NOT SUPPORTED': 'danger',
}

function scoreColor(score: number): string {
  if (score >= 80) return 'text-mint'
  if (score >= 60) return 'text-info'
  if (score >= 40) return 'text-warn'
  return 'text-danger'
}

/**
 * An inline citation. Collapsed it is a marker in the prose; expanded it shows
 * the figures the claim rests on, so verifying a number never means leaving
 * the report.
 */
function Citation({ citation, index }: { citation: AnalystCitation; index: number }) {
  const [open, setOpen] = useState(false)

  return (
    <span className="relative inline-block align-baseline">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`Citation ${index}: ${citation.title}`}
        className={`ml-1 inline-flex h-4 min-w-4 items-center justify-center border px-1 font-mono text-2xs transition-colors ${
          open
            ? 'border-mint bg-mint/20 text-mint'
            : 'border-edge text-ink-faint hover:border-mint/50 hover:text-mint'
        }`}
      >
        {index}
      </button>

      {open ? (
        <>
          {/* Click-away layer. Sits under the panel but over the page. */}
          <span
            className="fixed inset-0 z-20 cursor-default"
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          {/* bg-base-raised, not a translucent tint: the panel overlaps body
              copy, and anything see-through here is unreadable. */}
          <span className="absolute left-0 top-6 z-30 block w-[min(30rem,80vw)] border border-mint/40 bg-base-raised p-3 shadow-2xl">
            <span className="mb-1 block font-mono text-2xs uppercase tracking-wide text-mint">
              {citation.title}
            </span>
            <span className="mb-2 block text-xs leading-relaxed text-ink-dim">
              {citation.description}
            </span>
            <span className="block border-t border-edge pt-2">
              {citation.points.map((point) => (
                <span
                  key={point}
                  className="block py-0.5 font-mono text-2xs text-ink"
                >
                  <span className="text-ink-faint">— </span>
                  {point}
                </span>
              ))}
            </span>
            <span className="mt-2 block border-t border-edge pt-2 font-mono text-2xs text-ink-faint">
              SOURCE: {citation.source}
            </span>
          </span>
        </>
      ) : null}
    </span>
  )
}

function MetricCard({ metric }: { metric: AnalystMetric }) {
  const tone = BAND_TONE[metric.band] ?? 'muted'

  return (
    <Panel
      title={metric.label}
      badge={metric.band}
      badgeTone={tone as 'mint' | 'warn' | 'danger' | 'info'}
      actions={
        <span className={`font-mono text-lg tabular-nums ${scoreColor(metric.score)}`}>
          {metric.score}
          <span className="text-2xs text-ink-faint">/100</span>
        </span>
      }
    >
      <div className="space-y-3 p-3">
        <div>
          <Label>Reasons</Label>
          <p className="mt-1 text-xs leading-relaxed text-ink">
            {metric.reasons}
            {metric.citations.map((c, i) => (
              <Citation key={c.title} citation={c} index={i + 1} />
            ))}
          </p>
        </div>

        <div>
          <Label>Gaps</Label>
          <p className="mt-1 text-xs leading-relaxed text-ink-dim">{metric.gaps}</p>
        </div>
      </div>
    </Panel>
  )
}

export default function Analyst() {
  const [report, setReport] = useState<AnalystReport | null>(null)
  const [status, setStatus] = useState<AnalystStatus | null>(null)
  const [universe, setUniverse] = useState(50)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(
    async (opts: { refresh?: boolean; size?: number } = {}) => {
      const size = opts.size ?? universe
      opts.refresh ? setRunning(true) : setLoading(true)
      setError(null)
      try {
        const [r, s] = await Promise.all([
          api.analystReport({ universeSize: size, refresh: opts.refresh }),
          api.analystStatus(),
        ])
        setReport(r)
        setStatus(s)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
        setRunning(false)
      }
    },
    [universe],
  )

  useEffect(() => {
    void load()
    // Intentionally runs once: changing the universe re-runs via its own handler
    // so a selection does not fire a request per keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (loading) return <Loading label="GENERATING ASSESSMENT" />
  if (error) return <ErrorState error={error} onRetry={() => void load()} />
  if (!report) return <Empty message="No assessment available" />

  const verdictTone = VERDICT_TONE[report.verdict] ?? 'muted'

  return (
    <div className="space-y-4">
      {/* Verdict */}
      <Panel
        title="Analyst Assessment"
        badge={report.verdict}
        badgeTone={verdictTone as 'mint' | 'warn' | 'danger' | 'info'}
        actions={
          <div className="flex items-center gap-3">
            <select
              value={universe}
              onChange={(e) => {
                const size = Number(e.target.value)
                setUniverse(size)
                void load({ size, refresh: true })
              }}
              className="border border-edge bg-transparent px-2 py-1 font-mono text-2xs text-ink focus:border-mint focus:outline-none"
            >
              {UNIVERSE_OPTIONS.map((n) => (
                <option key={n} value={n} className="bg-base-raised">
                  TOP {n === 567 ? 'ALL' : n}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => void load({ refresh: true })}
              disabled={running}
              className="border border-edge px-2 py-1 font-mono text-2xs uppercase text-ink-dim transition-colors hover:border-mint hover:text-mint disabled:opacity-40"
            >
              {running ? 'Running…' : 'Re-run'}
            </button>
          </div>
        }
      >
        <div className="flex items-start gap-4 p-4">
          {/* The verdict mark. Deliberately large: it is the one thing a reader
              should take away if they read nothing else. */}
          <div
            className={`flex h-16 w-16 shrink-0 items-center justify-center border text-3xl ${
              verdictTone === 'mint'
                ? 'border-mint/40 bg-mint/10 text-mint'
                : verdictTone === 'danger'
                  ? 'border-danger/40 bg-danger/10 text-danger'
                  : verdictTone === 'info'
                    ? 'border-info/40 bg-info/10 text-info'
                    : 'border-warn/40 bg-warn/10 text-warn'
            }`}
            aria-label={`Verdict: ${report.verdict}`}
          >
            ☝
          </div>

          <div className="min-w-0 flex-1">
            <div className="mb-2 flex items-baseline gap-3">
              <span className={`font-mono text-3xl tabular-nums ${scoreColor(report.overall_score)}`}>
                {report.overall_score}
                <span className="text-xs text-ink-faint">/100</span>
              </span>
              <span className="truncate font-mono text-2xs text-ink-faint">
                {report.run_name}
              </span>
            </div>
            <p className="text-sm leading-relaxed text-ink">{report.summary}</p>
          </div>
        </div>

        <Provenance>
          {report.is_template
            ? 'PROSE: rule-based templates — no language model was reachable. '
            : `PROSE: ${report.provider} (${report.model}). `}
          SCORES: computed by the platform rubric and identical across providers —
          the model explains the assessment, it does not make it.
          {report.cached ? ' Served from cache; use Re-run to regenerate.' : ''}
        </Provenance>
      </Panel>

      {report.notes.length > 0 ? (
        <Panel title="Generation Notes" badge="DEGRADED" badgeTone="warn">
          <div className="space-y-1 p-3">
            {report.notes.map((n) => (
              <p key={n} className="font-mono text-2xs text-warn">
                {n}
              </p>
            ))}
          </div>
        </Panel>
      ) : null}

      {/* Per-metric detail */}
      <div className="grid gap-4 lg:grid-cols-2">
        {report.metrics.map((m) => (
          <MetricCard key={m.key} metric={m} />
        ))}
      </div>

      {status ? (
        <Panel title="Provider Chain">
          <div className="flex flex-wrap items-center gap-2 p-3">
            {status.configured_chain.map((name) => {
              const isAvailable = status.available.includes(name)
              const isActive = name === status.active_provider
              return (
                <Pill
                  key={name}
                  text={name}
                  tone={isActive ? 'mint' : isAvailable ? 'muted' : 'danger'}
                />
              )
            })}
          </div>
          <Provenance>
            Providers are tried in order; the first reachable one writes the
            report. The template link never fails, so an instance with no API
            keys still produces a full assessment rather than an error.
          </Provenance>
        </Panel>
      ) : null}
    </div>
  )
}
