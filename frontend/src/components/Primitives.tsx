/**
 * Shared UI primitives in the terminal style from the Figma:
 * dark panels, hairline borders, uppercase monospace labels, mint accents.
 */

import type { ReactNode } from 'react'
import { DASH } from '../lib/format'

export function Panel({
  title,
  badge,
  badgeTone = 'mint',
  actions,
  children,
  className = '',
  bodyClass = '',
}: {
  title?: ReactNode
  badge?: string
  badgeTone?: 'mint' | 'warn' | 'danger' | 'info'
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClass?: string
}) {
  const toneMap = {
    mint: 'text-mint border-mint/30 bg-mint/10',
    warn: 'text-warn border-warn/30 bg-warn/10',
    danger: 'text-danger border-danger/30 bg-danger/10',
    info: 'text-info border-info/30 bg-info/10',
  }
  return (
    <section className={`border border-edge bg-base-panel flex flex-col min-h-0 ${className}`}>
      {(title || badge || actions) && (
        <header className="flex items-center justify-between px-3 py-2 border-b border-edge shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            {typeof title === 'string' ? (
              <h2 className="text-sm font-medium text-ink truncate">{title}</h2>
            ) : (
              title
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {actions}
            {badge && (
              <span
                className={`text-2xs font-mono uppercase tracking-wider px-1.5 py-0.5 border ${toneMap[badgeTone]}`}
              >
                {badge}
              </span>
            )}
          </div>
        </header>
      )}
      <div className={`flex-1 min-h-0 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function StatTile({
  label,
  value,
  sub,
  tone = 'text-ink',
  bar,
  large = false,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: string
  bar?: number
  large?: boolean
}) {
  return (
    <div className="px-3 py-2.5 flex flex-col gap-1 min-w-0">
      <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint truncate">
        {label}
      </span>
      <span className={`font-mono tabular-nums ${large ? 'text-3xl' : 'text-xl'} ${tone}`}>
        {value}
      </span>
      {sub && <span className="text-2xs text-ink-muted truncate">{sub}</span>}
      {bar !== undefined && (
        <div className="h-0.5 bg-edge mt-1">
          <div
            className="h-full bg-mint transition-all duration-500"
            style={{ width: `${Math.min(100, Math.max(0, bar * 100))}%` }}
          />
        </div>
      )}
    </div>
  )
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint">{children}</span>
  )
}

export function Dot({ tone = 'mint', pulse = false }: { tone?: string; pulse?: boolean }) {
  const colour =
    tone === 'mint' ? 'bg-mint' :
    tone === 'danger' ? 'bg-danger' :
    tone === 'warn' ? 'bg-warn' : 'bg-ink-faint'
  return (
    <span className="relative inline-flex h-1.5 w-1.5 shrink-0">
      {pulse && (
        <span className={`absolute inline-flex h-full w-full rounded-full ${colour} opacity-60 animate-ping`} />
      )}
      <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${colour}`} />
    </span>
  )
}

/** Explicit empty state. Screens must never fall back to placeholder numbers. */
export function Empty({ message, hint }: { message: string; hint?: string }) {
  return (
    <div className="h-full min-h-[120px] flex flex-col items-center justify-center gap-1.5 p-6 text-center">
      <span className="text-xs text-ink-muted">{message}</span>
      {hint && <span className="text-2xs font-mono text-ink-faint max-w-xs">{hint}</span>}
    </div>
  )
}

export function Loading({ label = 'LOADING' }: { label?: string }) {
  return (
    <div className="h-full min-h-[120px] flex items-center justify-center gap-2">
      <Dot pulse />
      <span className="text-2xs font-mono uppercase tracking-wider text-ink-faint">{label}</span>
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="h-full min-h-[120px] flex flex-col items-center justify-center gap-2 p-6 text-center">
      <span className="text-xs text-danger">{error}</span>
      <span className="text-2xs font-mono text-ink-faint">
        Is the API running? <code>make serve</code>
      </span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 text-2xs font-mono uppercase tracking-wider px-2 py-1 border border-edge hover:border-mint hover:text-mint transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  )
}

export function Tabs({
  options,
  value,
  onChange,
}: {
  options: string[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex border border-edge">
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`px-2 py-0.5 text-2xs font-mono uppercase tracking-wider transition-colors ${
            value === opt ? 'bg-mint/15 text-mint' : 'text-ink-faint hover:text-ink-muted'
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  )
}

export function Cell({
  children,
  tone = 'text-ink',
  mono = true,
  align = 'left',
}: {
  children: ReactNode
  tone?: string
  mono?: boolean
  align?: 'left' | 'right' | 'center'
}) {
  const alignment =
    align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : 'text-left'
  return (
    <td className={`px-2.5 py-1.5 ${mono ? 'font-mono tabular-nums' : ''} ${tone} ${alignment}`}>
      {children ?? DASH}
    </td>
  )
}

export function Th({
  children,
  align = 'left',
  onClick,
  active,
}: {
  children: ReactNode
  align?: 'left' | 'right' | 'center'
  onClick?: () => void
  active?: boolean
}) {
  const alignment =
    align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : 'text-left'
  return (
    <th
      onClick={onClick}
      className={`px-2.5 py-1.5 text-2xs font-mono uppercase tracking-wider font-normal border-b border-edge whitespace-nowrap ${alignment} ${
        onClick ? 'cursor-pointer hover:text-ink' : ''
      } ${active ? 'text-mint' : 'text-ink-faint'}`}
    >
      {children}
      {active && <span className="ml-1">▼</span>}
    </th>
  )
}

export function Pill({ text, tone = 'mint' }: { text: string; tone?: string }) {
  const map: Record<string, string> = {
    mint: 'text-mint border-mint/40 bg-mint/10',
    danger: 'text-danger border-danger/40 bg-danger/10',
    warn: 'text-warn border-warn/40 bg-warn/10',
    muted: 'text-ink-faint border-edge bg-transparent',
  }
  return (
    <span
      className={`inline-block text-2xs font-mono uppercase tracking-wide px-1.5 py-0.5 border ${map[tone] ?? map.muted}`}
    >
      {text}
    </span>
  )
}

/** Provenance note. Used wherever a number could be misread as validated. */
export function Provenance({ children }: { children: ReactNode }) {
  return (
    <p className="text-2xs font-mono text-ink-faint leading-relaxed px-3 py-2 border-t border-edge">
      {children}
    </p>
  )
}
