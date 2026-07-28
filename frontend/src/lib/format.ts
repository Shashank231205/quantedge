/** Display formatting. Null-safe throughout: a missing value renders as an
 *  em-dash rather than "NaN" or a fabricated zero. */

export const DASH = '—'

export function pct(value: number | null | undefined, digits = 2, signed = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  const out = `${(value * 100).toFixed(digits)}%`
  return signed && value > 0 ? `+${out}` : out
}

export function num(value: number | null | undefined, digits = 2, signed = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  const out = value.toFixed(digits)
  return signed && value > 0 ? `+${out}` : out
}

export function money(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  const sign = value < 0 ? '-' : ''
  return `${sign}$${Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`
}

export function compact(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return value.toLocaleString('en-US')
}

export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = value
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`
}

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  if (value >= 60_000) return `${(value / 60_000).toFixed(1)}m`
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)}s`
  return `${value.toFixed(1)}ms`
}

export function ago(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return DASH
  if (seconds < 60) return `${seconds.toFixed(1)}s AGO`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m AGO`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h AGO`
  return `${Math.floor(seconds / 86400)}d AGO`
}

export function shortDate(value: string | null | undefined): string {
  if (!value) return DASH
  const d = new Date(value)
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: '2-digit' })
}

export function clockTime(iso: string | null | undefined): string {
  if (!iso) return DASH
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleTimeString('en-US', { hour12: false }) + '.' +
    String(d.getMilliseconds()).padStart(3, '0')
}

/** Positive-is-good colouring used across metric tiles. */
export function toneFor(value: number | null | undefined, invert = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-ink-muted'
  const good = invert ? value < 0 : value > 0
  if (Math.abs(value) < 1e-12) return 'text-ink-muted'
  return good ? 'text-mint' : 'text-danger'
}

export function levelTone(level: string): string {
  switch (level.toUpperCase()) {
    case 'ERROR':
    case 'CRITICAL':
      return 'text-danger'
    case 'WARNING':
      return 'text-warn'
    case 'DEBUG':
      return 'text-ink-faint'
    default:
      return 'text-info'
  }
}

export function statusTone(status: string): string {
  const s = status.toUpperCase()
  if (['SUCCESS', 'OPERATIONAL', 'PRODUCTION_READY', 'ACTIVE', 'OK', 'HEALTHY', 'ONLINE'].includes(s))
    return 'text-mint'
  if (['FAILED', 'BREACHED', 'ERROR', 'HIGH'].includes(s)) return 'text-danger'
  if (['RUNNING', 'RETRYING', 'DEGRADED', 'MEDIUM', 'WARNING'].includes(s)) return 'text-warn'
  return 'text-ink-muted'
}
