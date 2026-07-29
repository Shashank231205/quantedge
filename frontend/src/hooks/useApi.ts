/**
 * Data-fetching hooks.
 *
 * `useApi` polls; `useWebSocket` receives pushed events. Together they make
 * the screens live without inventing data between updates — the underlying
 * cadence is daily bars plus on-demand backtest runs, and the UI reflects
 * exactly that.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

export interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => void
  lastUpdated: Date | null
}

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  pollMs = 0,
): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  // Keeps the latest fetcher without making it a dependency of the effect.
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const load = useCallback(async (showSpinner: boolean) => {
    if (showSpinner) setLoading(true)
    try {
      const result = await fetcherRef.current()
      setData(result)
      setError(null)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    void load(true)

    if (pollMs > 0) {
      const id = setInterval(() => {
        // Background refreshes must not flash a spinner over live content.
        if (!cancelled && document.visibilityState === 'visible') void load(false)
      }, pollMs)
      return () => {
        cancelled = true
        clearInterval(id)
      }
    }
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, pollMs, load])

  return { data, loading, error, refetch: () => void load(false), lastUpdated }
}

export interface WsEvent {
  type: string
  data: Record<string, any>
  timestamp: string
}

export function useWebSocket(path = '/ws/events') {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let retry: ReturnType<typeof setTimeout>
    let closed = false

    const connect = () => {
      // In development the Vite proxy fronts the API, so same-origin is right.
      // Deployed, the API lives on a different host and VITE_API_URL points at
      // it — deriving the socket URL from window.location would aim the socket
      // at the static host, which serves no WebSocket at all.
      const base = import.meta.env.VITE_API_URL
      const origin = base ? base.replace(/^http/, 'ws') : null
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(
        origin ? `${origin}${path}` : `${proto}//${window.location.host}${path}`,
      )
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        // Reconnect so a backend restart doesn't leave the UI silently stale.
        if (!closed) retry = setTimeout(connect, 5000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (msg) => {
        try {
          setLastEvent(JSON.parse(msg.data))
        } catch {
          /* ignore malformed frames */
        }
      }
    }

    connect()
    return () => {
      closed = true
      clearTimeout(retry)
      wsRef.current?.close()
    }
  }, [path])

  return { connected, lastEvent }
}
