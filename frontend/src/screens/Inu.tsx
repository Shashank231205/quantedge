/**
 * INU AI (screen 7).
 *
 * A chat over the platform's own data. Every number in an answer came from a
 * tool call against the same endpoints the other screens read, and each reply
 * shows which free model wrote it and which tools it consulted — the same
 * provenance the rest of the app exposes, applied to prose.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { Empty, Panel, Pill } from '../components/Primitives'
import {
  api,
  type InuConversation,
  type InuMessage,
  type InuStatus,
} from '../lib/api'

/** Openers that show what INU can actually do, rather than generic prompts. */
const SUGGESTIONS = [
  'Is this strategy actually any good?',
  'Why is the deflated Sharpe so low?',
  'What does the Calmar ratio measure?',
  'Which names rank highest right now?',
  'How do you avoid lookahead bias?',
]

function relativeTime(iso: string | null): string {
  if (!iso) return ''
  const secs = (Date.now() - new Date(iso).getTime()) / 1000
  if (secs < 60) return 'just now'
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`
  return `${Math.floor(secs / 86400)}d ago`
}

/**
 * Minimal markdown: paragraphs, bullets and `code`. Deliberately not a full
 * renderer — the prompt asks for prose, and anything richer would invite the
 * model to answer in headers and tables.
 */
function Prose({ text }: { text: string }) {
  const blocks = text.split(/\n\n+/)
  return (
    <>
      {blocks.map((block, bi) => {
        const lines = block.split('\n')
        const isList = lines.every((l) => /^\s*[-*•]\s+/.test(l))
        if (isList) {
          return (
            <ul key={bi} className="my-1.5 space-y-1 pl-4">
              {lines.map((l, li) => (
                <li key={li} className="list-disc text-ink">
                  {inline(l.replace(/^\s*[-*•]\s+/, ''))}
                </li>
              ))}
            </ul>
          )
        }
        return (
          <p key={bi} className="mb-2 last:mb-0 leading-relaxed text-ink">
            {inline(block)}
          </p>
        )
      })}
    </>
  )
}

function inline(text: string) {
  // Split on `code` and **bold** together so both render in one pass.
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} className="bg-base-raised px-1 font-mono text-2xs text-mint">
          {part.slice(1, -1)}
        </code>
      )
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={i} className="text-ink">
          {part.slice(2, -2)}
        </strong>
      )
    }
    return part
  })
}

function Bubble({ msg }: { msg: InuMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`border px-3 py-2 text-xs ${
            isUser
              ? 'border-mint/30 bg-mint/10 text-ink'
              : 'border-edge bg-base-panel text-ink'
          }`}
        >
          {msg.attachment_name ? (
            <div className="mb-1 font-mono text-2xs text-ink-faint">
              📎 {msg.attachment_name}
            </div>
          ) : null}
          <Prose text={msg.content} />
        </div>

        {/* Provenance. A reader should always be able to see which free model
            answered and what platform data it actually looked at. */}
        {!isUser && msg.model ? (
          <div className="mt-1 flex flex-wrap items-center gap-1.5 font-mono text-2xs text-ink-faint">
            <span>{msg.model}</span>
            {msg.latency_ms ? <span>· {msg.latency_ms}ms</span> : null}
            {msg.tools_used.length > 0 ? (
              <span>· read {[...new Set(msg.tools_used)].join(', ')}</span>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function Inu() {
  const [messages, setMessages] = useState<InuMessage[]>([])
  const [threads, setThreads] = useState<InuConversation[]>([])
  const [status, setStatus] = useState<InuStatus | null>(null)
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const refreshThreads = useCallback(async () => {
    try {
      setThreads((await api.inuConversations()).conversations)
    } catch {
      /* the sidebar is not worth failing the screen over */
    }
  }, [])

  useEffect(() => {
    void refreshThreads()
    void api.inuStatus().then(setStatus).catch(() => {})
  }, [refreshThreads])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  const openThread = useCallback(async (id: number) => {
    setError(null)
    try {
      const thread = await api.inuConversation(id)
      setConversationId(id)
      setMessages(thread.messages)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const newThread = useCallback(() => {
    setConversationId(null)
    setMessages([])
    setError(null)
  }, [])

  const send = useCallback(
    async (text: string, file?: File) => {
      if ((!text.trim() && !file) || busy) return
      setError(null)
      setBusy(true)
      setInput('')

      // Show the question immediately. A chat that waits for the server before
      // echoing what you typed feels broken even when it is working.
      const optimistic: InuMessage = {
        id: -Date.now(),
        role: 'user',
        content: text || `[${file?.name}]`,
        model: null,
        provider: null,
        tools_used: [],
        latency_ms: null,
        attachment_name: file?.name ?? null,
        created_at: new Date().toISOString(),
      }
      setMessages((m) => [...m, optimistic])

      try {
        const reply = file
          ? await api.inuUpload(file, text, conversationId)
          : await api.inuChat(text, conversationId)

        setConversationId(reply.conversation_id)
        setMessages((m) => [
          ...m,
          {
            id: Date.now(),
            role: 'assistant',
            content: reply.content,
            model: reply.model,
            provider: reply.provider,
            tools_used: reply.tools_used,
            latency_ms: reply.latency_ms,
            attachment_name: null,
            created_at: new Date().toISOString(),
          },
        ])
        void refreshThreads()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        setMessages((m) => m.filter((x) => x.id !== optimistic.id))
      } finally {
        setBusy(false)
      }
    },
    [busy, conversationId, refreshThreads],
  )

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      {/* Conversation */}
      <div className="flex min-w-0 flex-1 flex-col">
        <Panel
          title="INU AI"
          badge={status ? `${status.models.length} FREE MODELS` : undefined}
          badgeTone="mint"
          className="flex min-h-0 flex-1 flex-col"
          bodyClass="flex min-h-0 flex-1 flex-col"
        >
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-4">
                <div className="text-center">
                  <div className="mb-1 font-mono text-sm text-mint">INU</div>
                  <p className="max-w-md text-xs leading-relaxed text-ink-dim">
                    Ask about the strategy, the factors, the risk state, or the
                    methodology. Answers are read from the platform's own data,
                    not from model memory.
                  </p>
                </div>
                <div className="flex max-w-lg flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => void send(s)}
                      className="border border-edge px-2 py-1 text-2xs text-ink-dim transition-colors hover:border-mint hover:text-mint"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m) => <Bubble key={m.id} msg={m} />)
            )}

            {busy ? (
              <div className="flex items-center gap-2 font-mono text-2xs text-ink-faint">
                <span className="inline-block h-1.5 w-1.5 animate-pulse bg-mint" />
                thinking…
              </div>
            ) : null}

            {error ? (
              <div className="border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
                {error}
              </div>
            ) : null}

            <div ref={bottomRef} />
          </div>

          {/* Composer */}
          <div className="border-t border-edge p-3">
            <div className="flex items-end gap-2">
              <input
                ref={fileRef}
                type="file"
                accept="image/*,.txt,.md,.csv,.json"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) void send(input, f)
                  e.target.value = ''
                }}
              />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={busy}
                aria-label="Attach an image or document"
                className="border border-edge px-2 py-1.5 font-mono text-2xs text-ink-dim transition-colors hover:border-mint hover:text-mint disabled:opacity-40"
              >
                📎
              </button>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  // Enter sends; Shift+Enter is a newline. Matches what people
                  // expect from every other chat they have used.
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void send(input)
                  }
                }}
                rows={1}
                placeholder="Ask about the strategy, a metric, or anything else…"
                disabled={busy}
                className="max-h-32 min-h-[2.25rem] flex-1 resize-y border border-edge bg-transparent px-2 py-1.5 text-xs text-ink placeholder:text-ink-faint focus:border-mint focus:outline-none disabled:opacity-40"
              />
              <button
                type="button"
                onClick={() => void send(input)}
                disabled={busy || !input.trim()}
                className="border border-mint/40 bg-mint/10 px-3 py-1.5 font-mono text-2xs uppercase text-mint transition-colors hover:bg-mint/20 disabled:opacity-30"
              >
                Send
              </button>
            </div>
          </div>
        </Panel>
      </div>

      {/* History */}
      <div className="hidden w-64 shrink-0 flex-col lg:flex">
        <Panel
          title="Chats"
          className="flex min-h-0 flex-1 flex-col"
          bodyClass="flex min-h-0 flex-1 flex-col"
          actions={
            <button
              type="button"
              onClick={newThread}
              className="border border-edge px-2 py-0.5 font-mono text-2xs uppercase text-ink-dim transition-colors hover:border-mint hover:text-mint"
            >
              + New
            </button>
          }
        >
          <div className="min-h-0 flex-1 overflow-y-auto">
            {threads.length === 0 ? (
              <div className="p-3">
                <Empty message="No chats yet" />
              </div>
            ) : (
              threads.map((t) => (
                <div
                  key={t.id}
                  className={`group flex items-start gap-1 border-b border-edge/50 px-3 py-2 transition-colors ${
                    t.id === conversationId ? 'bg-mint/5' : 'hover:bg-base-raised'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => void openThread(t.id)}
                    className="min-w-0 flex-1 text-left"
                  >
                    <div
                      className={`truncate text-2xs ${
                        t.id === conversationId ? 'text-mint' : 'text-ink'
                      }`}
                    >
                      {t.title}
                    </div>
                    <div className="mt-0.5 font-mono text-2xs text-ink-faint">
                      {t.n_messages} msg · {relativeTime(t.updated_at)}
                    </div>
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete ${t.title}`}
                    onClick={async () => {
                      await api.inuDelete(t.id)
                      if (t.id === conversationId) newThread()
                      void refreshThreads()
                    }}
                    className="opacity-0 transition-opacity group-hover:opacity-100 font-mono text-2xs text-ink-faint hover:text-danger"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>

          {status ? (
            <div className="border-t border-edge p-3">
              <div className="mb-1.5 font-mono text-2xs uppercase text-ink-faint">
                Capabilities
              </div>
              <div className="flex flex-wrap gap-1">
                <Pill text="web search" tone="mint" />
                <Pill text="images" tone="mint" />
                <Pill text="documents" tone="mint" />
              </div>
              <p className="mt-2 font-mono text-2xs leading-relaxed text-ink-faint">
                {status.models.length} free open-weight models. Numbers come from{' '}
                {status.tools.length} platform tools, not model memory.
              </p>
            </div>
          ) : null}
        </Panel>
      </div>
    </div>
  )
}
