import Textarea from '@/components/ui/Textarea'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { useCallback, useEffect, useRef, useState } from 'react'
import { multiKBQueryStream, type QueryMode, type Reference } from '@/api/lightrag'
import type { MultiKBQueryRequest } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import QuerySettings from '@/components/retrieval/QuerySettings'
import { ChatMessage, MessageWithError } from '@/components/retrieval/ChatMessage'
import { EraserIcon, SendIcon, CopyIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { copyToClipboard } from '@/utils/clipboard'
import { listKBs, type KBMeta } from '@/api/kb'

const generateUniqueId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `id-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

export default function GlobalSearchView() {
  const { t } = useTranslation()
  const currentTab = useSettingsStore.use.currentTab()
  const isSearchTabActive = currentTab === 'search'

  const [messages, setMessages] = useState<MessageWithError[]>(() => {
    try {
      const history = useSettingsStore.getState().globalSearchMessages || []
      return history.map((msg, index) => {
        const m = msg as MessageWithError
        return {
          ...msg,
          id: m.id || `hist-${Date.now()}-${index}`,
          mermaidRendered: m.mermaidRendered ?? true,
          latexRendered: m.latexRendered ?? true,
        }
      })
    } catch {
      return []
    }
  })
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const [availableKBs, setAvailableKBs] = useState<KBMeta[]>([])
  const [selectedKbIds, setSelectedKbIds] = useState<string[]>([])
  const [kbError, setKbError] = useState('')
  // KB names for each assistant message id → source attribution footer
  const [messageSources, setMessageSources] = useState<Record<string, string[]>>(
    () => useSettingsStore.getState().globalSearchMessageSources || {}
  )

  const [inputError, setInputError] = useState('')
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const responseTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const serverResponseTimeRef = useRef<number | null>(null)
  const serverTokenUsageRef = useRef<{ inputTokens: number; outputTokens: number } | null>(null)
  const queryStartTimeRef = useRef<number | null>(null)

  const hasMultipleLines = inputValue.includes('\n')

  useEffect(() => {
    listKBs()
      .then((kbs) => {
        setAvailableKBs(kbs)
        setSelectedKbIds((prev) => (kbs.length > 0 && prev.length === 0 ? kbs.map((k) => k.id) : prev))
      })
      .catch((err) => {
        setKbError(errorMessage(err))
      })
  }, [])

  const toggleKbSelection = useCallback((kbId: string) => {
    setSelectedKbIds((prev) =>
      prev.includes(kbId) ? prev.filter((id) => id !== kbId) : [...prev, kbId]
    )
  }, [])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [])

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setInputValue(e.target.value)
      if (inputError) setInputError('')
    },
    [inputError]
  )

  const parseModePrefix = useCallback((value: string) => {
    const allowedModes: QueryMode[] = ['naive', 'local', 'global', 'hybrid', 'mix', 'bypass']
    const prefixMatch = value.match(/^\/(\w+)\s+([\s\S]+)/)
    if (/^\/\S+/.test(value) && !prefixMatch) {
      return { error: t('retrievePanel.retrieval.queryModePrefixInvalid') }
    }
    if (prefixMatch) {
      const mode = prefixMatch[1] as QueryMode
      if (!allowedModes.includes(mode)) {
        return {
          error: t('retrievePanel.retrieval.queryModeError', {
            modes: 'naive, local, global, hybrid, mix, bypass'
          })
        }
      }
      return { modeOverride: mode, query: prefixMatch[2] }
    }
    return { query: value }
  }, [t])

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!inputValue.trim() || isLoading) return

      const parsed = parseModePrefix(inputValue)
      if (parsed.error) {
        setInputError(parsed.error)
        return
      }

      if (selectedKbIds.length === 0) {
        setKbError(t('globalSearch.noKBSelected'))
        return
      }

      setInputError('')
      setKbError('')

      const userMessage: MessageWithError = {
        id: generateUniqueId(),
        content: inputValue,
        role: 'user'
      }
      const assistantMessage: MessageWithError = {
        id: generateUniqueId(),
        content: '',
        role: 'assistant',
        mermaidRendered: false,
        latexRendered: false,
        thinkingTime: null,
        thinkingContent: undefined,
        displayContent: undefined,
        isThinking: false
      }

      // Capture KB names for source attribution footer
      const kbNames = selectedKbIds
        .map((id) => availableKBs.find((kb) => kb.id === id)?.name ?? id)

      const prevMessages = [...messages]
      const controller = new AbortController()
      abortControllerRef.current = controller

      setMessages([...prevMessages, userMessage, assistantMessage])
      const newSources = { ...useSettingsStore.getState().globalSearchMessageSources, [assistantMessage.id]: kbNames }
      setMessageSources(newSources)
      useSettingsStore.getState().setGlobalSearchMessageSources(newSources)
      useSettingsStore.getState().setGlobalSearchMessages([...prevMessages, userMessage, assistantMessage])
      setInputValue('')
      setIsLoading(true)

      // Start response timer
      serverResponseTimeRef.current = null
      serverTokenUsageRef.current = null
      queryStartTimeRef.current = Date.now()
      assistantMessage.responseTime = 0
      if (responseTimerRef.current) clearInterval(responseTimerRef.current)
      responseTimerRef.current = setInterval(() => {
        const elapsed = (Date.now() - (queryStartTimeRef.current ?? Date.now())) / 1000
        const rounded = parseFloat(elapsed.toFixed(1))
        assistantMessage.responseTime = rounded
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last && last.id === assistantMessage.id) Object.assign(last, { responseTime: rounded })
          return next
        })
      }, 200)

      setTimeout(() => scrollToBottom(), 0)

      const triggerUpdate = () => {
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last && last.id === assistantMessage.id) {
            Object.assign(last, { ...assistantMessage })
          }
          return next
        })
      }

      const state = useSettingsStore.getState()
      const queryRequest: MultiKBQueryRequest = {
        ...state.querySettings,
        query: parsed.query!,
        include_references: true,
        include_chunk_content: false,
        include_progress: true,
        kbs: selectedKbIds,
        stream: true,
        ...(parsed.modeOverride ? { mode: parsed.modeOverride } : {})
      }

      try {
        await multiKBQueryStream(
          queryRequest,
          (chunk: string) => {
            assistantMessage.content += chunk
            assistantMessage.displayContent = assistantMessage.content
            triggerUpdate()
            scrollToBottom()
          },
          (err: string) => {
            const errorText = `${t('retrievePanel.retrieval.error')}\n${err}`
            assistantMessage.content = assistantMessage.content
              ? `${assistantMessage.content}\n\n${errorText}`
              : errorText
            assistantMessage.displayContent = assistantMessage.content
            assistantMessage.isError = true
            triggerUpdate()
          },
          controller.signal,
          (seconds: number) => {
            serverResponseTimeRef.current = seconds
          },
          undefined,
          (refs: Reference[]) => {
            assistantMessage.references = refs
            triggerUpdate()
          },
          (inputTokens: number, outputTokens: number) => {
            serverTokenUsageRef.current = { inputTokens, outputTokens }
          }
        )
      } catch (err) {
        if (!controller.signal.aborted) {
          const errorText = `${t('retrievePanel.retrieval.error')}\n${errorMessage(err)}`
          assistantMessage.content = errorText
          assistantMessage.displayContent = errorText
          assistantMessage.isError = true
          triggerUpdate()
        }
      } finally {
        if (responseTimerRef.current) {
          clearInterval(responseTimerRef.current)
          responseTimerRef.current = null
        }
        const authTime = serverResponseTimeRef.current
        if (authTime !== null) {
          assistantMessage.responseTime = authTime
        } else if (queryStartTimeRef.current) {
          assistantMessage.responseTime = parseFloat(
            ((Date.now() - queryStartTimeRef.current) / 1000).toFixed(1)
          )
        }
        serverResponseTimeRef.current = null
        const tokenUsage = serverTokenUsageRef.current
        if (tokenUsage) {
          assistantMessage.inputTokens = tokenUsage.inputTokens
          assistantMessage.outputTokens = tokenUsage.outputTokens
        }
        serverTokenUsageRef.current = null
        queryStartTimeRef.current = null
        triggerUpdate()
        // Persist finalized messages to store
        setMessages((prev) => {
          useSettingsStore.getState().setGlobalSearchMessages(prev)
          return prev
        })
        if (abortControllerRef.current === controller) {
          setIsLoading(false)
          abortControllerRef.current = null
        }
      }
    },
    [inputValue, isLoading, messages, parseModePrefix, selectedKbIds, availableKBs, scrollToBottom, t]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault()
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0
        const newValue = inputValue.slice(0, start) + '\n' + inputValue.slice(end)
        setInputValue(newValue)
        setTimeout(() => {
          if (target.setSelectionRange) {
            target.setSelectionRange(start + 1, start + 1)
          }
        }, 0)
      } else if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit(e as any)
      }
    },
    [inputValue, handleSubmit]
  )

  const clearMessages = useCallback(() => {
    setMessages([])
    setMessageSources({})
    useSettingsStore.getState().setGlobalSearchMessages([])
    useSettingsStore.getState().setGlobalSearchMessageSources({})
  }, [])

  useEffect(() => {
    return () => {
      const controller = abortControllerRef.current
      abortControllerRef.current = null
      controller?.abort()
      if (responseTimerRef.current) {
        clearInterval(responseTimerRef.current)
        responseTimerRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const handleCopyMessage = useCallback(
    async (message: MessageWithError) => {
      const contentToCopy =
        message.role === 'user'
          ? message.content || ''
          : message.displayContent !== undefined
            ? message.displayContent
            : message.content || ''

      if (!contentToCopy.trim()) {
        toast.error(t('retrievePanel.chatMessage.copyEmpty'))
        return
      }

      try {
        const result = await copyToClipboard(contentToCopy)
        if (result.success) {
          toast.success(t('retrievePanel.chatMessage.copySuccess'))
        } else {
          toast.error(t('retrievePanel.chatMessage.copyFailed'))
        }
      } catch {
        toast.error(t('retrievePanel.chatMessage.copyError'))
      }
    },
    [t]
  )

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <div className="flex grow flex-col gap-4">
        <div className="relative grow overflow-hidden">
          <div className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2">
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {messages.length === 0 ? (
                <div className="text-muted-foreground flex h-full flex-col items-center justify-center gap-2 text-center">
                  <p className="text-lg">{t('globalSearch.startPrompt')}</p>
                  {availableKBs.length === 0 && (
                    <p className="text-sm">{t('globalSearch.createKBHint')}</p>
                  )}
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} items-end gap-2`}
                  >
                    {message.role === 'user' && (
                      <Button
                        onClick={() => handleCopyMessage(message)}
                        className="mb-2 size-6 rounded-md opacity-60 transition-opacity hover:opacity-100 shrink-0"
                        tooltip={t('retrievePanel.chatMessage.copyTooltip')}
                        variant="ghost"
                        size="icon"
                      >
                        <CopyIcon className="size-4" />
                      </Button>
                    )}
                    {message.role === 'user' ? (
                      <ChatMessage message={message} isTabActive={isSearchTabActive} />
                    ) : (
                      <div className="flex flex-col min-w-0 flex-1">
                        <ChatMessage message={message} isTabActive={isSearchTabActive} />
                        {messageSources[message.id] && (
                          <p className="mt-1 text-xs text-muted-foreground px-1">
                            {t('globalSearch.sources')}: {messageSources[message.id].join(', ')}
                          </p>
                        )}
                      </div>
                    )}
                    {message.role === 'assistant' && (
                      <Button
                        onClick={() => handleCopyMessage(message)}
                        className="mb-2 size-6 rounded-md opacity-60 transition-opacity hover:opacity-100 shrink-0"
                        tooltip={t('retrievePanel.chatMessage.copyTooltip')}
                        variant="ghost"
                        size="icon"
                      >
                        <CopyIcon className="size-4" />
                      </Button>
                    )}
                  </div>
                ))
              )}
              <div ref={messagesEndRef} className="pb-1" />
            </div>
          </div>
        </div>

        {inputError && (
          <div className="text-xs text-red-500">{inputError}</div>
        )}

        <form onSubmit={handleSubmit} className="flex shrink-0 items-center gap-2" role="search">
          <input type="submit" style={{ display: 'none' }} tabIndex={-1} />
          <Button
            type="button"
            variant="outline"
            onClick={clearMessages}
            disabled={isLoading}
            size="sm"
          >
            <EraserIcon />
            {t('retrievePanel.retrieval.clear')}
          </Button>
          <div className="flex-1 relative">
            <label htmlFor="global-search-input" className="sr-only">
              {t('globalSearch.placeholder')}
            </label>
            {hasMultipleLines ? (
              <Textarea
                ref={inputRef as React.RefObject<HTMLTextAreaElement>}
                id="global-search-input"
                className="w-full min-h-[40px] max-h-[120px] overflow-y-auto"
                value={inputValue}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                placeholder={t('globalSearch.placeholder')}
                disabled={isLoading}
                rows={1}
                style={{ resize: 'none', minHeight: '40px', maxHeight: '120px' }}
                onInput={(e: React.FormEvent<HTMLTextAreaElement>) => {
                  const target = e.target as HTMLTextAreaElement
                  target.style.height = 'auto'
                  target.style.height = Math.min(target.scrollHeight, 120) + 'px'
                }}
              />
            ) : (
              <Input
                ref={inputRef as React.RefObject<HTMLInputElement>}
                id="global-search-input"
                className="w-full"
                value={inputValue}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                placeholder={t('globalSearch.placeholder')}
                disabled={isLoading}
              />
            )}
          </div>
          <Button type="submit" variant="default" size="sm" disabled={isLoading}>
            <SendIcon />
            {t('retrievePanel.retrieval.send')}
          </Button>
        </form>
      </div>
      <QuerySettings
        availableKBs={availableKBs}
        selectedKbIds={selectedKbIds}
        onToggleKb={toggleKbSelection}
        kbError={kbError}
        disabled={isLoading}
      />
    </div>
  )
}
