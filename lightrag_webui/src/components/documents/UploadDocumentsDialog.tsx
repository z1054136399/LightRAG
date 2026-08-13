import { useState, useCallback, useEffect } from 'react'
import { FileRejection } from 'react-dropzone'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import FileUploader from '@/components/ui/FileUploader'
import { toast } from 'sonner'
import { supportedFileTypes } from '@/lib/constants'
import {
  deriveUploaderInputs,
  flattenAcceptExtensions,
  formatFileTypesLabel,
  normalizeSupportedFileTypes,
  type FileTypesState
} from '@/lib/fileTypes'
import { errorMessage } from '@/lib/utils'
import { getSupportedFileTypes, getChunkingDefaults, uploadDocument, type ChunkingStrategy, type UploadChunkingOptions, type ChunkingDefaults } from '@/api/lightrag'

import { UploadIcon, CheckIcon, XIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'

interface StrategyOption {
  value: ChunkingStrategy | ''
  labelKey: string
  descKey: string
  supportsOverlap: boolean
}

const CHUNKING_STRATEGIES: StrategyOption[] = [
  {
    value: '',
    labelKey: 'documentPanel.uploadDocuments.chunking.strategyDefault',
    descKey: 'documentPanel.uploadDocuments.chunking.strategyDefaultDesc',
    supportsOverlap: true,
  },
  {
    value: 'fixed_token',
    labelKey: 'documentPanel.uploadDocuments.chunking.strategies.fixedToken',
    descKey: 'documentPanel.uploadDocuments.chunking.strategies.fixedTokenDesc',
    supportsOverlap: true,
  },
  {
    value: 'recursive_character',
    labelKey: 'documentPanel.uploadDocuments.chunking.strategies.recursiveCharacter',
    descKey: 'documentPanel.uploadDocuments.chunking.strategies.recursiveCharacterDesc',
    supportsOverlap: true,
  },
  {
    value: 'semantic_vector',
    labelKey: 'documentPanel.uploadDocuments.chunking.strategies.semanticVector',
    descKey: 'documentPanel.uploadDocuments.chunking.strategies.semanticVectorDesc',
    supportsOverlap: false,
  },
  {
    value: 'paragraph_semantic',
    labelKey: 'documentPanel.uploadDocuments.chunking.strategies.paragraphSemantic',
    descKey: 'documentPanel.uploadDocuments.chunking.strategies.paragraphSemanticDesc',
    supportsOverlap: true,
  },
]

interface UploadDocumentsDialogProps {
  onDocumentsUploaded?: () => Promise<void>
  /**
   * Fired once per batch as soon as the first file is accepted by the server.
   * Lets the parent start its activity probe as early as possible (rather
   * than waiting for the whole sequential batch to finish).
   */
  onUploadBatchAccepted?: () => void
}

export default function UploadDocumentsDialog({
  onDocumentsUploaded,
  onUploadBatchAccepted
}: UploadDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [step, setStep] = useState<1 | 2>(1)
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({})
  const [fileTypes, setFileTypes] = useState<FileTypesState>({ status: 'idle' })
  const [chunkingOptions, setChunkingOptions] = useState<UploadChunkingOptions>({})
  const [selectedStrategy, setSelectedStrategy] = useState<ChunkingStrategy | ''>('')
  const [serverDefaults, setServerDefaults] = useState<ChunkingDefaults | null>(null)
  const [splitByCharacter, setSplitByCharacter] = useState('')
  const [separators, setSeparators] = useState<string[]>([])
  const [separatorInput, setSeparatorInput] = useState('')

  // Fetch the live allowlist + engine capability matrix while the dialog is
  // open. `loading` is entered synchronously in onOpenChange (not here) so
  // the very first open render already has the uploader disabled.
  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    getSupportedFileTypes(controller.signal)
      .then((res) => {
        if (controller.signal.aborted) return
        const data = normalizeSupportedFileTypes(res)
        setFileTypes(data ? { status: 'ready', data } : { status: 'fallback' })
      })
      .catch((err) => {
        if (controller.signal.aborted) return
        // Old backend (404) or transient failure: fall back to the static
        // allowlist and let the server judge hinted filenames.
        console.warn('Failed to fetch supported file types:', errorMessage(err))
        setFileTypes({ status: 'fallback' })
      })
    getChunkingDefaults(controller.signal)
      .then((d) => { if (!controller.signal.aborted) setServerDefaults(d) })
      .catch(() => {})
    return () => controller.abort()
  }, [open])

  const resetDialog = useCallback(() => {
    setStep(1)
    setPendingFiles([])
    setFileErrors({})
    setFileTypes({ status: 'idle' })
    setChunkingOptions({})
    setSelectedStrategy('')
    setServerDefaults(null)
    setSplitByCharacter('')
    setSeparators([])
    setSeparatorInput('')
  }, [])

  const handleRejectedFiles = useCallback(
    (rejectedFiles: FileRejection[]) => {
      // Process rejected files and add them to fileErrors
      rejectedFiles.forEach(({ file, errors }) => {
        // Get the first error message
        let errorMsg = errors[0]?.message || t('documentPanel.uploadDocuments.fileUploader.fileRejected', { name: file.name })

        // Simplify error message for unsupported file types
        if (errorMsg.includes('file-invalid-type')) {
          errorMsg = t('documentPanel.uploadDocuments.fileUploader.unsupportedType')
        }

        // Add error message to fileErrors
        setFileErrors(prev => ({
          ...prev,
          [file.name]: errorMsg
        }))
      })
    },
    [t]
  )

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      let hasSuccessfulUpload = false

      // Show uploading toast
      const toastId = toast.loading(t('documentPanel.uploadDocuments.batch.uploading'))

      try {
        // Track errors locally to ensure we have the final state
        const uploadErrors: Record<string, string> = {}
        let batchProbeTriggered = false

        // Create a collator that supports Chinese sorting
        const collator = new Intl.Collator(['zh-CN', 'en'], {
          sensitivity: 'accent',  // consider basic characters, accents, and case
          numeric: true           // enable numeric sorting, e.g., "File 10" will be after "File 2"
        });
        const sortedFiles = [...filesToUpload].sort((a, b) =>
          collator.compare(a.name, b.name)
        );

        // Upload files in sequence, not parallel
        for (const file of sortedFiles) {
          try {
            const opts = Object.keys(chunkingOptions).length > 0 ? chunkingOptions : undefined
            const result = await uploadDocument(file, () => {}, opts)

            if (result.status !== 'success') {
              uploadErrors[file.name] = result.message
            } else {
              // Mark that we had at least one successful upload
              hasSuccessfulUpload = true
              if (!batchProbeTriggered) {
                batchProbeTriggered = true
                onUploadBatchAccepted?.()
              }
            }
          } catch (err) {
            console.error(`Upload failed for ${file.name}:`, err)

            // Handle HTTP errors, including 400 errors
            let errorMsg = errorMessage(err)
            const duplicateFileMsg = t('documentPanel.uploadDocuments.fileUploader.duplicateFile')

            // If it's an axios error with response data, try to extract more detailed error info
            if (err && typeof err === 'object' && 'response' in err) {
              const axiosError = err as { response?: { status: number, data?: { detail?: string } } }
              const status = axiosError.response?.status
              const detail = axiosError.response?.data?.detail
              if (status === 409) {
                // Server now rejects same-name uploads with HTTP 409 instead of
                // returning a 200 ``status="duplicated"`` payload.  Map the most
                // common cases (existing record / file in INPUT dir) back to the
                // dedicated "duplicate file" UI affordance, and surface other
                // 409 reasons (pipeline busy / scanning) verbatim from the
                // server detail so users can tell why they were rejected.
                if (
                  typeof detail === 'string' &&
                  (/already contains/i.test(detail) || /Status:/i.test(detail))
                ) {
                  errorMsg = duplicateFileMsg
                } else {
                  errorMsg = detail || errorMsg
                }
              } else if (status === 400 || status === 413 || status === 429 || status === 503) {
                // 400 invalid request, 413 body/file too large, 429 pipeline at
                // capacity (MAX_PENDING_DOCUMENTS — the detail carries how many
                // documents are active, how many were requested, the capacity and
                // a retry hint), 503 document storage unavailable. Each detail is
                // written to be shown to a user verbatim.
                errorMsg = detail || errorMsg
              }
            }

            // Record error message in local tracking
            uploadErrors[file.name] = errorMsg
          }
        }

        // Check if any files failed to upload using our local tracking
        const hasErrors = Object.keys(uploadErrors).length > 0

        // Update toast status
        if (hasErrors) {
          toast.error(t('documentPanel.uploadDocuments.batch.error'), { id: toastId })
        } else {
          toast.success(t('documentPanel.uploadDocuments.batch.success'), { id: toastId })
        }

        // Only update if at least one file was uploaded successfully
        if (hasSuccessfulUpload) {
          // Refresh document list
          onDocumentsUploaded?.().catch(err => {
            console.error('Error refreshing documents:', err)
          })
        }
      } catch (err) {
        console.error('Unexpected error during upload:', err)
        toast.error(t('documentPanel.uploadDocuments.generalError', { error: errorMessage(err) }), { id: toastId })
      }
    },
    [t, chunkingOptions, onDocumentsUploaded, onUploadBatchAccepted]
  )

  const handleFinish = useCallback(async () => {
    if (pendingFiles.length === 0) return
    setOpen(false)
    resetDialog()
    await handleDocumentsUpload(pendingFiles)
  }, [pendingFiles, handleDocumentsUpload, resetDialog])

  const handleSelectStrategy = useCallback((value: ChunkingStrategy | '') => {
    setSelectedStrategy(value)
    setSplitByCharacter('')
    setSeparators([])
    setSeparatorInput('')
    setChunkingOptions((o) => {
      const next: UploadChunkingOptions = { ...o, strategy: value || undefined }
      const s = CHUNKING_STRATEGIES.find(s => s.value === value)
      if (s && !s.supportsOverlap) {
        delete next.chunkOverlapTokenSize
      }
      delete next.splitByCharacter
      delete next.separators
      return next
    })
  }, [])

  // \n, \t, \r escape sequences → real characters
  const unescape = (s: string) => s.replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\r/g, '\r')
  const escapeForDisplay = (s: string) =>
    s === '' ? '(空字符串)' : s.replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/\r/g, '\\r')

  const handleAddSeparator = useCallback((raw: string) => {
    const val = unescape(raw)
    setSeparators((prev) => {
      if (prev.includes(val)) return prev
      const next = [...prev, val]
      setChunkingOptions((o) => ({ ...o, separators: next }))
      return next
    })
    setSeparatorInput('')
  }, [])

  const handleRemoveSeparator = useCallback((idx: number) => {
    setSeparators((prev) => {
      const next = prev.filter((_, i) => i !== idx)
      setChunkingOptions((o) => ({ ...o, separators: next.length > 0 ? next : undefined }))
      return next
    })
  }, [])

  const currentStrategy = CHUNKING_STRATEGIES.find(s => s.value === selectedStrategy) ?? CHUNKING_STRATEGIES[0]
  const uploaderInputs = deriveUploaderInputs(fileTypes)

  // Build the live description for the "server default" card
  const serverDefaultDesc = serverDefaults
    ? t('documentPanel.uploadDocuments.chunking.strategyDefaultDescLive', {
        strategy: t(`documentPanel.uploadDocuments.chunking.strategies.${serverDefaults.strategy === 'fixed_token' ? 'fixedToken' : serverDefaults.strategy === 'recursive_character' ? 'recursiveCharacter' : serverDefaults.strategy === 'semantic_vector' ? 'semanticVector' : 'paragraphSemantic'}`),
        size: serverDefaults.chunk_token_size,
        overlap: serverDefaults.chunk_overlap_token_size,
      })
    : t('documentPanel.uploadDocuments.chunking.strategyDefaultDesc')

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (nextOpen) {
          // Enter loading synchronously so the first open render already has
          // the uploader disabled — no window where a hinted file could start
          // uploading before the capability matrix arrives.
          setFileTypes({ status: 'loading' })
        } else {
          resetDialog()
        }
        setOpen(nextOpen)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="default" side="bottom" tooltip={t('documentPanel.uploadDocuments.tooltip')} size="sm">
          <UploadIcon /> {t('documentPanel.uploadDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.uploadDocuments.title')}</DialogTitle>
          <DialogDescription className="sr-only">
            {t('documentPanel.uploadDocuments.description')}
          </DialogDescription>
        </DialogHeader>

        {/* Step indicator */}
        <div className="flex items-center gap-2 mb-1">
          {[1, 2].map((s) => (
            <div key={s} className="flex items-center gap-2">
              {s > 1 && <div className={cn('h-px flex-1 w-8', step >= s ? 'bg-primary' : 'bg-muted')} />}
              <div className={cn(
                'flex size-6 items-center justify-center rounded-full text-xs font-medium',
                step > s
                  ? 'bg-primary text-primary-foreground'
                  : step === s
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground'
              )}>
                {step > s ? <CheckIcon className="size-3" /> : s}
              </div>
              <span className={cn('text-xs', step >= s ? 'text-foreground' : 'text-muted-foreground')}>
                {t(s === 1 ? 'documentPanel.uploadDocuments.step1Label' : 'documentPanel.uploadDocuments.step2Label')}
              </span>
            </div>
          ))}
        </div>

        {/* Step 1: File selection */}
        {step === 1 && (
          <div className="space-y-3">
            <FileUploader
              maxFileCount={Infinity}
              maxSize={200 * 1024 * 1024}
              description={t('documentPanel.uploadDocuments.fileTypes', {
                types: formatFileTypesLabel(
                  uploaderInputs.acceptedExtensions ?? flattenAcceptExtensions(supportedFileTypes)
                )
              })}
              value={pendingFiles}
              onValueChange={setPendingFiles}
              onReject={handleRejectedFiles}
              progresses={{}}
              fileErrors={fileErrors}
              disabled={uploaderInputs.disabled}
              acceptedExtensions={uploaderInputs.acceptedExtensions}
              engineCapabilities={uploaderInputs.engineCapabilities}
            />
            <div className="flex justify-end">
              <Button
                variant="default"
                size="sm"
                disabled={pendingFiles.length === 0}
                onClick={() => setStep(2)}
              >
                {t('documentPanel.uploadDocuments.nextStep')} →
              </Button>
            </div>
          </div>
        )}

        {/* Step 2: Chunking settings */}
        {step === 2 && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              {t('documentPanel.uploadDocuments.step2Hint', { count: pendingFiles.length })}
            </p>

            {/* Strategy cards */}
            <div className="space-y-1.5 max-h-[280px] overflow-y-auto pr-1">
              {CHUNKING_STRATEGIES.map((s) => (
                <button
                  key={s.value}
                  type="button"
                  onClick={() => handleSelectStrategy(s.value)}
                  className={cn(
                    'w-full rounded-md border px-3 py-2 text-left transition-colors',
                    selectedStrategy === s.value
                      ? 'border-primary bg-primary/5'
                      : 'border-input hover:border-primary/50 hover:bg-muted/40'
                  )}
                >
                  <div className="flex items-center gap-2">
                    <div className={cn(
                      'mt-0.5 size-3.5 shrink-0 rounded-full border-2',
                      selectedStrategy === s.value
                        ? 'border-primary bg-primary'
                        : 'border-muted-foreground'
                    )} />
                    <div className="min-w-0">
                      <p className="text-sm font-medium leading-tight">{t(s.labelKey)}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground leading-snug">
                        {s.value === '' ? serverDefaultDesc : t(s.descKey)}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {/* Size / Overlap inputs */}
            <div className={cn('grid gap-3', currentStrategy.supportsOverlap ? 'grid-cols-2' : 'grid-cols-1')}>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  {t('documentPanel.uploadDocuments.chunking.chunkTokenSize')}
                </label>
                <input
                  type="number"
                  min={1}
                  value={chunkingOptions.chunkTokenSize ?? ''}
                  onChange={(e) => setChunkingOptions((o) => ({
                    ...o,
                    chunkTokenSize: e.target.value ? Number(e.target.value) : undefined
                  }))}
                  placeholder={t('documentPanel.uploadDocuments.chunking.default')}
                  className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
              {currentStrategy.supportsOverlap && (
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">
                    {t('documentPanel.uploadDocuments.chunking.chunkOverlapTokenSize')}
                  </label>
                  <input
                    type="number"
                    min={0}
                    value={chunkingOptions.chunkOverlapTokenSize ?? ''}
                    onChange={(e) => setChunkingOptions((o) => ({
                      ...o,
                      chunkOverlapTokenSize: e.target.value ? Number(e.target.value) : undefined
                    }))}
                    placeholder={t('documentPanel.uploadDocuments.chunking.default')}
                    className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                </div>
              )}
            </div>

            {/* split_by_character — fixed_token only */}
            {selectedStrategy === 'fixed_token' && (
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  {t('documentPanel.uploadDocuments.chunking.splitByCharacter')}
                </label>
                <input
                  type="text"
                  value={splitByCharacter}
                  onChange={(e) => {
                    const val = e.target.value
                    setSplitByCharacter(val)
                    const unescaped = val.replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\r/g, '\r')
                    setChunkingOptions((o) => ({ ...o, splitByCharacter: unescaped || undefined }))
                  }}
                  placeholder={t('documentPanel.uploadDocuments.chunking.splitByCharacterPlaceholder')}
                  className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
            )}

            {/* separators — recursive_character only */}
            {selectedStrategy === 'recursive_character' && (
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  {t('documentPanel.uploadDocuments.chunking.separators')}
                </label>
                <p className="text-xs text-muted-foreground/70">
                  {t('documentPanel.uploadDocuments.chunking.separatorsDefault')}
                </p>
                {separators.length > 0 && (
                  <div className="flex flex-wrap gap-1 pt-0.5">
                    {separators.map((sep, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-xs font-mono"
                      >
                        {escapeForDisplay(sep)}
                        <button
                          type="button"
                          onClick={() => handleRemoveSeparator(idx)}
                          className="text-muted-foreground hover:text-foreground"
                        >
                          <XIcon className="size-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                <input
                  type="text"
                  value={separatorInput}
                  onChange={(e) => setSeparatorInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleAddSeparator(separatorInput)
                    }
                  }}
                  placeholder={t('documentPanel.uploadDocuments.chunking.separatorsPlaceholder')}
                  className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between pt-1">
              <Button variant="outline" size="sm" onClick={() => setStep(1)}>
                ← {t('documentPanel.uploadDocuments.prevStep')}
              </Button>
              <Button variant="default" size="sm" onClick={handleFinish}>
                {t('documentPanel.uploadDocuments.finish')}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
