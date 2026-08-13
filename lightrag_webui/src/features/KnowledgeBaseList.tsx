// lightrag_webui/src/features/KnowledgeBaseList.tsx
import { useState, useEffect, useCallback, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { PlusIcon, TrashIcon, DatabaseIcon } from 'lucide-react'

import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import EmptyCard from '@/components/ui/EmptyCard'
import { Card, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import { listKBs, createKB, deleteKB, type KBMeta } from '@/api/kb'
import { errorMessage } from '@/lib/utils'

type KnowledgeBaseListProps = {
  onSelect: (kb: KBMeta) => void
}

function KbIdRow({ id }: { id: string }) {
  const { t } = useTranslation()
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    navigator.clipboard.writeText(id).then(() => {
      setCopied(true)
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => setCopied(false), 1500)
    })
  }, [id])

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="mt-0.5 flex items-center gap-1 text-left font-mono text-[10px] text-muted-foreground hover:text-foreground transition-colors"
      title={t('documentPanel.knowledgeBases.card.copyId')}
    >
      <span className="truncate max-w-[180px]">{id}</span>
      <span className="shrink-0">{copied ? '✓' : '⎘'}</span>
    </button>
  )
}

export default function KnowledgeBaseList({ onSelect }: KnowledgeBaseListProps) {
  const { t } = useTranslation()
  const [kbs, setKbs] = useState<KBMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [createOpen, setCreateOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<KBMeta | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setKbs(await listKBs())
    } catch (err) {
      toast.error(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh()
  }, [refresh])

  if (loading) {
    return null
  }

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">{t('documentPanel.knowledgeBases.title')}</h1>
        <CreateKBDialog
          open={createOpen}
          onOpenChange={setCreateOpen}
          onCreated={async (kb) => {
            await refresh()
            onSelect(kb)
          }}
        />
      </div>

      {kbs.length === 0 ? (
        <EmptyCard
          icon={DatabaseIcon}
          title={t('documentPanel.knowledgeBases.emptyTitle')}
          description={t('documentPanel.knowledgeBases.emptyDescription')}
          action={
            <Button variant="default" onClick={() => setCreateOpen(true)}>
              <PlusIcon /> {t('documentPanel.knowledgeBases.createButton')}
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {kbs.map((kb) => (
            <Card
              key={kb.id}
              className="hover:border-primary cursor-pointer transition-colors"
              onClick={() => onSelect(kb)}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <CardTitle>{kb.name}</CardTitle>
                  <Button
                    variant="ghost"
                    size="icon"
                    tooltip={t('documentPanel.knowledgeBases.card.deleteAction')}
                    onClick={(e) => {
                      e.stopPropagation()
                      setDeleteTarget(kb)
                    }}
                  >
                    <TrashIcon className="size-4" />
                  </Button>
                </div>
                {kb.description ? <CardDescription>{kb.description}</CardDescription> : null}
                <CardDescription>
                  {t('documentPanel.knowledgeBases.card.createdAt', {
                    date: new Date(kb.created_at).toLocaleDateString()
                  })}
                </CardDescription>
                <KbIdRow id={kb.id} />
              </CardHeader>
            </Card>
          ))}
        </div>
      )}

      <DeleteKBDialog
        kb={deleteTarget}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null)
        }}
        onDeleted={refresh}
      />
    </div>
  )
}

function CreateKBDialog({
  open,
  onOpenChange,
  onCreated
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated: (kb: KBMeta) => Promise<void>
}) {
  const { t } = useTranslation()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const reset = () => {
    setName('')
    setDescription('')
    setSubmitting(false)
  }

  const handleSubmit = async () => {
    if (!name.trim()) return
    setSubmitting(true)
    try {
      const kb = await createKB({ name: name.trim(), description: description.trim() })
      toast.success(t('documentPanel.knowledgeBases.createDialog.success'))
      onOpenChange(false)
      reset()
      await onCreated(kb)
    } catch (err) {
      const status = (err as { response?: { status?: number } })?.response?.status
      if (status === 409) {
        toast.error(t('documentPanel.knowledgeBases.createDialog.duplicateNameError'))
      } else {
        toast.error(errorMessage(err))
      }
      setSubmitting(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) reset()
        onOpenChange(next)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="default" onClick={() => onOpenChange(true)}>
          <PlusIcon /> {t('documentPanel.knowledgeBases.createButton')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t('documentPanel.knowledgeBases.createDialog.title')}</DialogTitle>
          <DialogDescription />
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={t('documentPanel.knowledgeBases.createDialog.namePlaceholder')}
            aria-label={t('documentPanel.knowledgeBases.createDialog.nameLabel')}
          />
          <Input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={t('documentPanel.knowledgeBases.createDialog.descriptionPlaceholder')}
            aria-label={t('documentPanel.knowledgeBases.createDialog.descriptionLabel')}
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('documentPanel.knowledgeBases.createDialog.cancel')}
          </Button>
          <Button variant="default" disabled={!name.trim() || submitting} onClick={handleSubmit}>
            {t('documentPanel.knowledgeBases.createDialog.submit')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function DeleteKBDialog({
  kb,
  onOpenChange,
  onDeleted
}: {
  kb: KBMeta | null
  onOpenChange: (open: boolean) => void
  onDeleted: () => Promise<void>
}) {
  const { t } = useTranslation()
  const [confirmText, setConfirmText] = useState('')
  const [deleting, setDeleting] = useState(false)

  const handleDelete = async () => {
    if (!kb || confirmText !== kb.name) return
    setDeleting(true)
    try {
      await deleteKB(kb.id)
      toast.success(t('documentPanel.knowledgeBases.deleteDialog.success'))
      setConfirmText('')
      onOpenChange(false)
      await onDeleted()
    } catch (err) {
      toast.error(errorMessage(err))
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Dialog open={kb !== null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t('documentPanel.knowledgeBases.deleteDialog.title')}</DialogTitle>
          <DialogDescription>
            {kb ? t('documentPanel.knowledgeBases.deleteDialog.warning', { name: kb.name }) : null}
          </DialogDescription>
        </DialogHeader>
        <Input
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
          placeholder={t('documentPanel.knowledgeBases.deleteDialog.confirmPrompt')}
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('documentPanel.knowledgeBases.deleteDialog.cancel')}
          </Button>
          <Button
            variant="destructive"
            disabled={!kb || confirmText !== kb.name || deleting}
            onClick={handleDelete}
          >
            {t('documentPanel.knowledgeBases.deleteDialog.confirmButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
