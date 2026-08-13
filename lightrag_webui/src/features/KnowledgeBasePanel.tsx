import { useCallback } from 'react'
import KnowledgeBaseList from '@/features/KnowledgeBaseList'
import DocumentManager from '@/features/DocumentManager'
import { useKBStore } from '@/stores/kb'
import type { KBMeta } from '@/api/kb'

export default function KnowledgeBasePanel() {
  const activeKbId = useKBStore((s) => s.activeKbId)
  const setActiveKbId = useKBStore((s) => s.setActiveKbId)

  const handleSelect = useCallback(
    (kb: KBMeta) => {
      setActiveKbId(kb.id, kb.name)
    },
    [setActiveKbId]
  )

  if (!activeKbId) {
    return <KnowledgeBaseList onSelect={handleSelect} />
  }

  return <DocumentManager />
}
