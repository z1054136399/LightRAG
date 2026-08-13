import { create } from 'zustand'

type KBState = {
  activeKbId: string | null
  activeKbName: string | null
  setActiveKbId: (id: string | null, name?: string | null) => void
}

export const useKBStore = create<KBState>((set) => ({
  activeKbId: null,
  activeKbName: null,
  setActiveKbId: (id, name = null) => set({ activeKbId: id, activeKbName: id ? (name ?? null) : null })
}))
