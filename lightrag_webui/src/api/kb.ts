// lightrag_webui/src/api/kb.ts
import { axiosInstance } from '@/api/lightrag'

export type KBChunkingConfig = {
  strategy: 'fixed_token' | 'recursive_character' | 'semantic_vector' | 'paragraph_semantic'
  params: Record<string, unknown>
}

export type KBMeta = {
  id: string
  name: string
  description: string
  created_at: string
  default_chunking: KBChunkingConfig | null
}

export type CreateKBPayload = {
  name: string
  description?: string
  default_chunking?: KBChunkingConfig
}

export type UpdateKBPayload = {
  name?: string
  description?: string
  default_chunking?: KBChunkingConfig
  clear_default_chunking?: boolean
}

export const listKBs = async (): Promise<KBMeta[]> => {
  const response = await axiosInstance.get('/api/kbs')
  return response.data
}

export const createKB = async (payload: CreateKBPayload): Promise<KBMeta> => {
  const response = await axiosInstance.post('/api/kbs', payload)
  return response.data
}

export const getKB = async (id: string): Promise<KBMeta> => {
  const response = await axiosInstance.get(`/api/kbs/${id}`)
  return response.data
}

export const updateKB = async (id: string, patch: UpdateKBPayload): Promise<KBMeta> => {
  const response = await axiosInstance.patch(`/api/kbs/${id}`, patch)
  return response.data
}

export const deleteKB = async (id: string): Promise<{ status: string }> => {
  const response = await axiosInstance.delete(`/api/kbs/${id}`)
  return response.data
}
