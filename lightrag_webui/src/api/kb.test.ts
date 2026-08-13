// lightrag_webui/src/api/kb.test.ts
import { describe, it, expect, mock, beforeEach } from 'bun:test'

// `mock.module` is global for the whole `bun test` run and is never undone (see
// the same caveat documented in lightrag-stream.test.ts). A factory that returns
// only `axiosInstance` would delete every other export of '@/api/lightrag' for
// every test file that imports it afterward in the same run. Spreading the real
// module keeps the rest of the surface intact while still swapping in mocks for
// the HTTP methods this file exercises.
//
// Importing the real './lightrag' module runs its top-level code, which reads
// `localStorage` while initializing the auth store — stub it first so that
// module load does not throw outside a browser environment.
if (typeof globalThis.localStorage === 'undefined') {
  const storageData = new Map<string, string>()
  Object.defineProperty(globalThis, 'localStorage', {
    value: {
      getItem: (key: string) => storageData.get(key) ?? null,
      setItem: (key: string, value: string) => { storageData.set(key, value) },
      removeItem: (key: string) => { storageData.delete(key) },
      clear: () => { storageData.clear() }
    },
    configurable: true
  })
}

const realLightrag = await import('@/api/lightrag')

const mockGet = mock()
const mockPost = mock()
const mockPatch = mock()
const mockDelete = mock()

mock.module('@/api/lightrag', () => ({
  ...realLightrag,
  axiosInstance: {
    get: mockGet,
    post: mockPost,
    patch: mockPatch,
    delete: mockDelete
  }
}))

const { listKBs, createKB, getKB, updateKB, deleteKB } = await import('./kb')

describe('kb API client', () => {
  beforeEach(() => {
    mockGet.mockReset()
    mockPost.mockReset()
    mockPatch.mockReset()
    mockDelete.mockReset()
  })

  it('listKBs GETs /api/kbs', async () => {
    mockGet.mockResolvedValue({ data: [{ id: '1', name: 'A', description: '', created_at: '', default_chunking: null }] })
    const result = await listKBs()
    expect(mockGet).toHaveBeenCalledWith('/api/kbs')
    expect(result[0].name).toBe('A')
  })

  it('createKB POSTs the payload to /api/kbs', async () => {
    mockPost.mockResolvedValue({ data: { id: '2', name: 'B', description: 'd', created_at: '', default_chunking: null } })
    const result = await createKB({ name: 'B', description: 'd' })
    expect(mockPost).toHaveBeenCalledWith('/api/kbs', { name: 'B', description: 'd' })
    expect(result.id).toBe('2')
  })

  it('getKB GETs /api/kbs/{id}', async () => {
    mockGet.mockResolvedValue({ data: { id: '3', name: 'C', description: '', created_at: '', default_chunking: null } })
    await getKB('3')
    expect(mockGet).toHaveBeenCalledWith('/api/kbs/3')
  })

  it('updateKB PATCHes /api/kbs/{id}', async () => {
    mockPatch.mockResolvedValue({ data: { id: '4', name: 'D', description: '', created_at: '', default_chunking: null } })
    await updateKB('4', { name: 'D' })
    expect(mockPatch).toHaveBeenCalledWith('/api/kbs/4', { name: 'D' })
  })

  it('deleteKB DELETEs /api/kbs/{id}', async () => {
    mockDelete.mockResolvedValue({ data: { status: 'success' } })
    await deleteKB('5')
    expect(mockDelete).toHaveBeenCalledWith('/api/kbs/5')
  })
})
