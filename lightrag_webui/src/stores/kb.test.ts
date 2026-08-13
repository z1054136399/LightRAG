import { describe, it, expect, beforeEach } from 'bun:test'
import { useKBStore } from './kb'

describe('useKBStore', () => {
  beforeEach(() => {
    useKBStore.setState({ activeKbId: null })
  })

  it('defaults to no active KB', () => {
    expect(useKBStore.getState().activeKbId).toBeNull()
  })

  it('setActiveKbId sets the active KB', () => {
    useKBStore.getState().setActiveKbId('kb-123')
    expect(useKBStore.getState().activeKbId).toBe('kb-123')
  })

  it('setActiveKbId(null) clears the active KB', () => {
    useKBStore.getState().setActiveKbId('kb-123')
    useKBStore.getState().setActiveKbId(null)
    expect(useKBStore.getState().activeKbId).toBeNull()
  })
})
