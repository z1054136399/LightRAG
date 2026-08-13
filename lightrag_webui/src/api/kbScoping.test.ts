import { describe, it, expect } from 'bun:test'
import { scopeUrlToActiveKb } from './kbScoping'

describe('scopeUrlToActiveKb', () => {
  it('returns the url unchanged when no active KB is set', () => {
    expect(scopeUrlToActiveKb('/documents', null)).toBe('/documents')
  })

  it('prefixes /documents paths with /kbs/{id} when an active KB is set', () => {
    expect(scopeUrlToActiveKb('/documents', 'kb-1')).toBe('/api/kbs/kb-1/documents')
  })

  it('prefixes /query paths', () => {
    expect(scopeUrlToActiveKb('/query', 'kb-1')).toBe('/api/kbs/kb-1/query')
  })

  it('prefixes /graphs and /graph/* paths', () => {
    expect(scopeUrlToActiveKb('/graphs?label=x', 'kb-1')).toBe('/api/kbs/kb-1/graphs?label=x')
    expect(scopeUrlToActiveKb('/graph/label/list', 'kb-1')).toBe('/api/kbs/kb-1/graph/label/list')
  })

  it('does not rewrite KB-management paths', () => {
    expect(scopeUrlToActiveKb('/api/kbs', 'kb-1')).toBe('/api/kbs')
    expect(scopeUrlToActiveKb('/api/kbs/kb-1', 'kb-1')).toBe('/api/kbs/kb-1')
  })

  it('prefixes /multimodal paths', () => {
    expect(scopeUrlToActiveKb('/multimodal/doc/img1', 'kb-1')).toBe('/api/kbs/kb-1/multimodal/doc/img1')
  })

  it('does not rewrite unrelated paths', () => {
    expect(scopeUrlToActiveKb('/health', 'kb-1')).toBe('/health')
  })

  it('does not double-prefix an already-scoped url', () => {
    expect(scopeUrlToActiveKb('/api/kbs/kb-1/documents', 'kb-1')).toBe('/api/kbs/kb-1/documents')
  })

  it('passes through undefined unchanged', () => {
    expect(scopeUrlToActiveKb(undefined, 'kb-1')).toBeUndefined()
  })
})
