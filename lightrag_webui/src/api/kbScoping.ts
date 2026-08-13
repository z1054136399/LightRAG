/** Path prefixes that are per-knowledge-base and must be rewritten to
 * `/api/kbs/{activeKbId}/...` when a KB is active. Kept in sync with the
 * routes KBRegistry._mount mounts per KB (documents/query/graph). */
const KB_SCOPED_PATH_PREFIXES = ['/documents', '/query', '/graph', '/graphs', '/multimodal']

export function scopeUrlToActiveKb(
  url: string | undefined,
  activeKbId: string | null
): string | undefined {
  if (!url || !activeKbId) return url
  if (url.startsWith('/api/kbs/')) return url // already scoped
  const isScoped = KB_SCOPED_PATH_PREFIXES.some((prefix) => url === prefix || url.startsWith(prefix))
  if (!isScoped) return url
  return `/api/kbs/${activeKbId}${url}`
}
