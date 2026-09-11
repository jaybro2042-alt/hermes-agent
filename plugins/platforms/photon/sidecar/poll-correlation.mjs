export function pollGuidForOption(content, messageId = null) {
  if (!content || typeof content !== "object") return null;
  const explicit = (
    content.poll?.id ??
    content.poll?.guid ??
    content.poll?.messageId ??
    content.parent?.id ??
    content.parentId ??
    content.pollId ??
    null
  );
  if (explicit) return explicit;
  if (typeof messageId !== "string" || !messageId) return null;
  const encoded = /^(.*):[^:]+:[^:]+:(?:selected|deselected):\d+$/.exec(
    messageId
  );
  return encoded?.[1] ?? null;
}
