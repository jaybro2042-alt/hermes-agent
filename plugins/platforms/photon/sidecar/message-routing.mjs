export function shouldForwardSpectrumMessage(message) {
  if (!message || typeof message !== "object") return true;
  if (!message.direction || message.direction === "inbound") return true;
  return message.direction === "outbound" && message.content?.type === "poll";
}
