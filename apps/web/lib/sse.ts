/**
 * Minimal server-sent-event reader for `fetch` responses.
 *
 * The browser's own `EventSource` only does GET, and the chat endpoint is a POST
 * with a JSON body, so the stream is parsed by hand. Document progress, which
 * *is* a GET, uses `EventSource` directly instead of this.
 */

export interface SseFrame {
  event: string;
  data: string;
}

const FRAME_SEPARATOR = /\r?\n\r?\n/;

/** Parse one raw frame. Returns null for comment-only frames (keepalives). */
function parseFrame(raw: string): SseFrame | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith(":")) continue; // comment / keepalive
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).replace(/^ /, ""));
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/** Yield frames from a streaming response body as they arrive. */
export async function* readSse(response: Response): AsyncGenerator<SseFrame> {
  if (!response.body) throw new Error("response has no body to stream");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // A frame ends at a blank line. Anything after the last one is a partial
      // frame and stays in the buffer until more bytes arrive.
      while (true) {
        const match = FRAME_SEPARATOR.exec(buffer);
        if (!match || match.index === undefined) break;
        const raw = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const frame = parseFrame(raw);
        if (frame) yield frame;
      }
    }

    // Flush a trailing frame that arrived without its blank line.
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}
