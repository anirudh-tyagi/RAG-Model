"use client";

import { ApiError, api } from "@/lib/api";
import { readSse } from "@/lib/sse";
import type { ChatMessage, Source } from "@/lib/types";
import { useCallback, useRef, useState } from "react";

let counter = 0;
const nextId = () => `m${++counter}`;

export interface UseChat {
  messages: ChatMessage[];
  isStreaming: boolean;
  send: (text: string) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

/**
 * Drives the chat stream.
 *
 * Deliberately hand-rolled rather than using the Vercel AI SDK's `useChat`:
 * that expects its own data-stream wire format, which would have to be
 * reimplemented on the Python side. Reading our own `meta`/`sources`/`token`/
 * `done` events directly is less machinery and less to get wrong.
 */
export function useChat(docIds: string[]): UseChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const conversationId = useRef<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  const patch = useCallback((id: string, changes: Partial<ChatMessage>) => {
    setMessages((current) =>
      current.map((message) => (message.id === id ? { ...message, ...changes } : message)),
    );
  }, []);

  const appendTokens = useCallback((id: string, text: string) => {
    setMessages((current) =>
      current.map((message) =>
        message.id === id ? { ...message, content: message.content + text } : message,
      ),
    );
  }, []);

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || isStreaming) return;

      const assistantId = nextId();
      setMessages((current) => [
        ...current,
        { id: nextId(), role: "user", content: question, sources: [] },
        { id: assistantId, role: "assistant", content: "", sources: [], streaming: true },
      ]);
      setIsStreaming(true);

      controller.current = new AbortController();
      try {
        const response = await api.chatStream(
          {
            message: question,
            conversation_id: conversationId.current,
            doc_ids: docIds,
          },
          controller.current.signal,
        );

        if (!response.ok) {
          const detail = await response
            .json()
            .then((body) => body?.detail ?? `Request failed with ${response.status}`)
            .catch(() => `Request failed with ${response.status}`);
          throw new ApiError(response.status, String(detail));
        }

        for await (const frame of readSse(response)) {
          switch (frame.event) {
            case "meta":
              conversationId.current = JSON.parse(frame.data).conversation_id as string;
              break;
            case "sources":
              patch(assistantId, {
                sources: JSON.parse(frame.data).sources as Source[],
              });
              break;
            case "token":
              appendTokens(assistantId, JSON.parse(frame.data).text as string);
              break;
            case "error":
              patch(assistantId, {
                error: JSON.parse(frame.data).message as string,
                streaming: false,
              });
              break;
            case "done":
              patch(assistantId, { streaming: false });
              break;
            default:
              break;
          }
        }
      } catch (error) {
        // An aborted request is the user pressing stop, not a failure.
        const aborted = error instanceof DOMException && error.name === "AbortError";
        if (!aborted) {
          patch(assistantId, {
            error: error instanceof Error ? error.message : "Something went wrong.",
          });
        }
      } finally {
        patch(assistantId, { streaming: false });
        setIsStreaming(false);
        controller.current = null;
      }
    },
    [docIds, isStreaming, patch, appendTokens],
  );

  const stop = useCallback(() => {
    controller.current?.abort();
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    conversationId.current = null;
    setMessages([]);
  }, []);

  return { messages, isStreaming, send, stop, reset };
}
