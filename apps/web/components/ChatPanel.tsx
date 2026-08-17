"use client";

import { Composer } from "@/components/Composer";
import { HealthBadge } from "@/components/HealthBadge";
import { MessageBubble } from "@/components/MessageBubble";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/Button";
import { NewChatIcon } from "@/components/ui/Icons";
import type { ChatMessage, Source } from "@/lib/types";
import { useSpeech } from "@/lib/useVoice";
import { useEffect, useRef } from "react";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  readyCount: number;
  onSend: (text: string) => void;
  onStop: () => void;
  onReset: () => void;
  onCite: (source: Source) => void;
}

const EXAMPLES = [
  "Summarise the main findings.",
  "What numbers are reported in the figures?",
  "Which methods or datasets are used?",
];

export function ChatPanel({
  messages,
  streaming,
  readyCount,
  onSend,
  onStop,
  onReset,
  onCite,
}: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const speech = useSpeech();

  // Follow the stream, but stop following if the user scrolls up to read.
  // `messages` is the trigger, not an input: the effect reads the DOM, not state.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run on each token
  useEffect(() => {
    const element = scroller.current;
    if (!element || !pinned.current) return;
    element.scrollTop = element.scrollHeight;
  }, [messages]);

  const onScroll = () => {
    const element = scroller.current;
    if (!element) return;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    pinned.current = distance < 80;
  };

  const noDocuments = readyCount === 0;

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold">Document RAG</h1>
          <p className="truncate text-xs text-faint">
            Hybrid retrieval · reranking · answers cited to the page
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <HealthBadge />
          {messages.length > 0 && (
            <Button size="sm" onClick={onReset} title="Start a new conversation">
              <NewChatIcon className="h-3.5 w-3.5" />
              New
            </Button>
          )}
          <ThemeToggle />
        </div>
      </header>

      <div
        ref={scroller}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-y-auto px-4 py-5 md:px-5"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-5">
          {messages.length === 0 ? (
            <div className="pt-10 text-center">
              <h2 className="text-base font-semibold">
                {noDocuments ? "Upload a PDF to get started" : "Ask a question"}
              </h2>
              <p className="mx-auto mt-1.5 max-w-md text-sm text-muted">
                {noDocuments
                  ? "Once a document has finished processing, you can ask questions about it and every answer will cite the page it came from."
                  : "Answers are grounded in the passages retrieved from your documents. Click any citation to see the source text."}
              </p>

              {!noDocuments && (
                <div className="mt-5 flex flex-wrap justify-center gap-2">
                  {EXAMPLES.map((example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() => onSend(example)}
                      className="rounded-full border border-border bg-panel px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent hover:text-fg"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onCite={onCite}
                onSpeak={speech.speak}
                speaking={speech.speakingId === message.id}
                speechSupported={speech.supported}
              />
            ))
          )}
        </div>
      </div>

      <Composer
        onSend={onSend}
        onStop={onStop}
        streaming={streaming}
        disabled={noDocuments}
        placeholder={noDocuments ? "Upload a PDF first…" : "Ask about your documents…"}
      />
    </section>
  );
}
