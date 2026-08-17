"use client";

import { Button } from "@/components/ui/Button";
import { AlertIcon, SpeakerIcon } from "@/components/ui/Icons";
import { cn } from "@/lib/cn";
import type { ChatMessage, Source } from "@/lib/types";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  message: ChatMessage;
  onCite: (source: Source) => void;
  onSpeak: (id: string, text: string) => void;
  speaking: boolean;
  speechSupported: boolean;
}

const CITATION_HREF = "#cite-";

// Full-width brackets are matched alongside ASCII: gpt-oss models routinely
// emit 【1】 rather than [1], and those must still become clickable chips.
const CITATION_PATTERN = /[[【](\d{1,2})[\]】]/g;

/**
 * Rewrite `[3]` into a markdown link so remark parses it into a node we can
 * render as a clickable chip. Fenced code blocks are left alone — a `[1]` in a
 * code sample is not a citation.
 */
function linkifyCitations(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?```)/g)
    .map((part, index) =>
      index % 2 === 1 ? part : part.replace(CITATION_PATTERN, `[$1](${CITATION_HREF}$1)`),
    )
    .join("");
}

export function MessageBubble({ message, onCite, onSpeak, speaking, speechSupported }: Props) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm text-accent-fg">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    );
  }

  const byNumber = new Map(message.sources.map((source) => [source.n, source]));

  const components: Components = {
    a({ href, children, ...props }) {
      if (href?.startsWith(CITATION_HREF)) {
        const n = Number(href.slice(CITATION_HREF.length));
        const source = byNumber.get(n);
        return (
          <button
            type="button"
            onClick={() => source && onCite(source)}
            disabled={!source}
            title={
              source
                ? `${source.doc_title}${source.page ? `, page ${source.page}` : ""}`
                : "Citation not found in the retrieved passages"
            }
            className={cn(
              "mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] items-center justify-center rounded px-1 align-[0.1em] text-[0.7rem] font-semibold no-underline transition-colors",
              source
                ? "bg-accent-soft text-accent hover:bg-accent hover:text-accent-fg"
                : "bg-raised text-faint line-through",
            )}
          >
            {n}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer" {...props}>
          {children}
        </a>
      );
    },
  };

  return (
    <div className="flex flex-col gap-2">
      <div
        className={cn(
          "prose-answer max-w-[92%] break-words rounded-2xl rounded-bl-md border border-border bg-panel px-4 py-3",
          message.streaming && "streaming-caret",
        )}
      >
        {message.content ? (
          <Markdown remarkPlugins={[remarkGfm]} components={components}>
            {linkifyCitations(message.content)}
          </Markdown>
        ) : message.streaming ? (
          <span className="flex items-center gap-1 py-0.5" aria-label="Thinking">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="typing-dot h-1.5 w-1.5 rounded-full bg-faint"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </span>
        ) : null}
      </div>

      {message.error && (
        <div className="flex items-start gap-2 rounded-lg border border-danger/40 bg-panel px-3 py-2 text-xs text-danger">
          <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
          <span>{message.error}</span>
        </div>
      )}

      {(message.sources.length > 0 || (speechSupported && message.content)) && (
        <div className="flex flex-wrap items-center gap-1.5 pl-1">
          {message.sources.map((source) => (
            <button
              key={source.chunk_id}
              type="button"
              onClick={() => onCite(source)}
              className="inline-flex max-w-[15rem] items-center gap-1.5 rounded-full border border-border bg-panel px-2 py-0.5 text-xs text-muted transition-colors hover:border-accent hover:text-fg"
            >
              <span className="font-semibold text-accent">{source.n}</span>
              <span className="truncate">{source.doc_title}</span>
              {source.page !== null && <span className="text-faint">p.{source.page}</span>}
            </button>
          ))}

          {speechSupported && message.content && !message.streaming && (
            <Button
              size="sm"
              onClick={() => onSpeak(message.id, message.content)}
              aria-label={speaking ? "Stop reading" : "Read aloud"}
              title={speaking ? "Stop reading" : "Read aloud"}
              className={cn("h-6 px-1.5", speaking && "text-accent")}
            >
              <SpeakerIcon className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
