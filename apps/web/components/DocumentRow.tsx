"use client";

import { Button } from "@/components/ui/Button";
import { AlertIcon, FileIcon, TrashIcon } from "@/components/ui/Icons";
import { cn, formatBytes } from "@/lib/cn";
import type { DocumentOut } from "@/lib/types";

interface Props {
  document: DocumentOut;
  selected: boolean;
  onToggle: (id: string) => void;
  onDelete: (id: string) => void;
}

const STAGE_LABEL: Record<DocumentOut["stage"], string> = {
  queued: "Queued",
  parsing: "Reading PDF",
  captioning: "Describing figures",
  chunking: "Splitting text",
  embedding: "Indexing",
  ready: "Ready",
  failed: "Failed",
};

export function DocumentRow({ document, selected, onToggle, onDelete }: Props) {
  const ready = document.stage === "ready";
  const failed = document.stage === "failed";
  const busy = !ready && !failed;

  return (
    <li
      className={cn(
        "group rounded-lg border px-2.5 py-2 transition-colors",
        selected && ready ? "border-accent bg-accent-soft" : "border-border bg-panel",
      )}
    >
      <div className="flex items-start gap-2.5">
        <input
          type="checkbox"
          checked={selected}
          disabled={!ready}
          onChange={() => onToggle(document.id)}
          className="mt-0.5 h-4 w-4 shrink-0 accent-[var(--accent)] disabled:opacity-40"
          aria-label={`Include ${document.title} in questions`}
        />

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <FileIcon className="h-3.5 w-3.5 shrink-0 text-faint" />
            <p className="truncate text-sm font-medium" title={document.filename}>
              {document.title}
            </p>
          </div>

          {/* Announced as ingestion advances, which is how progress reaches
              assistive tech — the bar below is purely decorative. */}
          <p aria-live="polite" className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
            {failed && <AlertIcon className="h-3.5 w-3.5 shrink-0 text-danger" />}
            <span className={cn("truncate", failed && "text-danger")}>
              {busy ? document.detail : STAGE_LABEL[document.stage]}
            </span>
          </p>

          {ready && (
            <p className="mt-0.5 text-xs text-faint">
              {formatBytes(document.size_bytes)}
              {document.pages !== null && ` · ${document.pages} pages`}
              {document.chunk_count !== null && ` · ${document.chunk_count} passages`}
              {document.captioned_images > 0 && ` · ${document.captioned_images} figures`}
            </p>
          )}

          {failed && document.error && (
            <p className="mt-1 rounded bg-raised px-2 py-1 text-xs text-muted">{document.error}</p>
          )}

          {busy && (
            /* Decorative: the stage text above already states progress in words,
               and it live-updates, so a screen reader gets the better version. */
            <div aria-hidden="true" className="mt-1.5 h-1 overflow-hidden rounded-full bg-raised">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-500"
                style={{ width: `${Math.max(4, document.progress * 100)}%` }}
              />
            </div>
          )}
        </div>

        <Button
          size="icon"
          variant="danger"
          className="h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
          onClick={() => onDelete(document.id)}
          aria-label={`Delete ${document.title}`}
          title="Delete"
        >
          <TrashIcon className="h-3.5 w-3.5" />
        </Button>
      </div>
    </li>
  );
}
