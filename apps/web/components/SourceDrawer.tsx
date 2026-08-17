"use client";

import { Button } from "@/components/ui/Button";
import { CloseIcon } from "@/components/ui/Icons";
import type { Source } from "@/lib/types";
import { type MouseEvent, useEffect, useRef } from "react";

interface Props {
  source: Source | null;
  onClose: () => void;
}

/**
 * Shows the passage behind a citation. This is the whole point of carrying page
 * and heading through ingestion — the user can check an answer rather than take
 * it on faith.
 *
 * Built on the native `<dialog>` element, which brings focus trapping, Escape
 * handling and the backdrop for free instead of reimplementing all three.
 */
export function SourceDrawer({ source, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (source && !element.open) element.showModal();
    if (!source && element.open) element.close();
  }, [source]);

  // Clicks on the backdrop are dispatched to the dialog itself, so a click
  // whose target *is* the dialog landed outside the panel content.
  const onBackdropClick = (event: MouseEvent<HTMLDialogElement>) => {
    if (event.target === dialog.current) onClose();
  };

  return (
    // Backdrop dismissal is inherently a pointer affordance; keyboard users get
    // Escape (handled natively, surfaced via onClose) and the Close button.
    // biome-ignore lint/a11y/useKeyWithClickEvents: Escape and Close cover keyboards
    <dialog
      ref={dialog}
      onClose={onClose}
      onClick={onBackdropClick}
      aria-label={source ? `Source ${source.n}: ${source.doc_title}` : "Source"}
      className="source-drawer my-0 ml-auto mr-0 h-dvh max-h-none w-full max-w-md border-l border-border bg-panel p-0 text-fg"
    >
      {source && (
        <div className="flex h-full flex-col">
          <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-accent-soft px-1 text-xs font-semibold text-accent">
                  {source.n}
                </span>
                <h2 className="truncate text-sm font-semibold">{source.doc_title}</h2>
              </div>
              <p className="mt-1 text-xs text-muted">
                {source.page !== null && `Page ${source.page}`}
                {source.page !== null && source.heading && " · "}
                {source.heading}
              </p>
            </div>
            <Button size="icon" onClick={onClose} aria-label="Close">
              <CloseIcon className="h-4 w-4" />
            </Button>
          </header>

          <div className="flex-1 overflow-y-auto px-4 py-4">
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{source.excerpt}</p>
          </div>

          <footer className="border-t border-border px-4 py-2.5">
            <p className="text-xs text-faint">
              Relevance score {source.score.toFixed(3)} · excerpt from the retrieved passage
            </p>
          </footer>
        </div>
      )}
    </dialog>
  );
}
