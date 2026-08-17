"use client";

import { DocumentRow } from "@/components/DocumentRow";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Button, Spinner } from "@/components/ui/Button";
import { AlertIcon, CloseIcon } from "@/components/ui/Icons";
import type { UseDocuments } from "@/lib/useDocuments";

interface Props {
  docs: UseDocuments;
  selected: string[];
  onToggle: (id: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
}

export function DocumentSidebar({
  docs,
  selected,
  onToggle,
  onSelectAll,
  onClearSelection,
}: Props) {
  const readyCount = docs.documents.filter((d) => d.stage === "ready").length;
  const allSelected = readyCount > 0 && selected.length === readyCount;

  return (
    <aside className="flex h-full w-full flex-col gap-3 overflow-hidden border-border bg-bg p-3 md:w-80 md:border-r">
      <UploadDropzone onFile={docs.upload} uploading={docs.uploading} />

      {docs.error && (
        <div className="flex items-start gap-2 rounded-lg border border-danger/40 bg-panel px-2.5 py-2 text-xs text-danger">
          <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
          <span className="flex-1">{docs.error}</span>
          <button type="button" onClick={docs.dismissError} aria-label="Dismiss error">
            <CloseIcon className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <div className="flex items-center justify-between px-0.5">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-faint">
          Documents{docs.documents.length > 0 && ` (${docs.documents.length})`}
        </h2>
        {readyCount > 1 && (
          <Button size="sm" onClick={allSelected ? onClearSelection : onSelectAll}>
            {allSelected ? "Clear" : "Select all"}
          </Button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {docs.loading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted">
            <Spinner />
            Loading…
          </div>
        ) : docs.documents.length === 0 ? (
          <p className="px-1 py-6 text-center text-sm text-faint">
            No documents yet. Upload a PDF to start asking questions about it.
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {docs.documents.map((document) => (
              <DocumentRow
                key={document.id}
                document={document}
                selected={selected.includes(document.id)}
                onToggle={onToggle}
                onDelete={docs.remove}
              />
            ))}
          </ul>
        )}
      </div>

      {readyCount > 0 && (
        <p className="border-t border-border px-1 pt-2.5 text-xs text-faint">
          {selected.length === 0
            ? "Searching all ready documents."
            : `Searching ${selected.length} of ${readyCount} documents.`}
        </p>
      )}
    </aside>
  );
}
