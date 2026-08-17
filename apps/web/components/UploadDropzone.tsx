"use client";

import { Spinner } from "@/components/ui/Button";
import { UploadIcon } from "@/components/ui/Icons";
import { cn } from "@/lib/cn";
import { type DragEvent, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  uploading: boolean;
}

export function UploadDropzone({ onFile, uploading }: Props) {
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "rounded-xl border border-dashed p-4 text-center transition-colors",
        dragging ? "border-accent bg-accent-soft" : "border-border bg-panel",
      )}
    >
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          // Reset so re-picking the same file still fires a change event.
          event.target.value = "";
        }}
      />

      <button
        type="button"
        onClick={() => input.current?.click()}
        disabled={uploading}
        className="mx-auto flex w-full flex-col items-center gap-1.5 rounded-lg px-2 py-1 text-sm disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        {uploading ? (
          <Spinner className="text-accent" />
        ) : (
          <UploadIcon className="h-5 w-5 text-faint" />
        )}
        <span className="font-medium">{uploading ? "Uploading…" : "Add a PDF"}</span>
        <span className="text-xs text-faint">Drop a file here, or click to browse</span>
      </button>
    </div>
  );
}
