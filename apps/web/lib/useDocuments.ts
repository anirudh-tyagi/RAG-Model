"use client";

import { api } from "@/lib/api";
import { type DocumentOut, type ProgressEventData, isTerminal } from "@/lib/types";
import { useCallback, useEffect, useRef, useState } from "react";

export interface UseDocuments {
  documents: DocumentOut[];
  loading: boolean;
  uploading: boolean;
  error: string | null;
  upload: (file: File) => Promise<DocumentOut | null>;
  remove: (id: string) => Promise<void>;
  dismissError: () => void;
}

/**
 * Loads the document list and keeps in-flight ingestions live.
 *
 * Progress arrives over SSE per document rather than by polling, which is what
 * lets the UI name the actual stage ("Describing 12 figures") instead of the old
 * front end's fixed 1.5 second wait followed by an unconditional redirect.
 */
export function useDocuments(): UseDocuments {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const streams = useRef(new Map<string, EventSource>());

  const upsert = useCallback((document: DocumentOut) => {
    setDocuments((current) => {
      const index = current.findIndex((d) => d.id === document.id);
      if (index === -1) return [document, ...current];
      const next = [...current];
      next[index] = document;
      return next;
    });
  }, []);

  // --- initial load ---------------------------------------------------------

  useEffect(() => {
    let active = true;
    api
      .listDocuments()
      .then((list) => {
        if (active) setDocuments(list);
      })
      .catch((cause: unknown) => {
        if (active) {
          setError(
            cause instanceof Error
              ? `Could not reach the API: ${cause.message}`
              : "Could not reach the API.",
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  // --- live progress --------------------------------------------------------

  useEffect(() => {
    const open = streams.current;

    for (const document of documents) {
      if (isTerminal(document.stage) || open.has(document.id)) continue;

      const source = new EventSource(api.documentEventsUrl(document.id));
      open.set(document.id, source);

      source.addEventListener("progress", (raw) => {
        const payload = JSON.parse((raw as MessageEvent<string>).data) as ProgressEventData;
        upsert(payload.document);
        if (isTerminal(payload.document.stage)) {
          source.close();
          open.delete(payload.document.id);
        }
      });

      source.onerror = () => {
        // EventSource retries on its own; close only once the work is finished
        // so a transient blip doesn't abandon a running ingestion.
        if (source.readyState === EventSource.CLOSED) {
          open.delete(document.id);
        }
      };
    }

    // Drop streams for documents that no longer exist.
    for (const [id, source] of open) {
      if (!documents.some((d) => d.id === id)) {
        source.close();
        open.delete(id);
      }
    }
  }, [documents, upsert]);

  useEffect(
    () => () => {
      for (const source of streams.current.values()) source.close();
      streams.current.clear();
    },
    [],
  );

  // --- mutations ------------------------------------------------------------

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        const accepted = await api.uploadDocument(file);
        upsert(accepted.document);
        return accepted.document;
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Upload failed.");
        return null;
      } finally {
        setUploading(false);
      }
    },
    [upsert],
  );

  const remove = useCallback(async (id: string) => {
    try {
      await api.deleteDocument(id);
      streams.current.get(id)?.close();
      streams.current.delete(id);
      setDocuments((current) => current.filter((d) => d.id !== id));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete the document.");
    }
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  return { documents, loading, uploading, error, upload, remove, dismissError };
}
