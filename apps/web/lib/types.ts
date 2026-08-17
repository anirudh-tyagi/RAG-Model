/**
 * Mirrors the Pydantic schemas in `apps/api/src/rag/schemas.py`.
 * Keep the two in step — the API is the source of truth.
 */

export type IngestStage =
  | "queued"
  | "parsing"
  | "captioning"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

export const TERMINAL_STAGES: readonly IngestStage[] = ["ready", "failed"];

export interface DocumentOut {
  id: string;
  filename: string;
  title: string;
  size_bytes: number;
  stage: IngestStage;
  detail: string;
  progress: number;
  pages: number | null;
  chunk_count: number | null;
  captioned_images: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface UploadAccepted {
  document: DocumentOut;
  message: string;
}

export interface Source {
  n: number;
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  page: number | null;
  heading: string | null;
  excerpt: string;
  score: number;
}

export interface TranscriptionOut {
  text: string;
  language: string | null;
  duration_s: number | null;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  qdrant: boolean;
  queue: boolean;
  llm_configured: boolean;
  dense_model: string;
  llm_model: string;
  parser: string;
  rerank_enabled: boolean;
}

/** Server-sent event payloads, discriminated by the SSE event name. */
export type ChatEvent =
  | { name: "meta"; data: { conversation_id: string } }
  | { name: "sources"; data: { sources: Source[] } }
  | { name: "token"; data: { text: string } }
  | { name: "done"; data: { took_ms: number } }
  | { name: "error"; data: { message: string } };

export interface ProgressEventData {
  document: DocumentOut;
}

// --- client-side view models --------------------------------------------------

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  sources: Source[];
  /** Set while tokens are still arriving, so the UI can show a caret. */
  streaming?: boolean;
  error?: string | null;
}

export function isTerminal(stage: IngestStage): boolean {
  return TERMINAL_STAGES.includes(stage);
}
