/** Typed client for the RAG API. */

import type { DocumentOut, Health, TranscriptionOut, UploadAccepted } from "@/lib/types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Pull FastAPI's `detail` out of an error response, whatever shape it takes. */
async function toApiError(response: Response): Promise<ApiError> {
  let detail = `Request failed with ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      // Pydantic validation errors come back as a list.
      detail = body.detail[0].msg;
    }
  } catch {
    // Non-JSON error body; keep the generic message.
  }
  return new ApiError(response.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),

  listDocuments: () => request<DocumentOut[]>("/api/documents"),

  getDocument: (id: string) => request<DocumentOut>(`/api/documents/${id}`),

  uploadDocument: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadAccepted>("/api/documents", { method: "POST", body });
  },

  deleteDocument: (id: string) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),

  transcribe: (audio: Blob) => {
    const body = new FormData();
    body.append("audio", audio, "recording.webm");
    return request<TranscriptionOut>("/api/transcribe", { method: "POST", body });
  },

  /** URL for the document progress stream, consumed with `EventSource`. */
  documentEventsUrl: (id: string) => `${API_URL}/api/documents/${id}/events`,

  /** Opens the chat stream. The caller reads it with `readSse`. */
  chatStream: (
    payload: { message: string; conversation_id: string | null; doc_ids: string[] },
    signal: AbortSignal,
  ) =>
    fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    }),
};
