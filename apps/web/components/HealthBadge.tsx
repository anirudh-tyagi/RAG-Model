"use client";

import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { Health } from "@/lib/types";
import { useEffect, useState } from "react";

/**
 * Surfaces what `/api/health` reports. Worth having in the UI: the API stays up
 * when Qdrant, Redis or the model key are missing, so "why is nothing working?"
 * has a visible answer instead of only a log line.
 */
export function HealthBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let active = true;
    api
      .health()
      .then((result) => active && setHealth(result))
      .catch(() => active && setUnreachable(true));
    return () => {
      active = false;
    };
  }, []);

  if (unreachable) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-danger">
        <span className="h-1.5 w-1.5 rounded-full bg-danger" />
        API unreachable
      </span>
    );
  }

  if (!health) return null;

  const problems = [
    !health.llm_configured && "no model key",
    !health.qdrant && "vector store down",
    !health.queue && "queue down",
  ].filter(Boolean);

  return (
    <span
      className="flex items-center gap-1.5 text-xs text-muted"
      title={
        problems.length > 0
          ? `Degraded: ${problems.join(", ")}`
          : `${health.llm_model} · ${health.dense_model}${health.rerank_enabled ? " · reranking on" : ""}`
      }
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          problems.length === 0 ? "bg-success" : "bg-danger",
        )}
      />
      <span className="hidden sm:inline">
        {problems.length === 0 ? health.llm_model : `Degraded: ${problems[0]}`}
      </span>
    </span>
  );
}
