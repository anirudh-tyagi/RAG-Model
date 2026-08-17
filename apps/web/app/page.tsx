"use client";

import { ChatPanel } from "@/components/ChatPanel";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { SourceDrawer } from "@/components/SourceDrawer";
import type { Source } from "@/lib/types";
import { useChat } from "@/lib/useChat";
import { useDocuments } from "@/lib/useDocuments";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function Home() {
  const docs = useDocuments();
  const [selected, setSelected] = useState<string[]>([]);
  const [openSource, setOpenSource] = useState<Source | null>(null);

  const readyIds = useMemo(
    () => docs.documents.filter((d) => d.stage === "ready").map((d) => d.id),
    [docs.documents],
  );

  // Drop selections whose document was deleted, or that never finished
  // ingesting — an unready id would silently return nothing.
  useEffect(() => {
    setSelected((current) => {
      const pruned = current.filter((id) => readyIds.includes(id));
      return pruned.length === current.length ? current : pruned;
    });
  }, [readyIds]);

  const chat = useChat(selected);

  const toggle = useCallback((id: string) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id],
    );
  }, []);

  return (
    <main className="flex h-dvh flex-col overflow-hidden md:flex-row">
      <div className="h-64 shrink-0 md:h-full md:w-80">
        <DocumentSidebar
          docs={docs}
          selected={selected}
          onToggle={toggle}
          onSelectAll={() => setSelected(readyIds)}
          onClearSelection={() => setSelected([])}
        />
      </div>

      <ChatPanel
        messages={chat.messages}
        streaming={chat.isStreaming}
        readyCount={readyIds.length}
        onSend={chat.send}
        onStop={chat.stop}
        onReset={chat.reset}
        onCite={setOpenSource}
      />

      <SourceDrawer source={openSource} onClose={() => setOpenSource(null)} />
    </main>
  );
}
