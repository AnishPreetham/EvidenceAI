import { useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { AskView } from "@/components/views/AskView"
import { IngestView } from "@/components/views/IngestView"
import { SourcesView } from "@/components/views/SourcesView"
import { useHealth } from "@/hooks/useEvidenceAi"
import type { Backend } from "@/lib/types"
import { cn } from "@/lib/utils"

const TITLES: Record<string, { title: string; subtitle: string }> = {
  ask: { title: "Ask Evidence", subtitle: "Search across your documents, images, and audio with AI" },
  ingest: { title: "Ingestion Studio", subtitle: "Upload and process documents, images, and audio" },
  sources: { title: "Knowledge Library", subtitle: "Browse and manage your indexed sources" },
}

export default function App() {
  const [backend, setBackend] = useState<Backend>("auto")
  const [tab, setTab] = useState("ask")
  const { data: health } = useHealth(backend)
  const backendReady = Boolean(health?.inference.ready)
  const heading = TITLES[tab]

  // Resolved provider for display: for AUTO read what the backend chose.
  const activeProvider =
    backend === "auto"
      ? (health?.inference.active_provider ?? null)
      : backend

  return (
    <div className="flex h-screen bg-base-950">
      <Sidebar backend={backend} onBackendChange={setBackend} tab={tab} onTabChange={setTab} />
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-white/[0.06] bg-base-900/30 px-8 py-5">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">{heading.title}</h1>
            <p className="mt-0.5 text-sm text-base-400">{heading.subtitle}</p>
          </div>
          <span
            className={cn(
              "shrink-0 rounded-full px-3 py-1.5 text-[11px] font-medium ring-1 ring-inset",
              activeProvider === "online"
                ? "bg-violet-500/10 text-violet-300 ring-violet-500/20"
                : activeProvider === "offline"
                ? "bg-emerald-500/10 text-emerald-300 ring-emerald-500/20"
                : "bg-base-800/60 text-base-400 ring-white/[0.06]",
            )}
          >
            {activeProvider === "online"
              ? "☁️ Online — retrieval stays local, answers written in the cloud"
              : activeProvider === "offline"
              ? "🔒 Offline — every stage runs on this machine"
              : "⏳ Checking provider…"}
          </span>
        </div>
        <div className="flex-1 overflow-y-auto">
          {tab === "ask" && <AskView backend={backend} backendReady={backendReady} />}
          {tab === "ingest" && <IngestView />}
          {tab === "sources" && <SourcesView />}
        </div>
      </main>
    </div>
  )
}
