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
  const [backend, setBackend] = useState<Backend>("offline")
  const [tab, setTab] = useState("ask")
  const { data: health } = useHealth(backend)
  const backendReady = Boolean(health?.inference.ready)
  const heading = TITLES[tab]

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
              backend === "offline"
                ? "bg-emerald-500/10 text-emerald-300 ring-emerald-500/20"
                : "bg-sky-500/10 text-sky-300 ring-sky-500/20",
            )}
          >
            {backend === "offline"
              ? "🔒 Offline — every stage runs on this machine"
              : "☁️ Online — retrieval stays local, answers written in the cloud"}
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
