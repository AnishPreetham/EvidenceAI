import { useMemo, useState } from "react"
import { ChevronDown, FileText, Headphones, Image as ImageIcon } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { Card } from "@/components/ui/Card"
import { useSources } from "@/hooks/useEvidenceAi"
import type { Source, SourceType } from "@/lib/types"
import { cn } from "@/lib/utils"

const MODALITY_ICON: Record<string, typeof FileText> = { image: ImageIcon, audio: Headphones }

const DOCUMENT_TYPES: SourceType[] = ["pdf", "doc", "docx", "txt", "csv"]

const FILTERS: { id: string; label: string; match: (t: SourceType) => boolean }[] = [
  { id: "all", label: "All", match: () => true },
  { id: "documents", label: "Documents", match: (t) => DOCUMENT_TYPES.includes(t) },
  { id: "images", label: "Images", match: (t) => t === "image" },
  { id: "audio", label: "Audio", match: (t) => t === "audio" },
]

export function SourcesView() {
  const { data, isLoading } = useSources()
  const [openId, setOpenId] = useState<string | null>(null)
  const [filter, setFilter] = useState("all")

  const filtered = useMemo(() => {
    if (!data) return []
    const active = FILTERS.find((f) => f.id === filter) ?? FILTERS[0]
    return data.sources.filter((s) => active.match(s.source_type))
  }, [data, filter])

  if (isLoading) {
    return <div className="px-6 py-10 text-sm text-base-400">Loading sources…</div>
  }

  if (!data || data.sources.length === 0) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10">
        <Card className="!rounded-2xl px-6 py-10 text-center text-sm text-base-400">
          Nothing indexed yet. Use the Ingest tab.
        </Card>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4 px-6 py-10">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight text-white">Sources</h1>
        <span className="text-xs text-base-400">{data.sources.length} total</span>
      </div>

      <div className="flex gap-1 rounded-[10px] border border-white/[0.06] bg-base-900/50 p-1">
        {FILTERS.map((f) => {
          const count = f.id === "all" ? data.sources.length : data.sources.filter((s) => f.match(s.source_type)).length
          return (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={cn(
                "flex-1 rounded-[8px] px-3 py-1.5 text-sm font-medium transition-all duration-150",
                filter === f.id
                  ? "bg-gradient-to-r from-violet-500 to-blue-500 text-white shadow-[0_2px_12px_rgba(124,92,255,0.3)]"
                  : "text-base-400 hover:text-base-100",
              )}
            >
              {f.label} ({count})
            </button>
          )
        })}
      </div>

      {filtered.length === 0 && (
        <Card className="!rounded-[14px] px-6 py-8 text-center text-sm text-base-400">
          No sources match this filter.
        </Card>
      )}

      {filtered.map((source: Source) => {
        const Icon = MODALITY_ICON[source.source_type] ?? FileText
        const isOpen = openId === source.source_id
        return (
          <Card key={source.source_id} className="!rounded-[14px] transition-colors duration-150 hover:border-violet-400/20">
            <button
              onClick={() => setOpenId(isOpen ? null : source.source_id)}
              className="flex w-full items-center gap-3 px-4 py-3.5 text-left"
            >
              <span className="flex h-8 w-8 items-center justify-center rounded-[8px] bg-violet-500/10">
                <Icon className="h-4 w-4 text-violet-300" />
              </span>
              <span className="min-w-0 flex-1 truncate text-sm font-medium text-white">
                {source.filename}
              </span>
              <span className="font-mono text-[11px] text-base-400">
                {source.source_type} · {((source.file_size ?? 0) / 1024).toFixed(0)} KB
              </span>
              <ChevronDown className={cn("h-4 w-4 text-base-400 transition-transform duration-150", isOpen && "rotate-180")} />
            </button>
            {isOpen && <SourceItems sourceId={source.source_id} />}
          </Card>
        )
      })}
    </div>
  )
}

function SourceItems({ sourceId }: { sourceId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["source", sourceId],
    queryFn: () => api.sourceDetail(sourceId),
  })

  if (isLoading) {
    return <p className="border-t border-white/[0.06] px-4 py-3 text-xs text-base-400">Loading…</p>
  }
  if (!data) return null

  return (
    <div className="border-t border-white/[0.06] px-4 py-3">
      <p className="mb-2 font-mono text-[11px] text-base-400">
        source {data.source.source_id.slice(0, 8)} · {data.items.length} indexed items
      </p>
      <div className="space-y-1">
        {data.items.slice(0, 20).map((item: { item_id: string; content: string }) => (
          <p key={item.item_id} className="truncate text-xs text-base-300">
            {String(item.content).slice(0, 180)}
          </p>
        ))}
      </div>
    </div>
  )
}
