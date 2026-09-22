import { useState } from "react"
import { AlertTriangle, CheckCircle2, Clock3, FileStack } from "lucide-react"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { EvidenceCard } from "@/components/EvidenceCard"
import type { Citation, QueryResponse, RerankedResult, Relationship, Source } from "@/lib/types"
import { cn } from "@/lib/utils"

const TABS = [
  { id: "evidence", label: "Evidence" },
  { id: "relationships", label: "Relationships" },
] as const

const REL_LABELS: Record<string, string> = {
  same_source: "Same source",
  same_page: "Same page",
  ocr_of: "OCR of",
  transcript_of: "Transcript of",
  references: "References",
  related_to: "Related to",
}

export function ResponsePanel({
  response,
  sources,
}: {
  response: QueryResponse
  sources: Source[]
}) {
  const sourceById = new Map(sources.map((s) => [s.source_id, s]))
  const [subTab, setSubTab] = useState<(typeof TABS)[number]["id"]>("evidence")

  // Build item→source lookup for relationship display
  const itemToSource = new Map(
    response.retrieved_items.map((r) => [r.item.item_id, sourceById.get(r.item.source_id)])
  )
  const itemToPage = new Map(
    response.retrieved_items.map((r) => [r.item.item_id, r.item.location.page_number])
  )

  // Group citations by source for the right rail
  const citationsBySource = new Map<string, Citation[]>()
  for (const c of response.citations) {
    const arr = citationsBySource.get(c.source_id) ?? []
    arr.push(c)
    citationsBySource.set(c.source_id, arr)
  }

  const evidenceStrength =
    response.confidence != null ? Math.round(response.confidence * 100) : null

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
      <div className="min-w-0 space-y-6">
        {/* Answer card ------------------------------------------------------ */}
        <Card
          className={cn(
            "animate-rise-in p-5",
            response.abstained && "!border-amber-500/25 !bg-amber-500/[0.04]",
          )}
        >
          <div className="flex items-center gap-2">
            {response.abstained ? (
              <AlertTriangle className="h-4 w-4 text-amber-400" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            )}
            <span className="text-sm font-semibold text-white">
              {response.abstained ? "Abstained — insufficient evidence" : "Grounded answer"}
            </span>
            {response.error_code && <Badge tone="amber">{response.error_code}</Badge>}
            {evidenceStrength != null && (
              <span
                title="Normalised rerank score — not a calibrated probability"
                className="ml-auto"
              >
                <Badge tone={response.abstained ? "amber" : "emerald"}>
                  {evidenceStrength}% evidence match
                </Badge>
              </span>
            )}
          </div>
          <p className="mt-3 whitespace-pre-wrap text-[15px] leading-relaxed text-base-100">
            {response.answer}
          </p>
          <p className="mt-3 text-xs text-base-400">
            {evidenceStrength != null &&
              "Evidence match = normalised rerank score, not a calibrated probability. "}
            Based on {response.retrieved_items.length} passage
            {response.retrieved_items.length === 1 ? "" : "s"} from your knowledge base.
          </p>
        </Card>

        {/* Evidence / Relationships tabs ------------------------------------- */}
        {(response.retrieved_items.length > 0 || response.relationships.length > 0) && (
          <div className="animate-rise-in">
            <div className="mb-3 flex gap-1 rounded-[10px] border border-white/[0.06] bg-base-900/50 p-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setSubTab(t.id)}
                  className={cn(
                    "flex-1 rounded-[8px] px-3 py-1.5 text-sm font-medium transition-all duration-150",
                    subTab === t.id
                      ? "bg-gradient-to-r from-violet-500 to-blue-500 text-white shadow-[0_2px_12px_rgba(124,92,255,0.3)]"
                      : "text-base-400 hover:text-base-100",
                  )}
                >
                  {t.label} (
                  {t.id === "evidence"
                    ? response.retrieved_items.length
                    : response.relationships.length}
                  )
                </button>
              ))}
            </div>

            {subTab === "evidence" && (
              <div className="space-y-2">
                {response.retrieved_items.map((r: RerankedResult) => (
                  <EvidenceCard
                    key={r.item.item_id}
                    result={r}
                    source={sourceById.get(r.item.source_id)}
                  />
                ))}
                {response.retrieved_items.length === 0 && (
                  <p className="px-1 text-sm text-base-400">No evidence retrieved.</p>
                )}
              </div>
            )}

            {subTab === "relationships" && (
              <div className="space-y-1.5 rounded-[14px] border border-white/[0.06] bg-black/20 p-3">
                {response.relationships.slice(0, 20).map((rel: Relationship) => {
                  const label = REL_LABELS[rel.relationship_type] ?? rel.relationship_type
                  const aSource = itemToSource.get(rel.source_item_id)
                  const bSource = itemToSource.get(rel.target_item_id)
                  const aPage = itemToPage.get(rel.source_item_id)
                  const bPage = itemToPage.get(rel.target_item_id)

                  // For same_page / same_source both items are in the same file by definition.
                  // Use whichever side we know.
                  const knownSource = aSource ?? bSource
                  const isSameFile =
                    rel.relationship_type === "same_page" ||
                    rel.relationship_type === "same_source" ||
                    (aSource != null && bSource != null && aSource.source_id === bSource.source_id)

                  let locationText = ""
                  if (isSameFile && knownSource) {
                    const pages = [aPage, bPage].filter((p): p is number => p != null)
                    const uniquePages = Array.from(new Set(pages)).sort((a, b) => a - b)
                    locationText = knownSource.filename
                    if (uniquePages.length > 1) locationText += ` · pages ${uniquePages.join(" & ")}`
                    else if (uniquePages.length === 1) locationText += ` · page ${uniquePages[0]}`
                  } else {
                    const aName = aSource?.filename ?? rel.source_item_id.slice(0, 8)
                    const bName = bSource?.filename ?? rel.target_item_id.slice(0, 8)
                    locationText =
                      `${aName}${aPage != null ? ` · p.${aPage}` : ""}` +
                      ` ↔ ${bName}${bPage != null ? ` · p.${bPage}` : ""}`
                  }

                  return (
                    <div
                      key={rel.relationship_id}
                      className="flex flex-wrap items-start gap-2 rounded-[8px] border border-white/[0.04] bg-white/[0.02] px-2.5 py-2 text-xs"
                    >
                      <Badge tone="neutral">{label}</Badge>
                      <span className="text-base-400">
                        {locationText}
                        {rel.confidence != null && (
                          <span className="ml-2 font-mono text-base-500">
                            {rel.confidence.toFixed(2)}
                          </span>
                        )}
                      </span>
                    </div>
                  )
                })}
                {response.relationships.length === 0 && (
                  <p className="text-sm text-base-400">
                    No relationships surfaced for this answer.
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Timing ------------------------------------------------------------ */}
        {response.latency_ms && Object.keys(response.latency_ms).length > 0 && (
          <details className="text-xs text-base-400">
            <summary className="cursor-pointer select-none">Timing</summary>
            <div className="mt-1.5 space-y-0.5 font-mono">
              {Object.entries(response.latency_ms).map(([stage, value]) => (
                <div key={stage} className="flex items-center gap-1.5">
                  <Clock3 className="h-3 w-3" />
                  {stage}: {value.toFixed(0)} ms
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Technical detail -------------------------------------------------- */}
        {response.technical_detail && (
          <details className="text-xs text-base-400">
            <summary className="cursor-pointer select-none">Technical details</summary>
            <pre className="mt-1.5 whitespace-pre-wrap rounded-lg bg-black/30 p-3 font-mono">
              {response.technical_detail}
            </pre>
          </details>
        )}
      </div>

      {/* Sources cited rail ------------------------------------------------- */}
      <div className="animate-rise-in lg:sticky lg:top-6 lg:self-start">
        <Card className="p-4">
          <h3 className="mb-3 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-base-400">
            <FileStack className="h-3.5 w-3.5" />
            Sources cited · {citationsBySource.size}
          </h3>

          {response.citations.length === 0 ? (
            <p className="text-xs text-base-400">No citations for this answer.</p>
          ) : (
            <div className="space-y-2">
              {Array.from(citationsBySource.entries()).map(([sourceId, cits]) => {
                const first = cits[0]
                const pages = Array.from(
                  new Set(
                    cits
                      .map((c) => c.location.page_number)
                      .filter((p): p is number => p != null),
                  ),
                ).sort((a, b) => a - b)

                return (
                  <div
                    key={sourceId}
                    className="rounded-[10px] border border-white/[0.06] bg-black/20 px-3 py-2.5 transition-colors duration-150 hover:border-violet-400/20"
                  >
                    <div className="flex items-start gap-2">
                      <span className="mt-0.5 flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full bg-violet-500/15 font-mono text-[10px] font-semibold text-violet-300">
                        {first.citation_id}
                      </span>
                      <div className="min-w-0">
                        <p className="truncate text-xs font-medium text-base-100">
                          {first.filename}
                        </p>
                        <p className="mt-0.5 text-[11px] text-base-400">
                          {first.modality}
                          {cits.length > 1 && ` · ${cits.length} passages`}
                          {pages.length > 0 && ` · pages ${pages.slice(0, 5).join(", ")}${pages.length > 5 ? "…" : ""}`}
                        </p>
                        {cits.length === 1 && first.excerpt && (
                          <p className="mt-1 line-clamp-2 text-[11px] italic text-base-400">
                            {first.excerpt}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
