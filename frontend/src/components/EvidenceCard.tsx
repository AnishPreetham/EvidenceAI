import { useState } from "react"
import { ChevronDown, FileText, Image as ImageIcon, Headphones } from "lucide-react"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { API_BASE_URL } from "@/lib/api"
import type { RerankedResult, Source } from "@/lib/types"
import { cn, clock } from "@/lib/utils"

const MODALITY_ICON = { text: FileText, image: ImageIcon, audio: Headphones } as const
const MODALITY_LABEL = { text: "Document", image: "Image", audio: "Audio" } as const

export function EvidenceCard({
  result,
  source,
}: {
  result: RerankedResult
  source?: Source
}) {
  const [open, setOpen] = useState(false)
  const { item } = result
  const Icon = MODALITY_ICON[item.modality] ?? FileText
  const filename = source?.filename ?? "unknown"

  const where =
    item.location.page_number != null
      ? `Page ${item.location.page_number}`
      : item.location.section
        ? item.location.section
        : item.location.timestamp_start != null && item.location.timestamp_end != null
          ? `${clock(item.location.timestamp_start)}–${clock(item.location.timestamp_end)}`
          : null

  return (
    <Card className="overflow-hidden !rounded-[14px] transition-colors duration-150 hover:border-violet-400/20">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3.5 px-4 py-3.5 text-left"
      >
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] bg-white/[0.05] font-mono text-[11px] font-semibold text-base-200">
          {String(result.rank).padStart(2, "0")}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-base-400">
            <Icon className="h-3 w-3" />
            {MODALITY_LABEL[item.modality] ?? "Document"}
          </div>
          <p className="mt-1 truncate text-sm font-medium text-base-100">{filename}</p>
          {where && <p className="mt-0.5 text-[11px] text-base-400">{where}</p>}
          {!open && (
            <p className="mt-1.5 line-clamp-1 text-xs italic text-base-400">
              "{item.content.slice(0, 140)}"
            </p>
          )}
          <div className="mt-2 flex items-center gap-4 text-[11px] text-base-400">
            <span>Relevance <span className="font-mono text-base-200">{result.retrieval_score.toFixed(2)}</span></span>
            <span>Rerank <span className="font-mono text-violet-300">{result.rerank_score.toFixed(2)}</span></span>
          </div>
        </div>

        <ChevronDown className={cn("mt-1 h-4 w-4 shrink-0 text-base-400 transition-transform duration-150", open && "rotate-180")} />
      </button>

      {open && (
        <div className="border-t border-white/[0.06] px-4 py-4 text-sm">
          {item.modality === "image" && item.location.image_path ? (
            <ImageEvidence item={item} />
          ) : item.modality === "audio" ? (
            <AudioEvidence item={item} source={source} />
          ) : (
            <p className="whitespace-pre-wrap leading-relaxed text-base-200">{item.content}</p>
          )}
          <p className="mt-3 font-mono text-[11px] text-base-400">
            item {item.item_id.slice(0, 8)} · source {item.source_id.slice(0, 8)}
          </p>
        </div>
      )}
    </Card>
  )
}

function ImageEvidence({ item }: { item: RerankedResult["item"] }) {
  const description = item.metadata?.visual_description as string | undefined
  const ocrText = item.metadata?.ocr_text as string | undefined
  const visionOk = item.metadata?.vision_ok

  return (
    <div className="space-y-3">
      {item.location.image_path && (
        <img
          src={`${API_BASE_URL}/source/${item.source_id}/file`}
          alt=""
          className="max-h-64 rounded-lg border border-white/[0.08] object-contain"
        />
      )}
      {description && (
        <div>
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-base-400">
            Visual description (Qwen2.5-VL)
          </p>
          <p className="leading-relaxed text-base-200">{description}</p>
        </div>
      )}
      {ocrText && (
        <div>
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-base-400">
            OCR text (Tesseract)
          </p>
          <pre className="whitespace-pre-wrap rounded-lg bg-black/30 p-3 font-mono text-xs text-base-200">
            {ocrText.slice(0, 1200)}
          </pre>
        </div>
      )}
      {visionOk === false && <Badge tone="amber">vision model unavailable — OCR-only</Badge>}
    </div>
  )
}

function AudioEvidence({ item, source }: { item: RerankedResult["item"]; source?: Source }) {
  return (
    <div className="space-y-2">
      {source && (
        <audio controls className="w-full" src={`${API_BASE_URL}/source/${item.source_id}/file`} />
      )}
      <p className="whitespace-pre-wrap leading-relaxed text-base-200">{item.content}</p>
    </div>
  )
}
