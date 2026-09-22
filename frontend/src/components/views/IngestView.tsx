import { useState } from "react"
import { useDropzone } from "react-dropzone"
import { AlertCircle, CheckCircle2, FileUp, X } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { Card } from "@/components/ui/Card"
import { useIngest } from "@/hooks/useEvidenceAi"
import type { IngestFileResult } from "@/lib/types"
import { cn } from "@/lib/utils"

const ACCEPTED = [
  ".pdf", ".doc", ".docx", ".txt", ".csv",
  ".png", ".jpg", ".jpeg", ".webp",
  ".wav", ".mp3", ".m4a",
]

export function IngestView() {
  const [queued, setQueued] = useState<File[]>([])
  const ingest = useIngest()

  const dropzone = useDropzone({
    accept: { "*/*": ACCEPTED },
    onDrop: (accepted) => setQueued((prev) => [...prev, ...accepted]),
  })

  function removeQueued(index: number) {
    setQueued((prev) => prev.filter((_, i) => i !== index))
  }

  function handleIngest() {
    if (queued.length === 0) return
    ingest.mutate(queued, { onSuccess: () => setQueued([]) })
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-10">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">Add sources</h1>
        <p className="mt-1.5 text-sm text-base-400">
          Upload each source once — it stays indexed across restarts, so you
          can ask any number of questions afterwards without re-uploading.
        </p>
      </div>

      <div
        {...dropzone.getRootProps()}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-white/[0.1] bg-white/[0.02] px-4 py-14 text-center transition-colors duration-150",
          dropzone.isDragActive && "border-violet-400/60 bg-violet-500/[0.05]",
        )}
      >
        <input {...dropzone.getInputProps()} />
        <div className="glow-brand flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500/25 to-cyan-500/25">
          <FileUp className="h-5 w-5 text-violet-300" />
        </div>
        <p className="text-sm text-base-300">
          Drop files here, or click to browse
          <br />
          <span className="text-[11px] text-base-400">
            PDF, DOC, DOCX, TXT, CSV, PNG, JPG, WEBP, WAV, MP3, M4A
          </span>
        </p>
      </div>

      {queued.length > 0 && (
        <div className="space-y-2">
          {queued.map((f, i) => (
            <div key={`${f.name}-${i}`} className="flex items-center justify-between rounded-[10px] border border-white/[0.06] bg-black/20 px-4 py-2.5">
              <span className="truncate text-sm text-base-200">{f.name}</span>
              <button onClick={() => removeQueued(i)} className="text-base-400 transition-colors hover:text-rose-400">
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}
          <Button onClick={handleIngest} disabled={ingest.isPending} className="w-full">
            {ingest.isPending ? "Processing…" : `Ingest ${queued.length} file(s)`}
          </Button>
        </div>
      )}

      {ingest.data && (
        <div className="space-y-2">
          {ingest.data.results.map((r: IngestFileResult, i: number) => (
            <Card
              key={i}
              className={cn(
                "animate-rise-in flex items-start gap-2.5 !rounded-[12px] px-4 py-3 text-sm",
                r.success ? "!border-emerald-500/20 !bg-emerald-500/[0.04]" : "!border-rose-500/20 !bg-rose-500/[0.04]",
              )}
            >
              {r.success ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              ) : (
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              )}
              <div>
                <p className="font-medium text-white">{r.filename}</p>
                {r.success ? (
                  <p className="text-xs text-base-400">
                    {r.items_indexed} items, {r.relationships} relationships
                  </p>
                ) : (
                  <p className="text-xs text-rose-300">{r.error_code}: {r.error}</p>
                )}
                {r.warnings?.map((w: string, wi: number) => (
                  <p key={wi} className="text-xs text-amber-300">{w}</p>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
