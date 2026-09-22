import { useState } from "react"
import { useDropzone } from "react-dropzone"
import { ArrowLeft, ArrowRight, Sparkles, Upload } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { Textarea } from "@/components/ui/Textarea"
import { Slider } from "@/components/ui/Slider"
import { Card } from "@/components/ui/Card"
import { ResponsePanel } from "@/components/ResponsePanel"
import { useRunQuery, useRunQueryFile, useSources } from "@/hooks/useEvidenceAi"
import type { Backend, QueryModality } from "@/lib/types"
import { cn } from "@/lib/utils"

const EXAMPLE_PROMPTS = [
  "Summarize the key findings",
  "Find supporting evidence",
  "Compare these reports",
  "What do the sources say about risk?",
]

const MODES: { id: QueryModality; label: string; ext: string[] }[] = [
  { id: "text", label: "Text", ext: [] },
  { id: "image", label: "Image", ext: [".png", ".jpg", ".jpeg", ".webp"] },
  { id: "audio", label: "Audio", ext: [".wav", ".mp3", ".m4a"] },
  { id: "document", label: "Document", ext: [".pdf", ".docx", ".txt", ".csv"] },
]

export function AskView({ backend, backendReady }: { backend: Backend; backendReady: boolean }) {
  const [modality, setModality] = useState<QueryModality>("text")
  const [question, setQuestion] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [topK, setTopK] = useState(10)
  const [rerankTopK, setRerankTopK] = useState(5)
  const [useVision, setUseVision] = useState(false)
  const [showTuning, setShowTuning] = useState(false)

  const { data: sourcesData } = useSources()
  const textQuery = useRunQuery(backend)
  const fileQuery = useRunQueryFile(backend)

  const active = modality === "text" ? textQuery : fileQuery
  const mode = MODES.find((m) => m.id === modality)!

  const dropzone = useDropzone({
    multiple: false,
    accept: mode.ext.length ? { "*/*": mode.ext } : undefined,
    onDrop: (accepted) => setFile(accepted[0] ?? null),
  })

  const canSearch = backendReady && (modality === "text" ? question.trim().length > 0 : file != null)

  function handleSearch() {
    if (modality === "text") {
      textQuery.mutate({
        query: question.trim(),
        top_k: topK,
        rerank_top_k: rerankTopK,
        inference_mode: useVision ? "local_vision" : "local",
        query_modality: "text",
      })
      return
    }
    if (!file) return
    fileQuery.mutate({
      file,
      instruction: question.trim() || `Find evidence related to this ${modality}.`,
      query_modality: modality,
      inference_mode: useVision ? "local_vision" : "local",
      top_k: topK,
      rerank_top_k: rerankTopK,
    })
  }

  const hasResult = Boolean(active.data)

  function reset() {
    active.reset()
  }

  return (
    <div className={cn("mx-auto space-y-8 px-6 py-10", hasResult ? "max-w-5xl" : "max-w-3xl")}>
      {hasResult && (
        <button
          onClick={reset}
          className="flex items-center gap-1.5 text-sm font-medium text-base-400 hover:text-white"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to search
        </button>
      )}

      {!hasResult && !active.isPending && (
        <div className="animate-rise-in pt-8 text-center">
          <div className="glow-brand mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-500">
            <Sparkles className="h-7 w-7 text-white" />
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-white md:text-[42px]">
            What do you want to <span className="gradient-text">uncover</span>?
          </h1>
          <p className="mx-auto mt-3 max-w-lg text-[15px] text-base-400">
            Ask a question, upload a file, or search across your indexed
            knowledge base. Evidence AI finds answers with sources, not just opinions.
          </p>

          <div className="mx-auto mt-7 flex max-w-2xl flex-wrap justify-center gap-2.5">
            {EXAMPLE_PROMPTS.map((p) => (
              <button
                key={p}
                onClick={() => setQuestion(p)}
                className="rounded-full border border-white/[0.1] bg-white/[0.03] px-4 py-2 text-sm text-base-300 transition-colors duration-150 hover:border-violet-400/40 hover:bg-violet-500/[0.06] hover:text-white"
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search composer -------------------------------------------------- */}
      {!hasResult && (
        <div className="animate-rise-in mx-auto max-w-2xl space-y-4">
          <Card className="gradient-ring !rounded-2xl p-3">
            <Textarea
              rows={modality === "text" ? 3 : 2}
              placeholder={
                modality === "text"
                  ? "Ask a question about your documents, images, or audio…"
                  : "Optional — leave blank to just find related material."
              }
              value={question}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuestion(e.target.value)}
              className="!border-0 !bg-transparent !text-base !ring-0 focus-visible:!ring-0"
            />
            <div className="flex items-center justify-between gap-3 px-1.5 pb-1 pt-1">
              <div className="flex flex-wrap items-center gap-2">
                {MODES.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => { setModality(m.id); setFile(null) }}
                    className={cn(
                      "rounded-[8px] border px-3 py-1.5 text-xs font-medium transition-all duration-150",
                      modality === m.id
                        ? "border-violet-400/40 bg-violet-500/[0.14] text-violet-200"
                        : "border-white/[0.08] bg-white/[0.02] text-base-400 hover:border-white/[0.16] hover:text-base-100",
                    )}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              <div className="flex shrink-0 items-center gap-3">
                <button
                  onClick={() => setShowTuning((v) => !v)}
                  className="text-xs font-medium text-base-400 hover:text-base-200"
                >
                  {showTuning ? "Hide tuning" : "Tuning"}
                </button>
                <Button onClick={handleSearch} disabled={!canSearch || active.isPending} size="md">
                  {active.isPending ? "Thinking…" : "Ask Evidence"}
                  {!active.isPending && <ArrowRight className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </Card>

          {modality !== "text" && (
            <div
              {...dropzone.getRootProps()}
              className={cn(
                "flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-white/[0.1] bg-white/[0.02] px-4 py-8 text-center transition-colors duration-150",
                dropzone.isDragActive && "border-violet-400/60 bg-violet-500/[0.05]",
              )}
            >
              <input {...dropzone.getInputProps()} />
              <Upload className="h-5 w-5 text-base-400" />
              {file ? (
                <p className="text-sm text-white">{file.name}</p>
              ) : (
                <p className="text-sm text-base-400">
                  Drop a {modality} file here, or click to choose one
                  <br />
                  <span className="text-[11px]">Used as the query itself — not added to the index.</span>
                </p>
              )}
            </div>
          )}

          {showTuning && (
            <Card className="grid grid-cols-3 gap-4 !rounded-xl p-4">
              <Slider label="Retrieve (top_k)" min={1} max={50} value={topK} onChange={setTopK} />
              <Slider label="Rerank to" min={1} max={20} value={rerankTopK} onChange={setRerankTopK} />
              <label className="flex items-center gap-2 self-end text-xs text-base-400">
                <input
                  type="checkbox"
                  checked={useVision}
                  onChange={(e) => setUseVision(e.target.checked)}
                  className="accent-violet-500"
                />
                Let Qwen inspect top image
              </label>
            </Card>
          )}
        </div>
      )}

      {!backendReady && (
        <p className="text-center text-xs text-amber-400">
          Inference backend isn't ready — check the sidebar.
        </p>
      )}

      {active.isError && (
        <p className="rounded-xl border border-rose-500/25 bg-rose-500/[0.06] px-4 py-3 text-sm text-rose-300">
          {(active.error as Error).message}
        </p>
      )}

      {active.data && <ResponsePanel response={active.data} sources={sourcesData?.sources ?? []} />}
    </div>
  )
}
