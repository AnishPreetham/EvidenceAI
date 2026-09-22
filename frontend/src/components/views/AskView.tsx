import { useState, useRef, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import {
  ArrowLeft, ArrowRight, Sparkles, Upload,
  X, RefreshCw, Image as ImageIcon, Mic, FileText,
} from "lucide-react"
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

type ModeConfig = {
  id: QueryModality
  label: string
  accept: Record<string, string[]>
  formatHint: string
  dropHint: string
  instructionPlaceholder: string
  emptyPlaceholder: string
  loadingLabel: string
  icon: React.ElementType
}

const MODES: ModeConfig[] = [
  {
    id: "text",
    label: "Text",
    accept: {},
    formatHint: "",
    dropHint: "",
    instructionPlaceholder: "Ask a question about your documents, images, or audio…",
    emptyPlaceholder: "Ask a question about your documents, images, or audio…",
    loadingLabel: "Searching evidence…",
    icon: FileText,
  },
  {
    id: "image",
    label: "Image",
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"] },
    formatHint: "PNG · JPG · JPEG · WEBP · BMP · TIFF",
    dropHint: "Drop an image here",
    instructionPlaceholder: "What would you like to know about this image? (optional)",
    emptyPlaceholder: "What would you like to know about this image?",
    loadingLabel: "Analyzing image…",
    icon: ImageIcon,
  },
  {
    id: "audio",
    label: "Audio",
    accept: { "audio/*": [".wav", ".mp3", ".m4a", ".flac", ".ogg"] },
    formatHint: "WAV · MP3 · M4A · FLAC · OGG",
    dropHint: "Drop an audio file here",
    instructionPlaceholder: "What should I find in this recording? (optional)",
    emptyPlaceholder: "What should I find in this recording?",
    loadingLabel: "Transcribing & searching…",
    icon: Mic,
  },
  {
    id: "document",
    label: "Document",
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "application/msword": [".doc"],
      "text/plain": [".txt"],
      "text/csv": [".csv"],
      "text/markdown": [".md"],
    },
    formatHint: "PDF · DOCX · DOC · TXT · CSV · MD",
    dropHint: "Drop a document here",
    instructionPlaceholder: "What should I compare or find in this document? (optional)",
    emptyPlaceholder: "What should I compare or find in this document?",
    loadingLabel: "Scanning document…",
    icon: FileText,
  },
]

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function friendlyError(msg: string): string {
  if (msg.includes("authenticate") || msg.includes("401") || msg.includes("403"))
    return "Online AI could not authenticate — check the provider status in the sidebar."
  if (msg.includes("OLLAMA_NOT_AVAILABLE") || msg.includes("No AI provider"))
    return "No AI provider is currently available. Check the sidebar for provider status."
  if (msg.includes("Could not reach") || msg.includes("ECONNREFUSED"))
    return "Evidence AI couldn't complete the request. Is the backend server running?"
  if (msg.includes("UNSUPPORTED_FILE_TYPE"))
    return "This file type isn't supported. Try a different format."
  if (msg.includes("FILE_TOO_LARGE"))
    return "This file is too large for the current configuration."
  if (msg.includes("No file") || msg.includes("Choose"))
    return "Choose a file before asking Evidence AI."
  return msg
}

export function AskView({ backend, backendReady }: { backend: Backend; backendReady: boolean }) {
  const [modality, setModality] = useState<QueryModality>("text")
  const [question, setQuestion] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [topK, setTopK] = useState(10)
  const [rerankTopK, setRerankTopK] = useState(5)
  const [useVision, setUseVision] = useState(false)
  const [showTuning, setShowTuning] = useState(false)
  const replaceInputRef = useRef<HTMLInputElement>(null)

  const { data: sourcesData } = useSources()
  const textQuery = useRunQuery(backend)
  const fileQuery = useRunQueryFile(backend)
  const active = modality === "text" ? textQuery : fileQuery
  const mode = MODES.find((m) => m.id === modality)!

  // Build/revoke image preview URL
  useEffect(() => {
    if (file && modality === "image") {
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      return () => URL.revokeObjectURL(url)
    }
    setPreviewUrl(null)
  }, [file, modality])

  const dropzone = useDropzone({
    multiple: false,
    accept: modality !== "text" ? mode.accept : undefined,
    disabled: modality === "text",
    onDrop: (accepted: File[], rejected: unknown[]) => {
      if (rejected.length > 0) {
        setFileError(`File type not supported. Expected: ${mode.formatHint}`)
        return
      }
      setFileError(null)
      setFile(accepted[0] ?? null)
    },
  })

  function switchModality(id: QueryModality) {
    setModality(id)
    setFile(null)
    setFileError(null)
    setPreviewUrl(null)
  }

  function handleReplaceChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setFileError(null)
    e.target.value = ""
  }

  const canSearch = backendReady && (
    modality === "text" ? question.trim().length > 0 : file != null
  )

  function handleSearch() {
    if (!canSearch || active.isPending) return
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
      instruction: question.trim() || undefined,
      query_modality: modality,
      inference_mode: useVision ? "local_vision" : "local",
      top_k: topK,
      rerank_top_k: rerankTopK,
    })
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && modality === "text") {
      e.preventDefault()
      handleSearch()
    }
  }

  const hasResult = Boolean(active.data)

  function reset() {
    active.reset()
    setFile(null)
    setFileError(null)
    setPreviewUrl(null)
  }

  // Shared bottom bar shared across all composer states
  const bottomBar = (
    <div className="flex items-center justify-between gap-3 px-1.5 pb-1 pt-1">
      <div className="flex flex-wrap items-center gap-2">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => switchModality(m.id)}
            className={cn(
              "rounded-[8px] border px-3 py-1.5 text-xs font-medium transition-all duration-150",
              modality === m.id
                ? "border-violet-400/40 bg-violet-500/[0.14] text-violet-200"
                : "border-white/[0.08] bg-white/[0.02] text-base-400 hover:border-white/[0.16] hover:text-base-100",
            )}
            aria-pressed={modality === m.id}
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
        <Button
          onClick={handleSearch}
          disabled={!canSearch || active.isPending}
          size="md"
          aria-label={active.isPending ? mode.loadingLabel : "Ask Evidence"}
        >
          {active.isPending ? mode.loadingLabel : "Ask Evidence"}
          {!active.isPending && <ArrowRight className="h-4 w-4" />}
        </Button>
      </div>
    </div>
  )

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

      {/* Hero (text mode, idle) -------------------------------------------- */}
      {!hasResult && !active.isPending && modality === "text" && (
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

      {/* Composer ------------------------------------------------------------ */}
      {!hasResult && (
        <div className="animate-rise-in mx-auto max-w-2xl space-y-4">

          {modality === "text" ? (
            /* ─── TEXT ──────────────────────────────────────────────── */
            <Card className="gradient-ring !rounded-2xl p-3">
              <Textarea
                rows={3}
                placeholder={mode.emptyPlaceholder}
                value={question}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                className="!border-0 !bg-transparent !text-base !ring-0 focus-visible:!ring-0"
                aria-label="Question"
              />
              {bottomBar}
            </Card>
          ) : file ? (
            /* ─── FILE SELECTED ─────────────────────────────────────── */
            <Card className="gradient-ring !rounded-2xl p-3 space-y-3">
              {/* Hidden replace input */}
              <input
                ref={replaceInputRef}
                type="file"
                className="sr-only"
                accept={Object.values(mode.accept).flat().join(",")}
                onChange={handleReplaceChange}
                aria-hidden="true"
              />
              <FilePreview
                file={file}
                modality={modality}
                previewUrl={previewUrl}
                onRemove={() => { setFile(null); setPreviewUrl(null) }}
                onReplace={() => replaceInputRef.current?.click()}
              />
              <Textarea
                rows={2}
                placeholder={mode.instructionPlaceholder}
                value={question}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuestion(e.target.value)}
                className="!border-0 !bg-transparent !text-sm !ring-0 focus-visible:!ring-0"
                aria-label="Instruction (optional)"
              />
              {bottomBar}
            </Card>
          ) : (
            /* ─── EMPTY DROP ZONE ────────────────────────────────────── */
            <Card className="gradient-ring !rounded-2xl overflow-hidden">
              <div
                {...dropzone.getRootProps()}
                className={cn(
                  "flex cursor-pointer flex-col items-center gap-3 px-6 py-14 text-center transition-all duration-200",
                  dropzone.isDragActive
                    ? "bg-violet-500/[0.08]"
                    : "hover:bg-white/[0.02]",
                )}
                role="button"
                aria-label={`Upload ${modality} file — ${mode.formatHint}`}
              >
                <input {...dropzone.getInputProps()} />
                <div
                  className={cn(
                    "flex h-12 w-12 items-center justify-center rounded-xl border transition-colors duration-200",
                    dropzone.isDragActive
                      ? "border-violet-400/40 bg-violet-500/20"
                      : "border-white/[0.08] bg-white/[0.05]",
                  )}
                >
                  <Upload
                    className={cn(
                      "h-5 w-5 transition-colors duration-200",
                      dropzone.isDragActive ? "text-violet-300" : "text-base-400",
                    )}
                  />
                </div>
                <div>
                  <p
                    className={cn(
                      "text-sm font-medium transition-colors duration-200",
                      dropzone.isDragActive ? "text-violet-200" : "text-base-200",
                    )}
                  >
                    {dropzone.isDragActive ? "Release to upload" : mode.dropHint}
                  </p>
                  <p className="mt-1 text-xs text-base-400">
                    or click to browse · <span className="text-base-500">{mode.formatHint}</span>
                  </p>
                  <p className="mt-3 text-[11px] text-base-500">
                    Used as the query — not added to your knowledge base
                  </p>
                </div>
              </div>

              {fileError && (
                <p className="border-t border-rose-500/15 bg-rose-500/[0.04] px-4 py-2.5 text-xs text-rose-400">
                  {fileError}
                </p>
              )}

              <div className="border-t border-white/[0.06] px-3 py-2.5">
                {bottomBar}
              </div>
            </Card>
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
          {friendlyError((active.error as Error).message)}
        </p>
      )}

      {active.data && (
        <ResponsePanel response={active.data} sources={sourcesData?.sources ?? []} />
      )}
    </div>
  )
}

function FilePreview({
  file,
  modality,
  previewUrl,
  onRemove,
  onReplace,
}: {
  file: File
  modality: QueryModality
  previewUrl: string | null
  onRemove: () => void
  onReplace: () => void
}) {
  const Icon = modality === "audio" ? Mic : FileText

  return (
    <div className="flex items-start gap-3 rounded-[10px] border border-white/[0.08] bg-black/20 p-3">
      {/* Preview or icon */}
      {modality === "image" && previewUrl ? (
        <img
          src={previewUrl}
          alt="Preview"
          className="h-14 w-14 shrink-0 rounded-[8px] border border-white/[0.08] object-cover"
        />
      ) : (
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-[8px] border border-white/[0.08] bg-white/[0.04]">
          <Icon className="h-6 w-6 text-base-400" />
        </div>
      )}

      {/* Filename + size */}
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-base-100">{file.name}</p>
        <p className="mt-0.5 text-xs text-base-400">{formatFileSize(file.size)}</p>
      </div>

      {/* Actions */}
      <div className="flex shrink-0 gap-1.5">
        <button
          onClick={onReplace}
          title="Replace file"
          className="flex items-center gap-1 rounded-[6px] border border-white/[0.08] px-2 py-1 text-[11px] font-medium text-base-400 transition-colors hover:border-white/[0.18] hover:text-base-100"
        >
          <RefreshCw className="h-3 w-3" />
          Replace
        </button>
        <button
          onClick={onRemove}
          title="Remove file"
          className="flex items-center gap-1 rounded-[6px] border border-white/[0.08] px-2 py-1 text-[11px] font-medium text-base-400 transition-colors hover:border-rose-500/40 hover:text-rose-300"
        >
          <X className="h-3 w-3" />
          Remove
        </button>
      </div>
    </div>
  )
}
