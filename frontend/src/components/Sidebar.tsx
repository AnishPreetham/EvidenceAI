import { useState } from "react"
import { Cpu, Database, FileStack, Layers, MessageSquareText, ScanText, Settings2, Sparkles, UploadCloud } from "lucide-react"
import { Badge } from "@/components/ui/Badge"
import { Toggle } from "@/components/ui/Toggle"
import type { Backend } from "@/lib/types"
import { useHealth } from "@/hooks/useEvidenceAi"
import { cn } from "@/lib/utils"

const NAV = [
  { id: "ask", label: "Ask Evidence", icon: MessageSquareText },
  { id: "ingest", label: "Ingestion Studio", icon: UploadCloud },
  { id: "sources", label: "Knowledge Library", icon: FileStack },
]

export function Sidebar({
  backend,
  onBackendChange,
  tab,
  onTabChange,
}: {
  backend: Backend
  onBackendChange: (b: Backend) => void
  tab: string
  onTabChange: (t: string) => void
}) {
  const { data: health, isLoading, isError } = useHealth(backend)
  const [showDebugOverride, setShowDebugOverride] = useState(false)

  // The provider that is actually serving requests right now.
  const activeProvider =
    backend === "auto"
      ? (health?.inference.active_provider ?? null)
      : backend

  return (
    <aside className="flex w-72 shrink-0 flex-col gap-7 border-r border-white/[0.06] bg-base-900/60 p-5">
      <div className="flex items-center gap-2.5 px-1 pt-1">
        <div className="glow-brand flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-cyan-500">
          <Sparkles className="h-4.5 w-4.5 text-white" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-[15px] font-bold leading-tight tracking-tight text-white">
            Evidence AI
          </div>
          <p className="text-[10.5px] uppercase tracking-wider text-base-400">Multimodal Intelligence</p>
        </div>
      </div>

      {/* Nav ------------------------------------------------------------ */}
      <nav className="space-y-1">
        {NAV.map((n) => {
          const Icon = n.icon
          const active = tab === n.id
          return (
            <button
              key={n.id}
              onClick={() => onTabChange(n.id)}
              className={cn(
                "relative flex w-full items-center gap-2.5 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all duration-150",
                active
                  ? "bg-violet-500/[0.12] text-white"
                  : "text-base-400 hover:bg-white/[0.04] hover:text-base-100",
              )}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-4 w-[3px] -translate-y-1/2 rounded-full bg-gradient-to-b from-violet-400 to-cyan-400" />
              )}
              <Icon className={cn("h-4 w-4", active ? "text-violet-300" : "text-base-400")} />
              {n.label}
            </button>
          )
        })}
      </nav>

      <div className="h-px bg-white/[0.06]" />

      {/* Inference mode ------------------------------------------------ */}
      <section>
        <div className="mb-2.5 flex items-center justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-base-400">
            Provider
          </h2>
          <button
            onClick={() => setShowDebugOverride((v) => !v)}
            title="Debug: override provider"
            className="text-base-600 hover:text-base-400"
          >
            <Settings2 className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Auto status display */}
        {backend === "auto" && (
          <div className="space-y-2">
            {isLoading && <Badge tone="neutral">checking…</Badge>}
            {isError && <Badge tone="rose">unreachable</Badge>}
            {activeProvider === "online" && (
              <div>
                <Badge tone="violet">● Online</Badge>
                <p className="mt-1.5 text-[11px] leading-relaxed text-base-400">
                  {health?.inference.fallback
                    ? `Using online — offline unavailable`
                    : `Online provider available`}
                </p>
              </div>
            )}
            {activeProvider === "offline" && (
              <div>
                <Badge tone="emerald">● Offline</Badge>
                <p className="mt-1.5 text-[11px] leading-relaxed text-base-400">
                  {health?.inference.fallback
                    ? "Using offline — online unavailable"
                    : "Offline AI ready"}
                </p>
              </div>
            )}
            {health && !health.inference.ready && (
              <div>
                <Badge tone="rose">● No provider</Badge>
                <p className="mt-1.5 text-[11px] leading-relaxed text-base-400">
                  {health.inference.message}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Non-auto: show direct readiness */}
        {backend !== "auto" && (
          <div>
            {isLoading && <Badge tone="neutral">checking…</Badge>}
            {isError && <Badge tone="rose">unreachable</Badge>}
            {health && (
              <Badge tone={health.inference.ready ? "emerald" : "amber"}>
                {health.inference.ready ? `● ${backend}` : "● not ready"}
              </Badge>
            )}
            {health && !health.inference.ready && (
              <p className="mt-2 text-[11px] leading-relaxed text-base-400">
                {health.inference.message}
              </p>
            )}
          </div>
        )}

        {/* Debug override -------------------------------------------- */}
        {showDebugOverride && (
          <div className="mt-3 rounded-[10px] border border-white/[0.07] bg-black/20 p-2.5">
            <p className="mb-2 text-[10px] uppercase tracking-wider text-base-500">
              Override
            </p>
            <Toggle
              options={[
                { id: "auto", label: "Auto" },
                { id: "offline", label: "Offline" },
                { id: "online", label: "Online" },
              ]}
              value={backend}
              onChange={(id) => onBackendChange(id as Backend)}
              className="w-full [&>button]:flex-1"
            />
            <p className="mt-2 text-[10px] leading-relaxed text-base-500">
              Auto selects online when available, offline otherwise.
            </p>
          </div>
        )}
      </section>

      {/* Runtime -------------------------------------------------------- */}
      <section>
        <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-base-400">
          Runtime
        </h2>
        <div className="space-y-1.5">
          <RuntimeRow icon={<ScanText className="h-3.5 w-3.5" />} label="Tesseract OCR" ok={health?.tesseract.available} />
          <RuntimeRow icon={<FileStack className="h-3.5 w-3.5" />} label="LibreOffice (.doc)" ok={health?.libreoffice.available} />
        </div>
      </section>

      {/* Index ------------------------------------------------------------ */}
      <section>
        <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-base-400">
          Index
        </h2>
        <div className="grid grid-cols-2 gap-2">
          <Stat icon={<FileStack className="h-3.5 w-3.5" />} label="Sources" value={health?.index.sources} />
          <Stat icon={<Layers className="h-3.5 w-3.5" />} label="Items" value={health?.index.items} />
          <Stat icon={<Cpu className="h-3.5 w-3.5" />} label="Vectors" value={health?.index.vectors} />
          <Stat icon={<Database className="h-3.5 w-3.5" />} label="Links" value={health?.index.relationships} />
        </div>
      </section>

      {/* Models ------------------------------------------------------------ */}
      {health?.models && (
        <section>
          <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-base-400">
            Models
          </h2>
          <ul className="space-y-1.5 rounded-[10px] border border-white/[0.06] bg-black/20 p-2.5 font-mono text-[11px] text-base-400">
            {Object.entries(health.models).map(([label, value]) => (
              <li key={label} className="flex justify-between gap-2">
                <span>{label}</span>
                <span className="truncate text-base-100">{value}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className="mt-auto text-[11px] leading-relaxed text-base-400">
        Ingestion — OCR, image understanding, transcription, embeddings,
        retrieval — always runs locally. Only answer generation uses the
        cloud in Online mode.
      </p>
    </aside>
  )
}

function RuntimeRow({ icon, label, ok }: { icon: React.ReactNode; label: string; ok: boolean | undefined }) {
  return (
    <div className="flex items-center gap-2.5 rounded-[8px] border border-white/[0.06] bg-black/20 px-2.5 py-1.5 text-xs text-base-200">
      <span className="text-base-400">{icon}</span>
      <span className="flex-1">{label}</span>
      <span className={cn(
        "h-1.5 w-1.5 rounded-full",
        ok === undefined ? "bg-base-600" : ok ? "bg-emerald-400 shadow-[0_0_6px] shadow-emerald-400/60" : "bg-rose-400",
      )} />
    </div>
  )
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: number | undefined }) {
  return (
    <div className="rounded-[10px] border border-white/[0.06] bg-black/20 px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-base-400">
        {icon}
        <span className="text-[11px]">{label}</span>
      </div>
      <div className="mt-0.5 font-mono text-xl font-semibold text-white">
        {value ?? "–"}
      </div>
    </div>
  )
}
