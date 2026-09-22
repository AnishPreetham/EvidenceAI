import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

type Tone = "violet" | "emerald" | "amber" | "rose" | "neutral"

const tones: Record<Tone, string> = {
  violet: "bg-violet-500/12 text-violet-300 ring-1 ring-inset ring-violet-500/25",
  emerald: "bg-emerald-500/12 text-emerald-300 ring-1 ring-inset ring-emerald-500/25",
  amber: "bg-amber-500/12 text-amber-300 ring-1 ring-inset ring-amber-500/25",
  rose: "bg-rose-500/12 text-rose-300 ring-1 ring-inset ring-rose-500/25",
  neutral: "bg-white/[0.04] text-base-200 ring-1 ring-inset ring-white/[0.08]",
}

export function Badge({
  children, tone = "neutral", className,
}: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1 rounded-full px-2.5 py-1 font-mono text-[11px] font-medium leading-none",
      tones[tone], className,
    )}>
      {children}
    </span>
  )
}
