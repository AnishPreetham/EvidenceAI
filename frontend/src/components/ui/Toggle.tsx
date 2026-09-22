import { cn } from "@/lib/utils"

export function Toggle({
  options, value, onChange, className,
}: { options: { id: string; label: string }[]; value: string; onChange: (id: string) => void; className?: string }) {
  return (
    <div className={cn("inline-flex gap-1 rounded-[10px] border border-white/[0.07] bg-base-900/60 p-1", className)}>
      {options.map((opt) => (
        <button
          key={opt.id}
          onClick={() => onChange(opt.id)}
          className={cn(
            "rounded-[8px] px-3 py-1.5 text-xs font-semibold transition-all duration-150",
            value === opt.id
              ? "bg-gradient-to-r from-violet-500 to-blue-500 text-white shadow-[0_2px_12px_rgba(124,92,255,0.3)]"
              : "text-base-400 hover:text-base-100",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
