import { cn } from "@/lib/utils"

export function Slider({
  label, value, min, max, onChange, className,
}: { label: string; value: number; min: number; max: number; onChange: (v: number) => void; className?: string }) {
  return (
    <label className={cn("flex flex-col gap-2", className)}>
      <span className="flex items-center justify-between text-xs font-medium text-base-400">
        <span>{label}</span>
        <span className="font-mono text-violet-300">{value}</span>
      </span>
      <input
        type="range" min={min} max={max} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/[0.08] accent-violet-500"
      />
    </label>
  )
}
