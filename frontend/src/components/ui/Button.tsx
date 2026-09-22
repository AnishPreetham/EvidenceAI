import { type ButtonHTMLAttributes, forwardRef } from "react"
import { cn } from "@/lib/utils"

type Variant = "primary" | "secondary" | "ghost" | "outline"
type Size = "sm" | "md" | "lg"

const variants: Record<Variant, string> = {
  primary:
    "bg-gradient-to-r from-violet-500 to-blue-500 text-white shadow-[0_4px_20px_rgba(124,92,255,0.28)] hover:shadow-[0_6px_28px_rgba(124,92,255,0.42)] hover:brightness-110 disabled:opacity-40 disabled:shadow-none",
  secondary:
    "bg-base-800 text-base-100 border border-white/[0.06] hover:bg-base-700 hover:border-violet-400/25 disabled:opacity-40",
  ghost:
    "bg-transparent text-base-200 hover:bg-white/[0.04] disabled:opacity-40",
  outline:
    "bg-transparent border border-base-600 text-base-100 hover:border-violet-400/50 hover:bg-violet-500/[0.06] disabled:opacity-40",
}

const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-sm rounded-[8px]",
  md: "h-10 px-5 text-sm rounded-[10px]",
  lg: "h-12 px-6 text-base rounded-[10px]",
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 font-semibold transition-all duration-150",
        "disabled:cursor-not-allowed active:scale-[0.98]",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  ),
)
Button.displayName = "Button"
