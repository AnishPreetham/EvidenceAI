import { type TextareaHTMLAttributes, forwardRef } from "react"
import { cn } from "@/lib/utils"

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "w-full resize-none rounded-[12px] border border-white/[0.08] bg-base-900/60 px-4 py-3 text-[15px] text-base-100 transition-colors duration-150",
        "placeholder:text-base-400 focus-visible:border-violet-400/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/15",
        className,
      )}
      {...props}
    />
  ),
)
Textarea.displayName = "Textarea"
