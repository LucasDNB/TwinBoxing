import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/* Rotulo mono, mayusculas, tracking 0,14em (03 §3). */
export function Rotulo({ children, className, as: Tag = "span" }: { children: ReactNode; className?: string; as?: "span" | "p" | "h2" | "h3" | "div" }) {
  return <Tag className={cn("rotulo", className)}>{children}</Tag>;
}
