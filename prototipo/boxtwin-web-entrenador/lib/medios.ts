"use client";

import { useSyncExternalStore } from "react";

/* Si el ancho es de escritorio (lg, >= 1024 px). En el prerender se asume celular. */
export function useEsEscritorio(): boolean {
  return useSyncExternalStore(
    (f) => {
      const m = window.matchMedia("(min-width: 1024px)");
      m.addEventListener("change", f);
      return () => m.removeEventListener("change", f);
    },
    () => window.matchMedia("(min-width: 1024px)").matches,
    () => false,
  );
}
