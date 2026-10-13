"use client";

/* Fase 2 de la identificacion: se completa en W9 (B3). */
import Link from "next/link";
import type { Sesion } from "@/lib/api/esquemas";
import type { FightCard } from "@/lib/fightcard";
import { Button } from "@/components/animate-ui/components/buttons/button";

export function FaseRevisar({ s }: { s: Sesion; fc: FightCard }) {
  return (
    <Button asChild className="min-h-11"><Link href={`/sesion/?id=${s.id}&vista=fightcard`}>Ver resultados</Link></Button>
  );
}
