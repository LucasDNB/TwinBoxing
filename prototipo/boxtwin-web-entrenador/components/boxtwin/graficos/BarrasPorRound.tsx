"use client";

/*
 * Golpes por round, A contra B, en SVG propio (03 §5). Lleva role="img" con la lectura en una
 * frase y la tabla con los mismos numeros al lado (plegada en el celular). Solo anima la
 * entrada de las barras, nunca la escala ni los ejes, y nada con menos movimiento pedido.
 */
import { motion, useReducedMotion } from "motion/react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/animate-ui/components/radix/accordion";
import { lados as ladosDe, lecturaRounds, prefijoRound, type FightCard, type Lado } from "@/lib/fightcard";

const ALTO = 140;

export function BarrasPorRound({ fc, nombres }: { fc: FightCard; nombres: Record<Lado, string> }) {
  const quieto = useReducedMotion();
  const lados = ladosDe(fc);
  const rondas = fc.video.rounds;
  const valor = (l: Lado, r: number) => fc.peleadores[l]?.por_round.find((x) => x.round === r)?.total ?? 0;
  const max = Math.max(1, ...rondas.flatMap((r) => lados.map((l) => valor(l, r.round))));
  const lectura = lados.map((l) => lecturaRounds(fc.peleadores[l]!.por_round, nombres[l])).filter(Boolean).join(". ");
  const anchoGrupo = 60;
  const ancho = rondas.length * anchoGrupo + 20;

  return (
    <div>
      <svg role="img" aria-label={lectura || "Golpes detectados por round"} viewBox={`0 0 ${ancho} ${ALTO + 24}`} className="h-auto w-full max-w-xl">
        <line x1="10" x2={ancho - 10} y1={ALTO} y2={ALTO} className="stroke-borde-fuerte" strokeWidth="1" />
        {rondas.map((r, i) =>
          lados.map((l, j) => {
            const v = valor(l, r.round);
            const h = (v / max) * (ALTO - 16);
            const x = 20 + i * anchoGrupo + j * 20;
            return (
              <g key={`${r.round}-${l}`}>
                <motion.rect
                  x={x}
                  width={16}
                  className={l === "A" ? "fill-a-trazo" : "fill-b-trazo"}
                  initial={quieto ? false : { y: ALTO, height: 0 }}
                  animate={{ y: ALTO - h, height: h }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  rx={l === "B" ? 8 : 2}
                />
                <text x={x + 8} y={ALTO - h - 4} textAnchor="middle" className="fill-texto-2 font-mono text-[10px]">{v}</text>
              </g>
            );
          }),
        )}
        {rondas.map((r, i) => (
          <text key={r.round} x={20 + i * anchoGrupo + (lados.length * 20 - 4) / 2} y={ALTO + 16} textAnchor="middle" className="fill-tenue font-mono text-[11px]">{prefijoRound(fc)}{r.round}</text>
        ))}
      </svg>
      <Accordion type="single" collapsible className="lg:hidden">
        <AccordionItem value="tabla" className="border-0">
          <AccordionTrigger className="min-h-11 py-2 text-sm text-texto-2">Ver la tabla</AccordionTrigger>
          <AccordionContent><Tabla fc={fc} lados={lados} nombres={nombres} valor={valor} /></AccordionContent>
        </AccordionItem>
      </Accordion>
      <div className="hidden lg:block"><Tabla fc={fc} lados={lados} nombres={nombres} valor={valor} /></div>
    </div>
  );
}

function Tabla({ fc, lados, nombres, valor }: { fc: FightCard; lados: Lado[]; nombres: Record<Lado, string>; valor: (l: Lado, r: number) => number }) {
  return (
    <table className="mt-2 w-full text-sm">
      <caption className="sr-only">Golpes detectados por round</caption>
      <thead>
        <tr><th scope="col" className="rotulo py-1 text-left font-normal">Round</th>{lados.map((l) => <th key={l} scope="col" className="rotulo py-1 text-right font-normal">{nombres[l]}</th>)}</tr>
      </thead>
      <tbody>
        {fc.video.rounds.map((r) => (
          <tr key={r.round} className="border-t border-borde-suave">
            <th scope="row" className="py-1 text-left font-normal">{prefijoRound(fc)}{r.round}</th>
            {lados.map((l) => <td key={l} className="dato py-1 text-right">{valor(l, r.round)}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
