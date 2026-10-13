/*
 * Agregados de pantalla sobre la Fight-Card (02 §3).
 *
 * POR QUE EXISTE
 *   Todo lo que la pantalla calcula sale de aca y de ningun componente, por dos motivos. El
 *   primero es CA-05.2: cambiar de round no le pide nada a la API, todo sale del documento
 *   cargado. El segundo es que cada numero tiene una regla del spec atras -los descartados
 *   no cambian los detectados (B4), las combinaciones respetan la pausa de /producto (B7)- y
 *   esas reglas tienen que tener prueba.
 */
import type { FightCardCruda, Golpe, Peleador, Round, TipoGolpe } from "@/lib/api/esquemas";
import { TIPOS_GOLPE } from "@/lib/api/esquemas";
import { decimal, nombreCorto } from "@/lib/formato";

export type Lado = "A" | "B";

export interface FightCard extends Omit<FightCardCruda, "peleadores"> {
  peleadores: { A: Peleador & { descartados: number }; B?: Peleador & { descartados: number } };
  /* 0.1 no trae los campos de tramos: la pantalla de tramos lo dice (05 §4). */
  sinRevisionDeTramos: boolean;
}

/*
 * Completa una 0.1 con lo que la 0.2 agrega: `descartado: false`, `descartados: 0` y los
 * campos de tramos en nulo (05 §4). Una 0.2 pasa igual, con `descartados` recontado si
 * faltara.
 */
export function normalizarFightCard(fc: FightCardCruda): FightCard {
  const pel = (p: Peleador | undefined) => {
    if (!p) return undefined;
    const golpes = p.golpes.map((g) => ({ ...g, descartado: g.descartado ?? false }));
    return {
      ...p,
      golpes,
      guardia: p.guardia ?? [],
      descartados: p.descartados ?? golpes.filter((g) => g.descartado).length,
    };
  };
  const ident = { ...fc.identidad };
  for (const k of ["tramos_total", "tramos_asignados_auto", "tramos_asignados_manual"] as const) {
    if (ident[k] === undefined) ident[k] = null;
  }
  if (ident.separacion_color === undefined) {
    ident.separacion_color = fc.identidad.separacion_de_perfiles ?? null;
  }
  return {
    ...fc,
    identidad: ident,
    peleadores: { A: pel(fc.peleadores.A)!, B: pel(fc.peleadores.B) },
    sinRevisionDeTramos: fc.version === "0.1",
  };
}

export function lados(fc: FightCard): Lado[] {
  return fc.peleadores.B ? ["A", "B"] : ["A"];
}

/* El tipo que cuenta: el corregido si existe, si no el del modelo. */
export function tipoEfectivo(g: Golpe): string | null {
  return g.corregido?.tipo ?? g.tipo ?? null;
}

export function enRound(g: Golpe, r: { inicio_s: number; fin_s: number }): boolean {
  return r.inicio_s <= g.t_inicio && g.t_inicio < r.fin_s;
}

export interface Resumen {
  detectados: number;
  izq: number;
  der: number;
  porMinuto: number | null;
  descartados: number;
  tipos: Record<TipoGolpe, number>;
  sinClasificar: number;
}

/*
 * Los numeros de una fila de la Fight-Card, para la sesion o para un round.
 *
 * "Detectados" es lo que el sistema detecto y no cambia con "no fue un golpe": eso suma a
 * descartados (B4). La distribucion por tipo cuenta con el tipo corregido y deja afuera los
 * descartados, que no son golpes.
 */
export function resumen(fc: FightCard, lado: Lado, round: number | null = null): Resumen | null {
  const p = fc.peleadores[lado];
  if (!p) return null;
  let golpes = p.golpes;
  let detectados = p.total.total;
  let izq = p.total.izq;
  let der = p.total.der;
  let minutos = fc.video.duracion_s / 60;
  if (round != null) {
    const r = p.por_round.find((x) => x.round === round);
    if (!r) return null;
    golpes = golpes.filter((g) => enRound(g, r));
    detectados = r.total;
    izq = r.izq;
    der = r.der;
    minutos = (r.fin_s - r.inicio_s) / 60;
  }
  const tipos = Object.fromEntries(TIPOS_GOLPE.map((t) => [t, 0])) as Record<TipoGolpe, number>;
  let sinClasificar = 0;
  let descartados = 0;
  for (const g of golpes) {
    if (g.descartado) {
      descartados += 1;
      continue;
    }
    const t = tipoEfectivo(g);
    if (t && t in tipos) tipos[t as TipoGolpe] += 1;
    else sinClasificar += 1;
  }
  const porMinuto =
    round != null
      ? p.por_round.find((x) => x.round === round)!.por_minuto
      : minutos > 0
        ? Math.round((detectados / minutos) * 100) / 100
        : null;
  return { detectados, izq, der, porMinuto, descartados, tipos, sinClasificar };
}

/* La suma de los rounds tiene que dar el total del documento (CA-05.1). */
export function sumaDeRounds(p: Peleador): number {
  return p.por_round.reduce((a, r) => a + r.total, 0);
}

/*
 * La lectura en una frase del grafico por round (04 §05, punto 7). Solo se usa con C2
 * encendido: sin C2 la caida puede ser del detector y no del boxeador (D3).
 */
export function lecturaRounds(rondas: Round[], nombre: string): string | null {
  if (rondas.length < 2) return null;
  const a = rondas[rondas.length - 2];
  const b = rondas[rondas.length - 1];
  if (a.total === 0) return null;
  const v = (b.total - a.total) / a.total;
  const verbo = v < 0 ? "cae" : v > 0 ? "sube" : "se mantiene";
  const signo = v < 0 ? "−" : "+";
  return `De R${a.round} a R${b.round}, ${nombreCorto(nombre)} ${verbo} de ${a.total} a ${b.total}, ${signo}${decimal(Math.abs(v) * 100, 0)}%`;
}

/* Las guardias usan el brazo en ingles en su id (A-left-75); los golpes, en castellano. */
export function guardiaPorGolpe(p: Peleador): Map<string, NonNullable<Peleador["guardia"]>[number]> {
  const m = new Map();
  for (const e of p.guardia ?? []) {
    const id = e.golpe.replace("-left-", "-izq-").replace("-right-", "-der-");
    m.set(id, e);
  }
  return m;
}

// -- Combinaciones (04 §06, punto 5) ----------------------------------------------
//
// Las calcula la API (B7, boxtwin_api/perfil.py): aca solo se nombran. Tener la regla de
// la pausa en los dos lados seria tenerla mal en uno.

/* Numeracion de gimnasio: 1 jab, 2 recto de atras, 3 y 4 hook, 5 y 6 uppercut. */
export function numeroDeGimnasio(familia: string, lado: "adelante" | "atras" | null): number | null {
  if (!lado) return null;
  const base: Record<string, number> = { jab: 1, cross: 1, hook: 3, uppercut: 5 };
  if (!(familia in base)) return null;
  if (familia === "jab" || familia === "cross") return lado === "adelante" ? 1 : 2;
  return base[familia] + (lado === "adelante" ? 0 : 1);
}

export function nombreDeCombinacion(
  c: { secuencia: { familia: string; lado: "adelante" | "atras" | null }[] },
  nomenclatura: Record<string, string>,
): string {
  const numeros = c.secuencia.map((s) => numeroDeGimnasio(s.familia, s.lado));
  if (numeros.every((n) => n != null)) return numeros.join("-");
  return c.secuencia.map((s) => nomenclatura[s.familia] ?? s.familia).join(" + ");
}
