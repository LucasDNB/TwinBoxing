/*
 * Agregados de pantalla sobre la Fight-Card (02 §3).
 *
 * POR QUE EXISTE
 *   Todo lo que la pantalla calcula sale de aca y de ningun componente, por dos motivos. El
 *   primero es CA-05.2: cambiar de round no le pide nada a la API, todo sale del documento
 *   cargado. El segundo es que cada numero tiene una regla del spec atras -los descartados
 *   no cambian los detectados (B4), las combinaciones salen del documento y no se recalculan
 *   aca (B10)- y esas reglas tienen que tener prueba.
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

/* "R" con rounds declarados, "S" con segmentos de 3 minutos (cambios-14-10, DC-5). */
export function prefijoRound(fc: FightCard): "R" | "S" {
  return fc.video.rounds_origen === "segmentos_180s" ? "S" : "R";
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

// -- Altura del golpe (ampliacion 03, B11 y W14) ---------------------------------------
//
// La cuenta el worker (etapa enriquecer) y la recuenta al aplicar correcciones, asi que aca
// solo se lee: cabeza + cuerpo + sin dato = detectados (CA-14.1).

export interface Altura {
  cabeza: number;
  cuerpo: number;
  sinDato: number;
}

/* La altura de un peleador en la sesion o en un round; null si el documento no la trae. */
export function alturaDe(fc: FightCard, lado: Lado, round: number | null = null): Altura | null {
  const p = fc.peleadores[lado];
  if (!p) return null;
  const c = round == null ? p.total : p.por_round.find((r) => r.round === round);
  if (!c || c.cabeza == null || c.cuerpo == null) return null;
  return { cabeza: c.cabeza, cuerpo: c.cuerpo, sinDato: c.objetivo_sin_dato ?? c.total - c.cabeza - c.cuerpo };
}

// -- Combinaciones (ampliacion 01, B10 y W13; regla del 14-10) ---------------------------
//
// Las cuenta el worker con la regla del anotador y vienen en el documento (0.3). Aca solo se
// leen: tener la regla en los dos lados seria tenerla mal en uno. Una 0.1 o 0.2 no las trae y
// la pantalla lo dice, en vez de calcularlas.

/* Cuantas combinaciones tiene un peleador en la sesion o en un round; null si el documento no las trae. */
export function combinacionesDe(fc: FightCard, lado: Lado, round: number | null = null): number | null {
  const c = fc.peleadores[lado]?.combinaciones;
  if (!c) return null;
  if (round == null) return c.total;
  return c.por_round.find((r) => r.round === round)?.total ?? 0;
}

/* Los golpes que forman parte de una combinacion, para la insignia de la lista. */
export function golpesEnCombinacion(fc: FightCard, lado: Lado): Set<string> {
  return new Set((fc.peleadores[lado]?.combinaciones?.lista ?? []).flatMap((c) => c.golpes));
}

/*
 * La secuencia de manos de cada combinacion ("izq-der-izq"), por golpe, para la insignia. Una
 * Fight-Card contada con la regla del 13-10 no trae `brazos`: eran todas izq-der-izq.
 */
export function secuenciaPorGolpe(fc: FightCard, lado: Lado): Map<string, string> {
  const m = new Map<string, string>();
  for (const c of fc.peleadores[lado]?.combinaciones?.lista ?? []) {
    const sec = c.brazos ? c.brazos.map((b) => b ?? "?").join("-") : "izq-der-izq";
    for (const g of c.golpes) m.set(g, sec);
  }
  return m;
}

/* "izq-der-izq" a "izquierda, derecha, izquierda", para lectores de pantalla y titulos. */
export function secuenciaEnPalabras(brazos: (string | null)[] | undefined): string {
  const nombre: Record<string, string> = { izq: "izquierda", der: "derecha" };
  return (brazos ?? ["izq", "der", "izq"]).map((b) => (b ? nombre[b] ?? b : "sin mano")).join(", ");
}

/* El recall con que se califica el numero: el de combinaciones si C11 se midio, si no el de golpes. */
export function recallDeCombinaciones(fc: FightCard): number {
  return fc.detector.combinaciones?.recall_medido ?? fc.detector.recall_medido;
}
