/*
 * Las lecturas del Rival Fantasma (04 §07, punto 4).
 *
 * POR QUE EXISTE
 *   Para que la comparacion entre dos perfiles diga algo sin inventar: cada lectura sale de
 *   una regla fija sobre una fila medida y cita la metrica de la que sale. Las reglas que
 *   usan rounds o guardia solo corren con C2 o C3 aprobados (CA-07.1). No hay pronostico ni
 *   probabilidad de victoria.
 *
 *   El texto de cada regla es criterio de dominio y lo valida Lucas antes de publicarlo
 *   (07 §3). Estos son los iniciales del spec.
 */
import { decimal, porcentajeAR } from "@/lib/formato";

export interface FilaRival {
  nombre: string;
  altura_m: number | null;
  envergadura_m: number | null;
  guardia: string | null;
  golpes_por_minuto: number | null;
  jab_fraccion: number | null;
  guardia_opuesta_arriba: number | null;
  caida_ultimo_round: number | null;
}

export interface Lectura {
  texto: string;
  metrica: string;
}

export interface Lecturas {
  A: { oportunidades: Lectura[]; amenazas: Lectura[] };
  B: { oportunidades: Lectura[]; amenazas: Lectura[] };
}

export const SIN_DIFERENCIAS = "Con los datos medidos no hay diferencias marcadas entre estos dos perfiles.";

export function lecturas(
  a: FilaRival,
  b: FilaRival,
  banderas: { comparacion_rounds: boolean; guardia: boolean },
): Lecturas {
  const r: Lecturas = { A: { oportunidades: [], amenazas: [] }, B: { oportunidades: [], amenazas: [] } };
  const lado = { A: a, B: b };

  // Envergadura: 3 cm o mas -> oportunidad para el de mayor envergadura.
  if (a.envergadura_m != null && b.envergadura_m != null) {
    const d = Math.round(Math.abs(a.envergadura_m - b.envergadura_m) * 100);
    if (d >= 3) {
      const mayor = a.envergadura_m > b.envergadura_m ? "A" : "B";
      r[mayor].oportunidades.push({
        texto: `Tiene ${d} cm más de envergadura: puede trabajar a distancia con el jab.`,
        metrica: "envergadura",
      });
    }
  }

  // Guardia: ortodoxo contra zurdo -> amenaza para los dos, como hecho de guardia.
  const g = [a.guardia, b.guardia];
  if (g.includes("ortodoxa") && g.includes("zurda")) {
    for (const l of ["A", "B"] as const) {
      r[l].amenazas.push({
        texto: "Guardias opuestas: la mano de atrás del rival entra por el lado abierto.",
        metrica: "guardia",
      });
    }
  }

  // Ritmo: 15% o mas de diferencia en golpes por minuto -> oportunidad para el de mayor ritmo.
  if (a.golpes_por_minuto && b.golpes_por_minuto) {
    const mayor = a.golpes_por_minuto >= b.golpes_por_minuto ? "A" : "B";
    const menor = mayor === "A" ? "B" : "A";
    const v = lado[mayor].golpes_por_minuto! / lado[menor].golpes_por_minuto! - 1;
    if (v >= 0.15) {
      r[mayor].oportunidades.push({
        texto: `Trabaja a más ritmo: ${decimal(lado[mayor].golpes_por_minuto, 1)} contra ${decimal(lado[menor].golpes_por_minuto, 1)} golpes detectados por minuto (${porcentajeAR(v)} más).`,
        metrica: "golpes por minuto",
      });
    }
  }

  // Caida en R3, solo con C2: el que cae 15% o mas -> amenaza para el, oportunidad para el otro.
  if (banderas.comparacion_rounds) {
    for (const l of ["A", "B"] as const) {
      const c = lado[l].caida_ultimo_round;
      if (c != null && c <= -0.15) {
        const otro = l === "A" ? "B" : "A";
        r[l].amenazas.push({ texto: `Su volumen cae ${porcentajeAR(-c)} en el último round.`, metrica: "caída en el último round" });
        r[otro].oportunidades.push({ texto: `El rival cae ${porcentajeAR(-c)} en el último round: el cierre es suyo.`, metrica: "caída en el último round" });
      }
    }
  }

  // Guardia opuesta, solo con C3: 10 puntos o mas -> amenaza para el mas bajo.
  if (banderas.guardia && a.guardia_opuesta_arriba != null && b.guardia_opuesta_arriba != null) {
    const d = Math.abs(a.guardia_opuesta_arriba - b.guardia_opuesta_arriba);
    if (d >= 0.1) {
      const bajo = a.guardia_opuesta_arriba < b.guardia_opuesta_arriba ? "A" : "B";
      r[bajo].amenazas.push({
        texto: `Deja la otra mano abajo más seguido: ${porcentajeAR(lado[bajo].guardia_opuesta_arriba)} de los golpes con la guardia arriba.`,
        metrica: "guardia opuesta arriba",
      });
    }
  }
  return r;
}

export function hayLecturas(l: Lecturas): boolean {
  return [l.A, l.B].some((x) => x.oportunidades.length + x.amenazas.length > 0);
}
