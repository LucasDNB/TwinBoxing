"use client";

/*
 * Los hooks de TanStack Query: claves, sondeo y mutaciones (02 §5).
 *
 * Sondeo de una sesion: cada 3 s mientras esta en curso; se detiene en espera_siembra, listo
 * y fallo; se pausa con la pestana oculta. El panel sondea cada 10 s solo si alguna sesion
 * visible esta en curso.
 */
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ErrorApi, pedir, pedirValidado } from "@/lib/api/cliente";
import {
  Boxeador, FightCard, ListaBoxeadores, ListaSesiones, Panel, Perfil, Producto, Sesion,
  Token, Tramos, type EstadoSesion, type TipoCorreccion,
} from "@/lib/api/esquemas";
import { normalizarFightCard, type FightCard as FC } from "@/lib/fightcard";

export const claves = {
  producto: ["producto"] as const,
  panel: ["panel"] as const,
  sesiones: ["sesiones"] as const,
  sesion: (id: string) => ["sesion", id] as const,
  fightcard: (id: string) => ["fightcard", id] as const,
  boxeadores: ["boxeadores"] as const,
  perfil: (id: string, contexto: string) => ["perfil", id, contexto] as const,
  tramos: (id: string) => ["tramos", id] as const,
  ticket: (id: string) => ["ticket", id] as const,
  procesado: (id: string) => ["procesado", id] as const,
};

export const EN_CURSO: EstadoSesion[] = ["subida", "en_cola", "procesando", "completando"];
export const enCurso = (e: EstadoSesion | undefined) => !!e && EN_CURSO.includes(e);

export const SONDEO_SESION_MS = 3_000;
export const SONDEO_PANEL_MS = 10_000;

export function useProducto() {
  return useQuery({
    queryKey: claves.producto,
    queryFn: () => pedirValidado(Producto, "/producto"),
    staleTime: 60_000,
  });
}

export function useSesiones() {
  return useQuery({
    queryKey: claves.sesiones,
    queryFn: () => pedirValidado(ListaSesiones, "/jobs"),
    refetchInterval: (q) =>
      (q.state.data ?? []).some((s) => enCurso(s.estado)) ? SONDEO_PANEL_MS : false,
    refetchIntervalInBackground: false,
  });
}

export function usePanel(activo = true) {
  return useQuery({
    queryKey: claves.panel,
    queryFn: () => pedirValidado(Panel, "/panel"),
    enabled: activo,
    refetchInterval: (q) => {
      const d = q.state.data;
      return d && (d.sesiones_mes.en_curso > 0) ? SONDEO_PANEL_MS : false;
    },
    refetchIntervalInBackground: false,
  });
}

export function useSesion(id: string | null) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: claves.sesion(id ?? ""),
    queryFn: () => pedirValidado(Sesion, `/jobs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (enCurso(q.state.data?.estado) ? SONDEO_SESION_MS : false),
    refetchIntervalInBackground: false,
  });
  // Cuando la sesion termina de nuevo (re-siembra, tramos asignados a mano) la Fight-Card
  // en cache es la anterior: se tira para que la pantalla muestre la nueva.
  const estado = q.data?.estado;
  const previo = useRef(estado);
  useEffect(() => {
    if (id && previo.current && previo.current !== "listo" && estado === "listo") {
      qc.invalidateQueries({ queryKey: claves.fightcard(id) });
      qc.invalidateQueries({ queryKey: claves.tramos(id) });
      qc.invalidateQueries({ queryKey: claves.procesado(id) });
      qc.invalidateQueries({ queryKey: claves.sesiones });
    }
    previo.current = estado;
  }, [estado, id, qc]);
  return q;
}

export function useFightCard(id: string | null, habilitada = true) {
  return useQuery<FC>({
    queryKey: claves.fightcard(id ?? ""),
    queryFn: async () => normalizarFightCard(await pedirValidado(FightCard, `/fightcards/${id}`)),
    enabled: !!id && habilitada,
    retry: false,
  });
}

export function useBoxeadores() {
  return useQuery({
    queryKey: claves.boxeadores,
    queryFn: () => pedirValidado(ListaBoxeadores, "/boxeadores"),
  });
}

/* El perfil no se cachea: el entrenador corrige tipos y espera verlos (CA-06.2). */
export function usePerfil(id: string | null, contexto = "todas") {
  return useQuery({
    queryKey: claves.perfil(id ?? "", contexto),
    queryFn: () =>
      pedirValidado(Perfil, `/boxeadores/${id}/perfil?contexto=${encodeURIComponent(contexto)}`),
    enabled: !!id,
    staleTime: 0,
    gcTime: 0,
  });
}

export function useTramos(id: string | null, habilitada: boolean) {
  return useQuery({
    queryKey: claves.tramos(id ?? ""),
    queryFn: () => pedirValidado(Tramos, `/jobs/${id}/tramos`),
    enabled: !!id && habilitada,
    retry: false,
  });
}

/*
 * El ticket de video: `<video>` e `<img>` no mandan Authorization, asi que la URL lleva un
 * permiso de 30 minutos para un solo video. Se renueva a los 25 (02 §7).
 */
export const RENOVAR_TICKET_MS = 25 * 60 * 1000;

export function useTicket(id: string | null) {
  return useQuery({
    queryKey: claves.ticket(id ?? ""),
    queryFn: () =>
      pedirValidado(z.object({ ticket: z.string(), minutos: z.number() }), `/videos/${id}/ticket`),
    enabled: !!id,
    staleTime: RENOVAR_TICKET_MS,
    refetchInterval: RENOVAR_TICKET_MS,
    refetchIntervalInBackground: true,
  });
}

/* Si existe el video con esqueleto. HEAD: solo interesa la existencia, el archivo pesa. */
export function useProcesado(id: string | null, ticket: string | undefined) {
  return useQuery({
    queryKey: claves.procesado(id ?? ""),
    queryFn: async () => {
      const r = await fetch(`/videos/${id}/procesado?t=${encodeURIComponent(ticket!)}`, { method: "HEAD" });
      return r.ok;
    },
    enabled: !!id && !!ticket,
    staleTime: 30_000,
  });
}

// -- Mutaciones -----------------------------------------------------------------

export function ingresar(email: string, clave: string) {
  return pedirValidado(Token, "/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, clave }),
  });
}

export function registrarse(email: string, clave: string, invitacion: string) {
  return pedirValidado(Token, "/auth/registro", {
    method: "POST",
    body: JSON.stringify({ email, clave, invitacion }),
  });
}

export function useSembrar(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (s: { track_a: number; track_b: number | null }) =>
      pedir(`/jobs/${id}/siembra`, { method: "POST", body: JSON.stringify(s) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: claves.sesion(id) });
      qc.invalidateQueries({ queryKey: claves.sesiones });
      qc.invalidateQueries({ queryKey: claves.panel });
    },
  });
}

export function useAsignarTramos(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (asignaciones: { track: number; rol: "A" | "B" | "ninguno" }[]) =>
      pedir(`/jobs/${id}/tramos`, { method: "PUT", body: JSON.stringify({ asignaciones }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: claves.sesion(id) });
      qc.invalidateQueries({ queryKey: claves.tramos(id) });
    },
  });
}

export function useRender(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => pedir(`/jobs/${id}/render`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: claves.sesion(id) }),
  });
}

/*
 * Correccion optimista (02 §5): la lista cambia al guardar y vuelve atras si la API
 * responde error. "no fue un golpe" marca descartado y no toca los detectados (B4).
 */
export function aplicarCorreccionLocal(fc: FC, golpe: string, tipo: TipoCorreccion): FC {
  const copia: FC = structuredClone(fc);
  for (const l of ["A", "B"] as const) {
    const p = copia.peleadores[l];
    if (!p) continue;
    for (const g of p.golpes) {
      if (g.id !== golpe) continue;
      const original = g.corregido?.tipo_original ?? g.tipo;
      g.corregido = { tipo, tipo_original: original ?? null };
      g.descartado = tipo === "no_golpe";
      if (tipo !== "no_golpe") g.tipo = tipo;
      g.confianza_tipo = null;
    }
    p.descartados = p.golpes.filter((x) => x.descartado).length;
  }
  return copia;
}

/*
 * La altura corregida a mano (W14). Igual que el tipo, optimista; ademas recuenta cabeza y
 * cuerpo del peleador y de su round, que la API recuenta al guardar (CA-14.3).
 */
export function aplicarAlturaLocal(fc: FC, golpe: string, objetivo: "cabeza" | "cuerpo"): FC {
  const copia: FC = structuredClone(fc);
  for (const l of ["A", "B"] as const) {
    const p = copia.peleadores[l];
    if (!p || !p.golpes.some((g) => g.id === golpe)) continue;
    for (const g of p.golpes) {
      if (g.id !== golpe) continue;
      g.objetivo_corregido = { objetivo, objetivo_original: g.objetivo_corregido?.objetivo_original ?? g.objetivo ?? null };
      g.objetivo = objetivo;
      g.objetivo_confianza = null;
    }
    const tres = (gs: typeof p.golpes) => {
      const cabeza = gs.filter((g) => g.objetivo === "cabeza").length;
      const cuerpo = gs.filter((g) => g.objetivo === "cuerpo").length;
      return { cabeza, cuerpo, objetivo_sin_dato: gs.length - cabeza - cuerpo };
    };
    if (p.total.cabeza != null) Object.assign(p.total, tres(p.golpes));
    for (const r of p.por_round) {
      if (r.cabeza != null) Object.assign(r, tres(p.golpes.filter((g) => r.inicio_s <= g.t_inicio && g.t_inicio < r.fin_s)));
    }
  }
  return copia;
}

export function useCorregirAltura(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ golpe, objetivo }: { golpe: string; objetivo: "cabeza" | "cuerpo" }) =>
      pedir(`/fightcards/${id}/golpes/${encodeURIComponent(golpe)}`, {
        method: "PATCH",
        body: JSON.stringify({ objetivo }),
      }),
    onMutate: async ({ golpe, objetivo }) => {
      await qc.cancelQueries({ queryKey: claves.fightcard(id) });
      const antes = qc.getQueryData<FC>(claves.fightcard(id));
      if (antes) qc.setQueryData(claves.fightcard(id), aplicarAlturaLocal(antes, golpe, objetivo));
      return { antes };
    },
    onError: (_e, _v, ctx) => {
      if (ctx?.antes) qc.setQueryData(claves.fightcard(id), ctx.antes);
    },
    onSettled: () => qc.invalidateQueries({ queryKey: claves.fightcard(id) }),
  });
}

export function useCorregir(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ golpe, tipo }: { golpe: string; tipo: TipoCorreccion }) =>
      pedir(`/fightcards/${id}/golpes/${encodeURIComponent(golpe)}`, {
        method: "PATCH",
        body: JSON.stringify({ tipo }),
      }),
    onMutate: async ({ golpe, tipo }) => {
      await qc.cancelQueries({ queryKey: claves.fightcard(id) });
      const antes = qc.getQueryData<FC>(claves.fightcard(id));
      if (antes) qc.setQueryData(claves.fightcard(id), aplicarCorreccionLocal(antes, golpe, tipo));
      return { antes };
    },
    onError: (_e, _v, ctx) => {
      if (ctx?.antes) qc.setQueryData(claves.fightcard(id), ctx.antes);
    },
    onSettled: () => qc.invalidateQueries({ queryKey: claves.fightcard(id) }),
  });
}

export function useGuardarBoxeador() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (b: {
      id?: string; nombre: string; guardia: "ortodoxa" | "zurda" | null; notas?: string | null;
      categoria?: string | null; altura_m?: number | null; envergadura_m?: number | null;
    }) => {
      const { id, ...cuerpo } = b;
      const datos = await pedir(id ? `/boxeadores/${id}` : "/boxeadores", {
        method: id ? "PATCH" : "POST",
        body: JSON.stringify(cuerpo),
      });
      return Boxeador.parse(datos);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: claves.boxeadores }),
  });
}

export function asignarBoxeadores(id: string, a: string | null, b: string | null) {
  return pedir(`/sesiones/${id}/boxeadores`, {
    method: "PUT",
    body: JSON.stringify({ boxeador_a: a, boxeador_b: b }),
  });
}

export function invalidarTodo(qc: QueryClient) {
  qc.invalidateQueries();
}

/* Si la pestana esta visible; el sondeo se pausa cuando no. */
export function useVisible() {
  const [v, setV] = useState(true);
  useEffect(() => {
    const f = () => setV(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", f);
    return () => document.removeEventListener("visibilitychange", f);
  }, []);
  return v;
}

export { ErrorApi };
