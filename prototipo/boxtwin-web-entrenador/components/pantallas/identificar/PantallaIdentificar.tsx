"use client";

/*
 * 03 · Identificacion de peleadores (04 §03).
 *
 * Dos fases, porque la propagacion tarda: el diseno las dibuja juntas, la web las separa en
 * el tiempo. Fase 1 (espera_siembra): marcar una vez a cada peleador. Fase 2 (listo):
 * revisar los tramos que quedaron sin asignar (FaseRevisar).
 *
 * Funciona sin B2: si el candidato no trae color, tramo o altura, la tarjeta muestra el
 * recorte y los botones, que es lo que la siembra necesita (01 §7, riesgo 2).
 *
 * En bolsa y sombra (W19, ampliacion 02 §3) la pantalla elige el modo por el contexto: se
 * marca a una sola persona, con "Es él" y sin A, B ni rincones, y antes de confirmar avisa si
 * el elegido coexiste con otra figura de guantes parecidos (el espejo).
 *
 * En la version Individual (versiones 26-10, 03 §I-03, W30) la pantalla le habla al boxeador:
 * "¿Cuál sos vos?" con "Soy yo", y en sparring y combate un segundo paso, "¿Y tu compañero?",
 * sin A, B ni rincones. "Soy yo" es track_a.
 */
import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { ErrorConReintento, Esqueleto, mensajeDeError } from "@/components/boxtwin/Estados";
import { Esquina } from "@/components/boxtwin/Esquina";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { Fade } from "@/components/animate-ui/primitives/effects/fade";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/animate-ui/components/radix/alert-dialog";
import { EsperaSesion } from "@/components/pantallas/sesion/EsperaSesion";
import { FaseRevisar } from "@/components/pantallas/identificar/FaseRevisar";
import { useFightCard, useSembrar, useSesion, useTicket } from "@/lib/api/consultas";
import type { Candidato, Sesion } from "@/lib/api/esquemas";
import { decimal, fechaCorta, mmss, nombreCorto } from "@/lib/formato";
import { NOMBRE_CONTEXTO, textos } from "@/lib/textos";
import { esIndividual as esCuentaIndividual, useCuenta } from "@/lib/cuenta";

type Rol = "A" | "B" | "ninguno";

export function PantallaIdentificar() {
  const id = useSearchParams().get("id");
  const sesion = useSesion(id);
  if (!id) return <ErrorConReintento error={new Error("Falta el identificador de la sesión.")} />;
  if (sesion.isPending) return <Esqueleto filas={3} />;
  if (sesion.isError) return <ErrorConReintento error={sesion.error} alReintentar={() => sesion.refetch()} />;
  return <Identificar s={sesion.data} />;
}

export function titularDeSesion(s: Sesion): string {
  const ctx = NOMBRE_CONTEXTO[s.contexto ?? "sparring"];
  const a = s.boxeadores?.A?.nombre;
  const b = s.boxeadores?.B?.nombre;
  const quienes = a && b ? `${nombreCorto(a)} vs ${nombreCorto(b)}` : a ? nombreCorto(a) : b ? nombreCorto(b) : null;
  return [ctx, fechaCorta(s.creada), quienes].filter(Boolean).join(" · ");
}

export function esIndividual(s: Sesion): boolean {
  return s.contexto === "bolsa" || s.contexto === "sombra";
}

function Identificar({ s }: { s: Sesion }) {
  const individual = esIndividual(s);
  const vos = esCuentaIndividual(useCuenta().data);
  const titulo = vos ? textos.individualVos.titulo : individual ? textos.individual.titulo : "¿Quién es quién?";
  const indicacion = vos
    ? individual ? textos.individualVos.indicacion : "Tocá primero tu figura y después la de tu compañero. El resto del video se sigue por el color de los guantes."
    : individual
      ? textos.individual.indicacion
      : "Marcá una vez a cada peleador. El resto del video se sigue por altura y color de guante.";
  const encabezado = (
    <Encabezado titulo={titulo} rotulo={titularDeSesion(s)} atras={`/sesion/?id=${s.id}`}>
      {s.estado === "espera_siembra" && <p className="text-texto-2">{indicacion}</p>}
    </Encabezado>
  );
  if (s.estado === "espera_siembra") {
    const fase = individual ? <FaseMarcarUna s={s} vos={vos} /> : vos ? <FaseMarcarVos s={s} /> : <FaseMarcar s={s} />;
    return <>{encabezado}{fase}</>;
  }
  if (s.estado === "listo") return <>{encabezado}<FaseRevisarSiHace s={s} /></>;
  return <>{encabezado}<EsperaSesion s={s} /></>;
}

/* Fase 2 solo si quedo algo sin asignar; si no, a la Fight-Card (04 §03). */
function FaseRevisarSiHace({ s }: { s: Sesion }) {
  const router = useRouter();
  const fc = useFightCard(s.id);
  const sinAsignar = fc.data?.identidad.sin_asignar ?? null;
  useEffect(() => {
    if (fc.data && (sinAsignar ?? 0) <= 0) router.replace(`/sesion/?id=${s.id}&vista=fightcard`);
  }, [fc.data, sinAsignar, router, s.id]);
  if (fc.isPending) return <Esqueleto />;
  if (fc.isError) return <ErrorConReintento error={fc.error} alReintentar={() => fc.refetch()} />;
  if ((sinAsignar ?? 0) <= 0) return <Esqueleto />;
  return <FaseRevisar s={s} fc={fc.data} />;
}

/* Un solo A y un solo B: marcar un segundo A le saca la marca al anterior, con aviso. */
export function marcar(actual: Record<number, Rol>, track: number, rol: Rol | ""): { nuevo: Record<number, Rol>; quitado: number | null } {
  const nuevo = { ...actual };
  let quitado: number | null = null;
  if (!rol) {
    delete nuevo[track];
    return { nuevo, quitado };
  }
  if (rol === "A" || rol === "B") {
    for (const [t, r] of Object.entries(nuevo)) {
      if (r === rol && Number(t) !== track) {
        delete nuevo[Number(t)];
        quitado = Number(t);
      }
    }
  }
  nuevo[track] = rol;
  return { nuevo, quitado };
}

function FaseMarcar({ s }: { s: Sesion }) {
  const sembrar = useSembrar(s.id);
  const ticket = useTicket(s.id);
  const [marcas, setMarcas] = useState<Record<number, Rol>>({});
  const [confirmar, setConfirmar] = useState(false);

  const candidatos = useMemo(() => {
    const cs = s.candidatos ?? [];
    const sug = new Set(s.pareja_sugerida ?? []);
    return [...cs].sort((a, b) => Number(sug.has(b.track) || !!b.sugerido) - Number(sug.has(a.track) || !!a.sugerido));
  }, [s.candidatos, s.pareja_sugerida]);
  const sugeridos = new Set([...(s.pareja_sugerida ?? []), ...candidatos.filter((c) => c.sugerido).map((c) => c.track)]);
  const persona = (track: number) => candidatos.findIndex((c) => c.track === track) + 1;

  const trackA = Number(Object.entries(marcas).find(([, r]) => r === "A")?.[0] ?? NaN);
  const trackB = Number(Object.entries(marcas).find(([, r]) => r === "B")?.[0] ?? NaN);
  const listo = !Number.isNaN(trackA) && !Number.isNaN(trackB);

  function cambiar(track: number, rol: string) {
    const { nuevo, quitado } = marcar(marcas, track, rol as Rol | "");
    if (quitado != null) toast(`Persona ${persona(quitado)} deja de ser ${rol}: hay un solo ${rol}.`);
    setMarcas(nuevo);
  }

  async function enviar() {
    try {
      await sembrar.mutateAsync({ track_a: trackA, track_b: trackB });
      toast.success("Listo. El análisis sigue con esos dos.");
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  const candA = candidatos.find((c) => c.track === trackA);
  const candB = candidatos.find((c) => c.track === trackB);
  const nombreA = s.boxeadores?.A?.nombre ?? "A";
  const nombreB = s.boxeadores?.B?.nombre ?? "B";
  const pregunta = candA?.color_guante && candB?.color_guante && candA.color_guante.nombre !== candB.color_guante.nombre
    ? `¿Guantes ${candA.color_guante.nombre} es ${nombreA} y guantes ${candB.color_guante.nombre} es ${nombreB}?`
    : `¿Persona ${persona(trackA)} es ${nombreA} y persona ${persona(trackB)} es ${nombreB}?`;

  if (candidatos.length === 0) {
    return <ErrorConReintento titulo="No hay personas para marcar" error={new Error("El sistema no encontró candidatos en este video. Revisá que se vean los dos de cuerpo entero.")} />;
  }

  return (
    <>
      {!s.pareja_sugerida && (
        <p role="note" className="mb-4 rounded-md border border-estimado-borde bg-estimado-fondo px-3 py-2 text-sm">
          No encontramos un momento con los dos en cuadro a la vez. Fijate bien que no sea la misma persona dos veces.
        </p>
      )}
      <h2 className="mb-3 font-display text-xl font-semibold lg:text-2xl">Personas encontradas</h2>
      <Fade className="grid gap-3 lg:grid-cols-3">
        {candidatos.map((c, i) => (
          <TarjetaCandidato
            key={c.track}
            c={c}
            n={i + 1}
            sugerido={sugeridos.has(c.track)}
            rol={marcas[c.track] ?? ""}
            onRol={(r) => cambiar(c.track, r)}
            ticket={ticket.data?.ticket}
          />
        ))}
      </Fade>

      <div
        className="fixed inset-x-0 z-30 border-t border-borde-suave bg-fondo/95 px-4 py-3 backdrop-blur lg:static lg:mt-6 lg:border-0 lg:bg-transparent lg:p-0"
        style={{ bottom: "calc(env(safe-area-inset-bottom) + 64px)" }}
      >
        <Button className="min-h-12 w-full text-base lg:w-auto lg:px-10" disabled={!listo || sembrar.isPending} onClick={() => setConfirmar(true)}>
          Completar análisis
        </Button>
        {!listo && <p className="mt-1 text-xs text-texto-3">Marcá a A y a B para seguir.</p>}
      </div>
      <div aria-hidden="true" className="h-28 lg:hidden" />

      <AlertDialog open={confirmar} onOpenChange={setConfirmar}>
        <AlertDialogContent className="bg-superficie">
          <AlertDialogHeader>
            <AlertDialogTitle>{pregunta}</AlertDialogTitle>
            <AlertDialogDescription className="text-texto-2">
              El resto del video se sigue con estos dos. Si después ves que quedaron al revés, se puede volver a marcar.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="grid grid-cols-2 gap-3">
            {[candA, candB].map((c, i) => c && (
              <figure key={c.track} className="flex flex-col items-center gap-2">
                <Recorte c={c} ticket={ticket.data?.ticket} grande />
                <figcaption className="flex items-center gap-2 text-sm">
                  <Esquina lado={i === 0 ? "A" : "B"} /> {i === 0 ? nombreA : nombreB}
                </figcaption>
              </figure>
            ))}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel className="min-h-11">Revisar</AlertDialogCancel>
            <AlertDialogAction className="min-h-11" onClick={enviar}>Sí, completar</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

/*
 * W19: una sola persona. El de mas tiempo en cuadro va primero y sugerido; no hay pareja
 * sugerida. Una sola eleccion en toda la pantalla, aunque cada tarjeta tenga su boton.
 */
export function ordenIndividual(cs: Candidato[]): Candidato[] {
  const dur = (c: Candidato) => (c.t_hasta ?? 0) - (c.t_desde ?? 0);
  return [...cs].sort((a, b) => dur(b) - dur(a) || a.track - b.track);
}

function FaseMarcarUna({ s, vos = false }: { s: Sesion; vos?: boolean }) {
  const sembrar = useSembrar(s.id);
  const ticket = useTicket(s.id);
  const [elegido, setElegido] = useState<number | null>(null);
  const [confirmar, setConfirmar] = useState(false);
  const candidatos = useMemo(() => ordenIndividual(s.candidatos ?? []), [s.candidatos]);
  const cand = candidatos.find((c) => c.track === elegido);
  const reflejos = (cand?.mismo_color_a_la_vez ?? [])
    .map((t) => candidatos.find((c) => c.track === t))
    .filter((c): c is Candidato => !!c);
  const nombre = s.boxeadores?.A?.nombre;

  async function enviar() {
    if (elegido == null) return;
    try {
      await sembrar.mutateAsync({ track_a: elegido, track_b: null });
      toast.success("Listo. El análisis sigue con esa persona.");
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  if (candidatos.length === 0) {
    return <ErrorConReintento titulo="No hay personas para marcar" error={new Error("El sistema no encontró a nadie en este video. Revisá que se vea de cuerpo entero.")} />;
  }

  return (
    <>
      <h2 className="mb-3 font-display text-xl font-semibold lg:text-2xl">Personas encontradas</h2>
      <Fade className="grid gap-3 lg:grid-cols-3">
        {candidatos.map((c, i) => (
          <article key={c.track} aria-label={`Persona ${i + 1}`} className={`rounded-md border bg-superficie p-3 ${elegido === c.track ? "border-a-trazo" : "border-borde"}`}>
            <div className="flex gap-3 lg:flex-col lg:items-center">
              <Recorte c={c} ticket={ticket.data?.ticket} />
              <div className="min-w-0 flex-1 lg:w-full">
                <div className="flex items-center gap-2">
                  <h3 className="font-display text-lg font-semibold">Persona {i + 1}</h3>
                  {i === 0 && <Rotulo className="text-estimado">sugerido</Rotulo>}
                </div>
                <DatosCandidato c={c} />
              </div>
            </div>
            <ToggleGroup
              type="single"
              value={elegido === c.track ? "el" : ""}
              onValueChange={(v) => setElegido(v ? c.track : null)}
              aria-label={`¿La persona ${i + 1} es el boxeador?`}
              className="mt-3 grid w-full grid-cols-1"
            >
              <ToggleGroupItem value="el" className="min-h-11 border border-borde">{vos ? "Soy yo" : "Es él"}</ToggleGroupItem>
            </ToggleGroup>
          </article>
        ))}
      </Fade>

      <div
        className="fixed inset-x-0 z-30 border-t border-borde-suave bg-fondo/95 px-4 py-3 backdrop-blur lg:static lg:mt-6 lg:border-0 lg:bg-transparent lg:p-0"
        style={{ bottom: "calc(env(safe-area-inset-bottom) + 64px)" }}
      >
        <Button className="min-h-12 w-full text-base lg:w-auto lg:px-10" disabled={elegido == null || sembrar.isPending} onClick={() => setConfirmar(true)}>
          Completar análisis
        </Button>
        {elegido == null && <p className="mt-1 text-xs text-texto-3">Elegí a la persona que está trabajando para seguir.</p>}
      </div>
      <div aria-hidden="true" className="h-28 lg:hidden" />

      <AlertDialog open={confirmar} onOpenChange={setConfirmar}>
        <AlertDialogContent className="bg-superficie">
          <AlertDialogHeader>
            <AlertDialogTitle>{vos ? "¿Sos vos?" : nombre ? `¿Este es ${nombre}?` : "¿Este es el boxeador?"}</AlertDialogTitle>
            <AlertDialogDescription className="text-texto-2">
              {reflejos.length > 0 ? textos.individual.espejo : "El resto del video se sigue con esta persona. Si después ves que no era, se puede volver a marcar."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {cand && (
            <div className={`grid gap-3 ${reflejos.length ? "grid-cols-2" : "grid-cols-1 justify-items-center"}`}>
              <figure className="flex flex-col items-center gap-2">
                <Recorte c={cand} ticket={ticket.data?.ticket} grande />
                <figcaption className="text-sm">{reflejos.length ? "elegido" : vos ? "vos" : nombre ?? "el boxeador"}</figcaption>
              </figure>
              {reflejos.map((r) => (
                <figure key={r.track} className="flex flex-col items-center gap-2">
                  <Recorte c={r} ticket={ticket.data?.ticket} grande />
                  <figcaption className="text-sm">a la vez, mismos guantes</figcaption>
                </figure>
              ))}
            </div>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel className="min-h-11">Revisar</AlertDialogCancel>
            <AlertDialogAction className="min-h-11" onClick={enviar}>Sí, completar</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

/*
 * Individual en sparring y combate (03 §I-03): primero "¿Cuál sos vos?", despues "¿Y tu
 * compañero?" sobre las tarjetas que quedan. El compañero no tiene perfil (DV-4), pero la
 * siembra lo necesita para separar a los dos.
 */
function FaseMarcarVos({ s }: { s: Sesion }) {
  const sembrar = useSembrar(s.id);
  const ticket = useTicket(s.id);
  const [yo, setYo] = useState<number | null>(null);
  const [compañero, setCompañero] = useState<number | null>(null);
  const [confirmar, setConfirmar] = useState(false);
  const candidatos = useMemo(() => ordenIndividual(s.candidatos ?? []), [s.candidatos]);
  const n = (track: number | null) => candidatos.findIndex((c) => c.track === track) + 1;
  const candYo = candidatos.find((c) => c.track === yo);
  const candComp = candidatos.find((c) => c.track === compañero);
  const pregunta = candYo?.color_guante && candComp?.color_guante && candYo.color_guante.nombre !== candComp.color_guante.nombre
    ? `¿Vos sos el de guantes ${candYo.color_guante.nombre} y tu compañero el de ${candComp.color_guante.nombre}?`
    : `¿Vos sos la persona ${n(yo)} y tu compañero la persona ${n(compañero)}?`;

  async function enviar() {
    if (yo == null || compañero == null) return;
    try {
      await sembrar.mutateAsync({ track_a: yo, track_b: compañero });
      toast.success("Listo. El análisis sigue con ustedes dos.");
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  if (candidatos.length < 2) {
    return <ErrorConReintento titulo="No hay dos personas para marcar" error={new Error("El sistema no encontró a los dos en este video. Revisá que se vean de cuerpo entero.")} />;
  }

  const tarjetas = (paso: "yo" | "compañero") => (
    <Fade className="grid gap-3 lg:grid-cols-3">
      {candidatos.filter((c) => paso === "yo" || c.track !== yo).map((c) => {
        const elegido = paso === "yo" ? yo === c.track : compañero === c.track;
        return (
          <article key={c.track} aria-label={`Persona ${n(c.track)}`} className={`rounded-md border bg-superficie p-3 ${elegido ? "border-a-trazo" : "border-borde"}`}>
            <div className="flex gap-3 lg:flex-col lg:items-center">
              <Recorte c={c} ticket={ticket.data?.ticket} />
              <div className="min-w-0 flex-1 lg:w-full">
                <h3 className="font-display text-lg font-semibold">Persona {n(c.track)}</h3>
                <DatosCandidato c={c} />
              </div>
            </div>
            <ToggleGroup
              type="single"
              value={elegido ? "si" : ""}
              onValueChange={(v) => {
                if (paso === "yo") {
                  setYo(v ? c.track : null);
                  if (v && compañero === c.track) setCompañero(null);
                } else setCompañero(v ? c.track : null);
              }}
              aria-label={paso === "yo" ? `¿La persona ${n(c.track)} sos vos?` : `¿La persona ${n(c.track)} es tu compañero?`}
              className="mt-3 grid w-full grid-cols-1"
            >
              <ToggleGroupItem value="si" className="min-h-11 border border-borde">{paso === "yo" ? "Soy yo" : "Es mi compañero"}</ToggleGroupItem>
            </ToggleGroup>
          </article>
        );
      })}
    </Fade>
  );

  return (
    <>
      <h2 className="mb-3 font-display text-xl font-semibold lg:text-2xl">{textos.individualVos.titulo}</h2>
      {tarjetas("yo")}
      {yo != null && (
        <>
          <h2 className="mb-3 mt-6 font-display text-xl font-semibold lg:text-2xl">{textos.individualVos.compañero}</h2>
          {tarjetas("compañero")}
        </>
      )}

      <div
        className="fixed inset-x-0 z-30 border-t border-borde-suave bg-fondo/95 px-4 py-3 backdrop-blur lg:static lg:mt-6 lg:border-0 lg:bg-transparent lg:p-0"
        style={{ bottom: "calc(env(safe-area-inset-bottom) + 64px)" }}
      >
        <Button className="min-h-12 w-full text-base lg:w-auto lg:px-10" disabled={yo == null || compañero == null || sembrar.isPending} onClick={() => setConfirmar(true)}>
          Completar análisis
        </Button>
        {(yo == null || compañero == null) && <p className="mt-1 text-xs text-texto-3">Marcate a vos y a tu compañero para seguir.</p>}
      </div>
      <div aria-hidden="true" className="h-28 lg:hidden" />

      <AlertDialog open={confirmar} onOpenChange={setConfirmar}>
        <AlertDialogContent className="bg-superficie">
          <AlertDialogHeader>
            <AlertDialogTitle>{pregunta}</AlertDialogTitle>
            <AlertDialogDescription className="text-texto-2">
              El resto del video se sigue con ustedes dos. Si después ves que quedaron al revés, se puede volver a marcar.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="grid grid-cols-2 gap-3">
            {[candYo, candComp].map((c, i) => c && (
              <figure key={c.track} className="flex flex-col items-center gap-2">
                <Recorte c={c} ticket={ticket.data?.ticket} grande />
                <figcaption className="text-sm">{i === 0 ? "vos" : "tu compañero"}</figcaption>
              </figure>
            ))}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel className="min-h-11">Revisar</AlertDialogCancel>
            <AlertDialogAction className="min-h-11" onClick={enviar}>Sí, completar</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

/*
 * El recorte entra con ticket en la URL: un <img> no manda Authorization (02 §7). La
 * proporcion queda reservada para que no salte la pagina al cargar (CLS, 03 §9).
 */
export function Recorte({ c, ticket, grande = false }: { c: { recorte_url?: string | null; track: number }; ticket?: string; grande?: boolean }) {
  const ancho = grande ? 120 : 96;
  const alto = grande ? 170 : 136;
  if (!c.recorte_url) {
    return (
      <div style={{ width: ancho, height: alto }} className="flex shrink-0 items-center justify-center rounded-sm border border-borde bg-superficie-2 p-2 text-center text-xs text-texto-3">
        sin recorte
      </div>
    );
  }
  if (!ticket) return <div style={{ width: ancho, height: alto }} className="shrink-0 animate-pulse rounded-sm bg-elevado" />;
  return (
    // eslint-disable-next-line @next/next/no-img-element -- exportacion estatica sin optimizador; la URL lleva ticket
    <img
      src={`${c.recorte_url}?t=${encodeURIComponent(ticket)}`}
      alt={`Recorte de la persona del tramo ${c.track}`}
      width={ancho}
      height={alto}
      style={{ width: ancho, height: alto }}
      className="shrink-0 rounded-sm border border-borde bg-superficie-2 object-cover"
      loading="lazy"
    />
  );
}

function TarjetaCandidato({
  c, n, sugerido, rol, onRol, ticket,
}: {
  c: Candidato; n: number; sugerido: boolean; rol: string; onRol: (r: string) => void; ticket?: string;
}) {
  return (
    <article aria-label={`Persona ${n}`} className={`rounded-md border bg-superficie p-3 ${rol === "A" ? "border-a-trazo" : rol === "B" ? "border-b-trazo" : "border-borde"}`}>
      <div className="flex gap-3 lg:flex-col lg:items-center">
        <Recorte c={c} ticket={ticket} />
        <div className="min-w-0 flex-1 lg:w-full">
          <div className="flex items-center gap-2">
            <h3 className="font-display text-lg font-semibold">Persona {n}</h3>
            {sugerido && <Rotulo className="text-estimado">sugerido</Rotulo>}
          </div>
          <DatosCandidato c={c} />
        </div>
      </div>
      <ToggleGroup type="single" value={rol} onValueChange={onRol} aria-label={`Quién es la persona ${n}`} className="mt-3 grid w-full grid-cols-3 gap-1">
        <ToggleGroupItem value="A" className="min-h-11 border border-borde">Es A</ToggleGroupItem>
        <ToggleGroupItem value="B" className="min-h-11 border border-borde">Es B</ToggleGroupItem>
        <ToggleGroupItem value="ninguno" className="min-h-11 border border-borde">Ninguno</ToggleGroupItem>
      </ToggleGroup>
    </article>
  );
}

function DatosCandidato({ c }: { c: Candidato }) {
  return (
    <dl className="mt-1 space-y-0.5 text-sm text-texto-2">
      {c.color_guante && (
        <div className="flex items-center gap-2">
          <dt className="sr-only">Guantes</dt>
          <span aria-hidden="true" className="inline-block size-3 rounded-full border border-borde-fuerte" style={{ background: c.color_guante.hex }} />
          <dd>guantes {c.color_guante.nombre}</dd>
        </div>
      )}
      {c.t_desde != null && c.t_hasta != null && (
        <div><dt className="sr-only">En cuadro</dt><dd className="dato">{mmss(c.t_desde)} a {mmss(c.t_hasta)}</dd></div>
      )}
      {c.alto_relativo != null && (
        <div><dt className="sr-only">Altura</dt><dd>altura {decimal(c.alto_relativo, 2)} de la mediana</dd></div>
      )}
    </dl>
  );
}

