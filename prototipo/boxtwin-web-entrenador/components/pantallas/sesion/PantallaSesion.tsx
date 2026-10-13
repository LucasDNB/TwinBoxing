"use client";

/*
 * 04 · Analisis de sesion y 05 · Fight-Card (04 §04, §05). Ruta /sesion/?id=&vista=.
 *
 * Encabezado comun: contexto, fecha, rounds, A vs B con sus marcas, y tres pestanas. La
 * vista va en la query para que el boton atras del celular haga lo esperable y un enlace
 * abra la misma vista (02 §4).
 *
 * Si la sesion no esta lista, la vista muestra el estado en lugar del contenido.
 */
import { useRouter, useSearchParams } from "next/navigation";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { NombreEsquina } from "@/components/boxtwin/Esquina";
import { ErrorConReintento, Esqueleto } from "@/components/boxtwin/Estados";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/animate-ui/components/radix/tabs";
import { EsperaSesion } from "@/components/pantallas/sesion/EsperaSesion";
import { VistaVideo } from "@/components/pantallas/sesion/VistaVideo";
import { VistaFightCard } from "@/components/pantallas/sesion/VistaFightCard";
import { VistaPerfiles } from "@/components/pantallas/sesion/VistaPerfiles";
import { useFightCard, useSesion } from "@/lib/api/consultas";
import type { Sesion } from "@/lib/api/esquemas";
import type { FightCard } from "@/lib/fightcard";
import { fechaCorta } from "@/lib/formato";
import { NOMBRE_CONTEXTO } from "@/lib/textos";

const VISTAS = ["video", "fightcard", "perfiles"] as const;
type Vista = (typeof VISTAS)[number];

export function nombresDe(s: Sesion, fc?: FightCard | null) {
  return {
    A: fc?.sesion?.boxeadores?.A?.nombre ?? s.boxeadores?.A?.nombre ?? null,
    B: fc?.sesion?.boxeadores?.B?.nombre ?? s.boxeadores?.B?.nombre ?? null,
  };
}

export function PantallaSesion() {
  const q = useSearchParams();
  const id = q.get("id");
  const sesion = useSesion(id);
  const lista = sesion.data?.estado === "listo";
  const fc = useFightCard(id, lista);

  if (!id) return <ErrorConReintento error={new Error("Falta el identificador de la sesión.")} />;
  if (sesion.isPending) return <Esqueleto filas={4} />;
  if (sesion.isError) return <ErrorConReintento error={sesion.error} alReintentar={() => sesion.refetch()} />;
  const s = sesion.data;

  return (
    <>
      <Cabecera s={s} fc={fc.data} />
      {!lista ? (
        <EsperaSesion s={s} />
      ) : fc.isPending ? (
        <Esqueleto filas={4} />
      ) : fc.isError ? (
        <ErrorConReintento titulo="No se pudo leer la Fight-Card" error={fc.error} alReintentar={() => fc.refetch()} />
      ) : (
        <Vistas s={s} fc={fc.data} />
      )}
    </>
  );
}

function Cabecera({ s, fc }: { s: Sesion; fc?: FightCard }) {
  const n = nombresDe(s, fc);
  const deDos = !!fc?.peleadores.B || (s.contexto !== "bolsa" && s.contexto !== "sombra");
  const rondas = fc?.video.rounds.length ?? s.rounds.cantidad ?? 0;
  return (
    <Encabezado
      atras="/sesiones/"
      rotulo={[NOMBRE_CONTEXTO[s.contexto ?? "sparring"], fechaCorta(s.creada), rondas ? `${rondas} ${rondas === 1 ? "round" : "rounds"}` : "sin rounds"].join(" · ")}
      titulo={
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[26px] lg:text-[40px]">
          <NombreEsquina lado="A" nombre={n.A} />
          {deDos && (<><span className="text-tenue">vs</span><NombreEsquina lado="B" nombre={n.B} /></>)}
        </span>
      }
    />
  );
}

function Vistas({ s, fc }: { s: Sesion; fc: FightCard }) {
  const router = useRouter();
  const q = useSearchParams();
  const vista: Vista = (VISTAS as readonly string[]).includes(q.get("vista") ?? "") ? (q.get("vista") as Vista) : "video";

  function ir(v: string) {
    const p = new URLSearchParams(q.toString());
    p.set("vista", v);
    p.delete("t");
    router.replace(`/sesion/?${p.toString()}`);
  }

  return (
    <Tabs value={vista} onValueChange={ir}>
      <div className="-mx-4 mb-4 overflow-x-auto px-4">
        <TabsList className="w-max">
          <TabsTrigger value="video" className="min-h-11 px-4">Video y golpes</TabsTrigger>
          <TabsTrigger value="fightcard" className="min-h-11 px-4">Fight-Card</TabsTrigger>
          <TabsTrigger value="perfiles" className="min-h-11 px-4">Perfiles de la sesión</TabsTrigger>
        </TabsList>
      </div>
      {/* Solo la vista activa se monta (el video no se carga si se mira la Fight-Card), pero
          dentro de su TabsContent: el aria-controls de la pestana tiene que apuntar a algo. */}
      <TabsContent value={vista} transition={{ duration: 0.2, ease: "easeOut" }}>
        {vista === "video" && <VistaVideo s={s} fc={fc} />}
        {vista === "fightcard" && <VistaFightCard s={s} fc={fc} />}
        {vista === "perfiles" && <VistaPerfiles s={s} fc={fc} />}
      </TabsContent>
    </Tabs>
  );
}
