"use client";
/*
 * Subir una sesion (spec 12): modalidad, perfil, guia de filmacion, video y rounds, progreso.
 * La subida va por XMLHttpRequest y no por fetch: fetch no informa el progreso de un body.
 */
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, type Cupo, type Perfil } from "@/lib/api";
import { num } from "@/lib/formato";
import { MODALIDAD } from "@/lib/textos";

const GUIA = {
  comun: ["Cámara fija, apoyada o en trípode.", "El cuerpo entero en cuadro todo el tiempo.", "Buena luz, sin contraluz de ventana."],
  sparring: ["Guantes de distinto color para cada uno: es lo que separa a uno del otro."],
};

function Contenido() {
  const { yo } = useYo();
  const router = useRouter();
  const [paso, setPaso] = useState(1);
  const [modalidad, setModalidad] = useState<"sombra" | "bolsa" | "sparring" | null>(null);
  const [perfiles, setPerfiles] = useState<Perfil[]>([]);
  const [cajon, setCajon] = useState("");
  const [archivo, setArchivo] = useState<File | null>(null);
  const [rounds, setRounds] = useState("");
  const [cupo, setCupo] = useState<Cupo | null>(null);
  const [progreso, setProgreso] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Perfil[]>("/perfiles").then(setPerfiles).catch(() => undefined);
    api<Cupo>("/cupo").then(setCupo).catch(() => undefined);
  }, []);
  const propio = perfiles.find((p) => p.ranura === 1);
  const cajones = perfiles.filter((p) => p.ranura > 1 && p.customizado);

  function subir() {
    if (!archivo || !propio || !modalidad) return;
    const fd = new FormData();
    fd.append("video", archivo);
    fd.append("modalidad", modalidad);
    fd.append("perfil_id", propio.id);
    if (rounds) fd.append("rounds_declarados", rounds);
    if (cajon && modalidad === "sparring") fd.append("rival_perfil_id", cajon);
    const x = new XMLHttpRequest();
    x.open("POST", "/api/sesiones");
    x.upload.onprogress = (e) => e.lengthComputable && setProgreso(e.loaded / e.total);
    x.onload = () => {
      if (x.status === 202) {
        const d = JSON.parse(x.responseText);
        router.push(`/sesion/?id=${d.id}`);
        return;
      }
      setProgreso(null);
      try {
        const det = JSON.parse(x.responseText).detail;
        setError(typeof det === "object" && det?.mensaje ? det.mensaje : "No se pudo subir el video");
      } catch {
        setError("No se pudo subir el video");
      }
    };
    x.onerror = () => { setProgreso(null); setError("Se cortó la conexión durante la subida"); };
    setError(null);
    setProgreso(0);
    x.send(fd);
  }

  return (
    <div className="space-y-5">
      <ol className="flex gap-1" aria-label="Pasos">
        {[1, 2, 3, 4, 5].map((n) => (
          <li key={n} aria-current={paso === n ? "step" : undefined} className={`h-1 flex-1 rounded ${paso >= n ? "bg-texto" : "bg-borde"}`} />
        ))}
      </ol>

      {paso === 1 && (
        <fieldset className="space-y-2">
          <legend className="mb-2 font-medium">1. ¿Qué filmaste?</legend>
          {(["sombra", "bolsa", "sparring"] as const).map((m) => (
            <button key={m} type="button" onClick={() => { setModalidad(m); setPaso(2); }}
              className="objetivo-tactil w-full rounded-lg border border-borde bg-superficie px-4 py-3 text-left text-lg">
              {MODALIDAD[m]}
            </button>
          ))}
        </fieldset>
      )}

      {paso === 2 && modalidad && (
        <section className="space-y-3">
          <h2 className="font-medium">2. Perfil</h2>
          <p className="rounded-lg border border-borde bg-superficie p-3">La sesión va a tu perfil: <strong>{propio?.nombre}</strong></p>
          {modalidad === "sparring" && yo.plan === "premium" && (
            <div className="space-y-1.5">
              <Label htmlFor="cajon">Guardar al rival en un cajón (opcional)</Label>
              <select id="cajon" value={cajon} onChange={(e) => setCajon(e.target.value)} className="objetivo-tactil w-full rounded-md border border-input bg-superficie-2 px-3">
                <option value="">No guardar</option>
                {cajones.map((c) => <option key={c.id} value={c.id}>{c.nombre}</option>)}
              </select>
              {cajones.length === 0 && <p className="text-xs text-tenue">Para tener cajones, personalizá una ranura en su perfil.</p>}
            </div>
          )}
          {modalidad === "sparring" && yo.plan === "free" && (
            <p className="text-sm text-texto-3">En el plan free el rival se analiza en esta sesión pero no se guarda en un perfil.</p>
          )}
          <Button className="objetivo-tactil w-full" onClick={() => setPaso(3)}>Seguir</Button>
        </section>
      )}

      {paso === 3 && modalidad && (
        <section className="space-y-3">
          <h2 className="font-medium">3. Antes de filmar</h2>
          <ul className="list-disc space-y-1 pl-5 text-texto-2">
            {[...GUIA.comun, ...(modalidad === "sparring" ? GUIA.sparring : [])].map((g) => <li key={g}>{g}</li>)}
          </ul>
          <Button className="objetivo-tactil w-full" onClick={() => setPaso(4)}>Ya lo tengo filmado</Button>
        </section>
      )}

      {paso === 4 && (
        <section className="space-y-4">
          <h2 className="font-medium">4. Video</h2>
          <div className="space-y-1.5">
            <Label htmlFor="video">Video (mp4 o mov)</Label>
            <Input id="video" type="file" accept="video/mp4,video/quicktime,.mp4,.mov,.m4v" onChange={(e) => setArchivo(e.target.files?.[0] ?? null)} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="rounds">Rounds (opcional)</Label>
            <Input id="rounds" type="number" inputMode="numeric" min={1} max={30} value={rounds} onChange={(e) => setRounds(e.target.value)} placeholder="Sin rounds: bloques de 3 minutos" />
          </div>
          <Button className="objetivo-tactil w-full" disabled={!archivo} onClick={() => { setPaso(5); subir(); }}>Subir</Button>
        </section>
      )}

      {paso === 5 && (
        <section className="space-y-3" aria-live="polite">
          <h2 className="font-medium">5. Subiendo</h2>
          {progreso != null && (
            <>
              <progress className="h-2 w-full" value={progreso} max={1} aria-label="Progreso de la subida" />
              <p className="dato">{Math.round(progreso * 100)}%</p>
            </>
          )}
          {error && (
            <>
              <p role="alert" className="text-error">{error}</p>
              <Button variant="outline" className="objetivo-tactil" onClick={() => setPaso(4)}>Volver</Button>
            </>
          )}
        </section>
      )}

      {cupo && (
        <p className="text-sm text-texto-3">
          Cupo del mes: <span className="dato">{num(cupo.restantes_min)}</span> de <span className="dato">{num(cupo.limite_min, 0)}</span> minutos disponibles.
        </p>
      )}
    </div>
  );
}

export default function Subir() {
  return <Marco titulo="Subir sesión"><Contenido /></Marco>;
}
