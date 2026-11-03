"use client";
/* Dialogo de denuncia de un perfil (spec 5 y 12). */
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { api, ErrorApi } from "@/lib/api";

const MOTIVOS: Record<string, string> = {
  no_es_la_persona: "No es la persona que dice ser", video_ajeno: "Usa videos ajenos",
  contenido_inapropiado: "Contenido inapropiado", menor_en_video: "Aparece un menor de edad", otro: "Otro",
};

export function Denunciar({ perfilId }: { perfilId: string }) {
  const dlg = useRef<HTMLDialogElement>(null);
  const [motivo, setMotivo] = useState("no_es_la_persona");
  const [detalle, setDetalle] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  async function enviar() {
    try {
      await api("/denuncias", { method: "POST", json: { perfil_id: perfilId, motivo, detalle: detalle || null } });
      setMsg("Recibimos la denuncia. La revisa una persona.");
    } catch (e) { setMsg(e instanceof ErrorApi ? e.message : "No se pudo enviar"); }
  }
  return (
    <>
      <Button variant="outline" className="objetivo-tactil" onClick={() => { setMsg(null); dlg.current?.showModal(); }}>Denunciar</Button>
      <dialog ref={dlg} aria-labelledby="t-den" className="m-auto w-[min(92vw,24rem)] rounded-lg border border-borde bg-superficie p-4 text-texto backdrop:bg-black/70">
        <h2 id="t-den" className="mb-3 font-medium">Denunciar perfil</h2>
        {msg ? <p role="status">{msg}</p> : (
          <div className="space-y-3">
            <fieldset className="space-y-1">
              <legend className="mb-1 text-sm">Motivo</legend>
              {Object.entries(MOTIVOS).map(([k, v]) => (
                <label key={k} className="flex items-center gap-2 text-sm"><input type="radio" name="motivo" checked={motivo === k} onChange={() => setMotivo(k)} />{v}</label>
              ))}
            </fieldset>
            <label className="block text-sm">Detalle{motivo === "otro" ? " (obligatorio)" : " (opcional)"}
              <textarea value={detalle} onChange={(e) => setDetalle(e.target.value)} rows={3} maxLength={2000} className="mt-1 w-full rounded-md border border-input bg-superficie-2 p-2" />
            </label>
            <Button className="objetivo-tactil w-full" onClick={enviar}>Enviar denuncia</Button>
          </div>
        )}
        <Button variant="ghost" className="mt-2 w-full" onClick={() => dlg.current?.close()}>Cerrar</Button>
      </dialog>
    </>
  );
}

