"use client";
/* Ajustes (spec 12): plan, premium simulado, guardia, perfil publico, reentrenamiento y baja. */
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { Rotulo } from "@/components/boxtwinz/Rotulo";
import { Button } from "@/components/ui/button";
import { api, type Cupo, type Perfil } from "@/lib/api";
import { num } from "@/lib/formato";
import { TEXTOS } from "@/lib/textos";

function Contenido() {
  const { yo, recargar } = useYo();
  const router = useRouter();
  const dlg = useRef<HTMLDialogElement>(null);
  const [p1, setP1] = useState<Perfil | null>(null);
  const [cupo, setCupo] = useState<Cupo | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    api<Perfil[]>("/perfiles").then((ps) => setP1(ps.find((p) => p.ranura === 1) ?? null)).catch(() => undefined);
    api<Cupo>("/cupo").then(setCupo).catch(() => undefined);
  }, [n]);
  const listo = () => { recargar(); setN((x) => x + 1); };

  return (
    <div className="space-y-6">
      <section aria-labelledby="t-plan" className="space-y-2 rounded-lg border border-borde p-3">
        <h2 id="t-plan" className="font-medium">Plan: {yo.plan === "premium" ? "Premium" : "Free"}</h2>
        {cupo && <p className="text-sm text-texto-3">Usaste <span className="dato">{num(cupo.usados_min)}</span> de <span className="dato">{num(cupo.limite_min, 0)}</span> minutos este mes.</p>}
        {yo.plan === "free" ? (
          <>
            <ul className="list-disc pl-5 text-sm text-texto-2">
              <li>300 minutos de video por mes en lugar de 60</li>
              <li>Cajones para guardar a tus sparrings y variantes de arquetipos</li>
              <li>Sin publicidad</li>
            </ul>
            <Button className="objetivo-tactil w-full" onClick={() => api("/yo/premium", { method: "POST" }).then(listo)}>Pasar a premium (5 USD/mes)</Button>
          </>
        ) : null}
        <p><Rotulo>{TEXTOS.pagoSimulado}</Rotulo></p>
      </section>

      <section aria-labelledby="t-perfil" className="space-y-3 rounded-lg border border-borde p-3">
        <h2 id="t-perfil" className="font-medium">Tu perfil</h2>
        <fieldset>
          <legend className="mb-1 text-sm">Guardia</legend>
          {(["ortodoxa", "zurda"] as const).map((g) => (
            <label key={g} className="objetivo-tactil mr-4 inline-flex items-center gap-2">
              <input type="radio" name="guardia" checked={yo.guardia === g} onChange={() => api("/yo", { method: "PATCH", json: { guardia: g } }).then(listo)} />
              {g === "ortodoxa" ? "Ortodoxa" : "Zurda"}
            </label>
          ))}
        </fieldset>
        {p1 && (
          <label className="flex items-center gap-3">
            <input type="checkbox" className="size-5" checked={p1.publico}
              onChange={(e) => api(`/perfiles/${p1.id}`, { method: "PATCH", json: { publico: e.target.checked } }).then(listo)} />
            <span>Perfil público <span className="block text-xs text-tenue">Aparecés en el ranking y otros te pueden usar como rival fantasma. Tus cajones nunca se muestran.</span></span>
          </label>
        )}
      </section>

      <section aria-labelledby="t-reent" className="rounded-lg border border-borde p-3">
        <h2 id="t-reent" className="mb-2 font-medium">Reentrenamiento</h2>
        <label className="flex items-start gap-3 text-sm">
          <input type="checkbox" className="mt-1 size-5 shrink-0" checked={yo.consentimiento_reentrenamiento}
            onChange={(e) => api("/yo", { method: "PATCH", json: { consentimiento_reentrenamiento: e.target.checked } }).then(listo)} />
          <span>{TEXTOS.consentimientoReent}</span>
        </label>
      </section>

      <section className="space-y-2">
        <Button variant="outline" className="objetivo-tactil w-full" onClick={() => api("/auth/salida", { method: "POST" }).then(() => router.push("/"))}>Salir</Button>
        {yo.rol === "admin" && <Button variant="outline" className="objetivo-tactil w-full" onClick={() => router.push("/admin/")}>Panel de administración</Button>}
        <Button variant="destructive" className="objetivo-tactil w-full" onClick={() => dlg.current?.showModal()}>Dar de baja la cuenta</Button>
      </section>

      <dialog ref={dlg} aria-labelledby="t-baja" className="m-auto w-[min(92vw,24rem)] rounded-lg border border-borde bg-superficie p-4 text-texto backdrop:bg-black/70">
        <h2 id="t-baja" className="mb-2 font-medium">¿Dar de baja la cuenta?</h2>
        <p className="mb-4 text-sm text-texto-2">Se borran tus videos, perfiles y sesiones. Las valoraciones que ya se usaron para entrenar quedan sin tu nombre.</p>
        <div className="flex gap-2">
          <Button variant="destructive" className="flex-1" onClick={() => api("/yo", { method: "DELETE" }).then(() => router.push("/"))}>Sí, borrar todo</Button>
          <Button variant="outline" className="flex-1" onClick={() => dlg.current?.close()}>Cancelar</Button>
        </div>
      </dialog>
    </div>
  );
}

export default function Ajustes() {
  return <Marco titulo="Ajustes"><Contenido /></Marco>;
}
