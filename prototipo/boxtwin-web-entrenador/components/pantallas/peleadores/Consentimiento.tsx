"use client";

/*
 * G-08 · Consentimiento del atleta, en Gimnasio (versiones 26-10, 03 §4, W33).
 *
 * Un bloque con el estado y la historia ("Registrado por Ana el 26/10", "Revocado por Juan
 * el 30/10"), y una hoja para registrarlo que muestra el texto vigente entero. Revocar no
 * borra la constancia (CA-G08.1): frena las asignaciones nuevas y nada mas.
 *
 * El texto no lo escribe la web ni la API (DV-5). Si Lucas todavia no lo cargo, la API
 * contesta 409 y la hoja lo dice en vez de mostrar uno inventado.
 */
import { useState } from "react";
import { toast } from "sonner";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/animate-ui/components/radix/sheet";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/animate-ui/components/radix/alert-dialog";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Casilla } from "@/components/boxtwin/Controles";
import { ErrorConReintento, Esqueleto, mensajeDeError } from "@/components/boxtwin/Estados";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  useConsentimiento, useRegistrarConsentimiento, useRevocarConsentimiento, useTextoConsentimiento,
} from "@/lib/api/consultas";
import type { EstadoConsentimiento } from "@/lib/api/esquemas";
import { fechaCorta } from "@/lib/formato";

const NOMBRE_ESTADO = { vigente: "Consentimiento vigente", falta: "Falta consentimiento", revocado: "Consentimiento revocado" } as const;
const CLASE_ESTADO = {
  vigente: "border-borde-fuerte text-texto",
  falta: "border-estimado-borde text-estimado",
  revocado: "border-error-borde text-error",
} as const;

export function InsigniaConsentimiento({ c }: { c: EstadoConsentimiento | null | undefined }) {
  if (!c) return null;
  return (
    <span className={`inline-flex min-h-6 items-center rounded-sm border px-2 text-xs ${CLASE_ESTADO[c.estado]}`}>
      {NOMBRE_ESTADO[c.estado]}
    </span>
  );
}

export function BloqueConsentimiento({ boxeador, nombre }: { boxeador: string; nombre: string }) {
  const c = useConsentimiento(boxeador, true);
  const [registrar, setRegistrar] = useState(false);
  const [revocar, setRevocar] = useState(false);
  const revocarlo = useRevocarConsentimiento(boxeador);

  if (c.isPending) return <Esqueleto filas={1} />;
  if (c.isError) return <ErrorConReintento error={c.error} alReintentar={() => c.refetch()} />;
  const d = c.data;
  const semilla = d.historia.some((h) => h.semilla && !h.revocado_en);

  return (
    <section aria-label="Consentimiento" className="rounded-md border border-borde bg-superficie p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <InsigniaConsentimiento c={d} />
        {d.estado === "vigente" ? (
          !semilla && (
            <Button variant="outline" className="min-h-11" onClick={() => setRevocar(true)}>Revocar</Button>
          )
        ) : (
          <Button className="min-h-11" onClick={() => setRegistrar(true)}>Registrar</Button>
        )}
      </div>
      {d.historia.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1 text-sm text-texto-2">
          {d.historia.map((h) => (
            <li key={h.id}>
              Registrado por {h.registrado_por ?? "—"} el {fechaCorta(h.registrado_en)}
              {h.nota ? ` (${h.nota})` : ""}
              {h.revocado_en && <span className="block">Revocado por {h.revocado_por ?? "—"} el {fechaCorta(h.revocado_en)}</span>}
            </li>
          ))}
        </ul>
      )}
      <HojaConsentimiento abierta={registrar} onCerrar={() => setRegistrar(false)} boxeador={boxeador} nombre={nombre} />
      <AlertDialog open={revocar} onOpenChange={setRevocar}>
        <AlertDialogContent className="bg-superficie">
          <AlertDialogHeader>
            <AlertDialogTitle>¿Revocar el consentimiento de {nombre}?</AlertDialogTitle>
            <AlertDialogDescription className="text-texto-2">
              Desde ahora no se lo va a poder elegir en sesiones nuevas. Sus sesiones anteriores quedan; borrarlas es la supresión, que esta versión todavía no hace.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="min-h-11">No revocar</AlertDialogCancel>
            <AlertDialogAction
              className="min-h-11"
              onClick={() => revocarlo.mutate(undefined, { onError: (e) => toast.error(mensajeDeError(e)) })}
            >
              Revocar
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}

export function HojaConsentimiento({ abierta, onCerrar, boxeador, nombre, onRegistrado }: {
  abierta: boolean;
  onCerrar: () => void;
  boxeador: string;
  nombre: string;
  onRegistrado?: () => void;
}) {
  const texto = useTextoConsentimiento(abierta);
  const registrar = useRegistrarConsentimiento(boxeador);
  const [leyo, setLeyo] = useState(false);
  const [mayor, setMayor] = useState(false);
  const [nota, setNota] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function confirmar() {
    if (!texto.data) return;
    setError(null);
    try {
      await registrar.mutateAsync({ texto_version: texto.data.version, mayor_de_edad: true, nota: nota.trim() || null });
      toast.success(`Consentimiento de ${nombre} registrado.`);
      onRegistrado?.();
      onCerrar();
    } catch (e) {
      setError(mensajeDeError(e));
    }
  }

  return (
    <Sheet open={abierta} onOpenChange={(o) => !o && onCerrar()}>
      <SheetContent side="bottom" className="h-auto max-h-[90dvh] overflow-y-auto bg-superficie lg:mx-auto lg:max-w-lg lg:rounded-t-lg">
        <SheetHeader>
          <SheetTitle className="font-display text-2xl">Consentimiento de {nombre}</SheetTitle>
          <SheetDescription className="text-texto-3">Queda registrado con la fecha y tu nombre.</SheetDescription>
        </SheetHeader>
        <div className="flex flex-col gap-3 px-4 pb-6">
          {texto.isPending ? (
            <Esqueleto filas={1} />
          ) : texto.isError ? (
            <p role="alert" className="rounded-md border border-error-borde bg-error-fondo px-3 py-2 text-sm text-error">
              {mensajeDeError(texto.error)}
            </p>
          ) : (
            <>
              <Rotulo as="p">Texto vigente · versión {texto.data.version}</Rotulo>
              <div tabIndex={0} aria-label="Texto del consentimiento" className="max-h-56 overflow-y-auto whitespace-pre-line rounded-md border border-borde bg-superficie-2 p-3 text-sm">
                {texto.data.texto}
              </div>
              <Casilla checked={leyo} onChange={setLeyo}>El atleta leyó y aceptó este texto.</Casilla>
              <Casilla checked={mayor} onChange={setMayor}>Tiene 18 años o más.</Casilla>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="c-nota">Nota (opcional)</Label>
                <Input id="c-nota" placeholder="firmado en papel el 26/10" value={nota} onChange={(e) => setNota(e.target.value)} className="h-11 text-base" />
              </div>
              {error && <p role="alert" className="text-sm text-error">{error}</p>}
              <Button className="min-h-11 text-base" disabled={!leyo || !mayor || registrar.isPending} onClick={confirmar}>
                {registrar.isPending ? "Registrando…" : "Confirmar"}
              </Button>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
