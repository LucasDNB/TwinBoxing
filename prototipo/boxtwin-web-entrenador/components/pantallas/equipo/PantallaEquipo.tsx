"use client";

/*
 * G-09 · Equipo, solo el titular de Gimnasio (versiones 26-10, 03 §4, W32).
 *
 * En la version de prueba: la version y sus horas, los miembros sin email y sin "Dar de baja",
 * la linea de que el equipo viene cargado y la descarga del registro de consentimientos.
 * Invitar y dar de baja quedan especificados con B23 y no se construyen (CA-G09.0).
 */
import { useState } from "react";
import { toast } from "sonner";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { ErrorConReintento, Esqueleto, mensajeDeError } from "@/components/boxtwin/Estados";
import { FueraDeVersion } from "@/components/boxtwin/FueraDeVersion";
import { Horas } from "@/components/boxtwin/Horas";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { leerToken } from "@/lib/sesion-usuario";
import { useMiembros } from "@/lib/api/consultas";
import { NOMBRE_VERSION, esGimnasio, esTitular, useCuenta } from "@/lib/cuenta";
import { decimal, fechaCorta } from "@/lib/formato";
import { textos } from "@/lib/textos";

const NOMBRE_ROL = { titular: "Titular", entrenador: "Entrenador" } as const;

export function PantallaEquipo() {
  const cuenta = useCuenta().data;
  const gimnasio = esGimnasio(cuenta);
  const titular = esTitular(cuenta);
  const miembros = useMiembros(gimnasio && titular);

  if (!cuenta) return <Esqueleto />;
  if (!gimnasio) return <FueraDeVersion titulo="Equipo" />;
  if (!titular) return <FueraDeVersion titulo="Equipo" mensaje="Solo el titular del gimnasio administra el equipo." />;

  return (
    <>
      <Encabezado titulo="Equipo" rotulo={cuenta.nombre} />
      <div className="grid gap-4 lg:grid-cols-2">
        <section aria-label="Tu versión" className="rounded-md border border-borde bg-superficie p-4">
          <h2 className="font-display text-xl font-semibold">Tu versión</h2>
          <p className="mt-2">{NOMBRE_VERSION[cuenta.version]} · {decimal(cuenta.funciones.horas_mes, 0)} h de video por mes</p>
          <p className="mt-1 text-sm text-texto-3">{textos.envivoNoIncluido}</p>
          {cuenta.horas && <Horas horas={cuenta.horas} className="mt-3" />}
        </section>

        <section aria-label="Miembros" className="rounded-md border border-borde bg-superficie p-4">
          <h2 className="font-display text-xl font-semibold">Miembros</h2>
          {miembros.isPending ? (
            <Esqueleto filas={2} />
          ) : miembros.isError ? (
            <ErrorConReintento error={miembros.error} alReintentar={() => miembros.refetch()} />
          ) : (
            <ul className="mt-2 divide-y divide-borde-suave">
              {miembros.data.map((m) => (
                <li key={m.usuario_id} className="flex min-h-14 items-center justify-between gap-3 py-2">
                  <span>
                    <span className="block font-medium">{m.nombre ?? "—"}</span>
                    {/* Con email solo fuera de la version de prueba, donde la API lo manda. */}
                    {!cuenta.demo && m.email && <span className="block text-xs text-texto-3">{m.email}</span>}
                  </span>
                  <span className="text-right text-sm text-texto-2">
                    {NOMBRE_ROL[m.rol]}
                    {m.desde && <span className="block text-xs text-texto-3">desde {fechaCorta(m.desde)}</span>}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-3 text-sm text-texto-3">{textos.demoEquipo}</p>
        </section>

        <section aria-label="Consentimientos" className="rounded-md border border-borde bg-superficie p-4 lg:col-span-2">
          <h2 className="font-display text-xl font-semibold">Consentimientos</h2>
          <p className="mt-1 text-sm text-texto-2">El registro de cada atleta: quién lo cargó, cuándo y si se revocó.</p>
          <DescargarCsv />
        </section>
      </div>
    </>
  );
}

/* El CSV pide el token, que un enlace no manda: se baja con fetch, como el export (W8). */
function DescargarCsv() {
  const [bajando, setBajando] = useState(false);
  async function bajar() {
    setBajando(true);
    try {
      const r = await fetch("/cuenta/consentimientos.csv", { headers: { Authorization: `Bearer ${leerToken()}` } });
      if (!r.ok) throw new Error(`no se pudo descargar (error ${r.status})`);
      const url = URL.createObjectURL(await r.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = "consentimientos.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      toast.error(mensajeDeError(e));
    } finally {
      setBajando(false);
    }
  }
  return (
    <Button variant="outline" className="mt-3 min-h-11" disabled={bajando} onClick={bajar}>
      {bajando ? "Descargando…" : "Descargar el registro (CSV)"}
    </Button>
  );
}
