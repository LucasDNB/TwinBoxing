"use client";

/*
 * Correccion del tipo de golpe (04 §04, punto 4; W7). Hoja desde abajo en el celular y
 * popover en escritorio (03 §4). Los nombres salen de fightcard.nomenclatura (D7) y se suma
 * "No fue un golpe". La correccion es optimista: la lista cambia al guardar y vuelve atras
 * si la API responde error, con un aviso (02 §5).
 */
import { useState } from "react";
import { toast } from "sonner";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/animate-ui/components/radix/sheet";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/animate-ui/components/radix/popover";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { mensajeDeError } from "@/components/boxtwin/Estados";
import { useCorregir } from "@/lib/api/consultas";
import { TIPOS_GOLPE, type Golpe, type TipoCorreccion } from "@/lib/api/esquemas";
import { tipoEfectivo } from "@/lib/fightcard";
import { mmss, nombreDeTipo } from "@/lib/formato";
import { useEsEscritorio } from "@/lib/medios";
import { cn } from "@/lib/utils";

function Opciones({ golpe, nomenclatura, onGuardar, guardando }: { golpe: Golpe; nomenclatura: Record<string, string>; onGuardar: (t: TipoCorreccion) => void; guardando: boolean }) {
  const actual = golpe.descartado ? "no_golpe" : tipoEfectivo(golpe) ?? "";
  const [elegido, setElegido] = useState<string>(actual);
  const opciones: { valor: TipoCorreccion; nombre: string }[] = [
    ...TIPOS_GOLPE.map((t) => ({ valor: t, nombre: nombreDeTipo(t, nomenclatura)! })),
    { valor: "no_golpe", nombre: "No fue un golpe" },
  ];
  return (
    <div className="flex flex-col gap-3">
      <div role="radiogroup" aria-label="Tipo de golpe" className="grid grid-cols-2 gap-2">
        {opciones.map((o) => (
          <button
            key={o.valor}
            type="button"
            role="radio"
            aria-checked={elegido === o.valor}
            onClick={() => setElegido(o.valor)}
            className={cn(
              "min-h-11 rounded-md border px-3 text-left text-[15px]",
              o.valor === "no_golpe" && "col-span-2",
              elegido === o.valor ? "border-texto bg-elevado" : "border-borde bg-superficie-2",
            )}
          >
            {o.nombre}
          </button>
        ))}
      </div>
      <Button className="min-h-11" disabled={!elegido || elegido === actual || guardando} onClick={() => onGuardar(elegido as TipoCorreccion)}>
        Guardar
      </Button>
    </div>
  );
}

export function Correccion({ sesionId, golpe, nomenclatura }: { sesionId: string; golpe: Golpe; nomenclatura: Record<string, string> }) {
  const escritorio = useEsEscritorio();
  const [abierta, setAbierta] = useState(false);
  const corregir = useCorregir(sesionId);

  function guardar(tipo: TipoCorreccion) {
    setAbierta(false);
    corregir.mutate(
      { golpe: golpe.id, tipo },
      {
        onSuccess: () => toast.success("Corrección guardada."),
        onError: (e) => toast.error(`No se pudo guardar la corrección: ${mensajeDeError(e)}`),
      },
    );
  }

  const titulo = `Corregir el golpe de ${mmss(golpe.t_inicio)}`;
  const boton = (
    <button
      type="button"
      onClick={escritorio ? undefined : () => setAbierta(true)}
      aria-label={titulo}
      className="min-h-11 min-w-11 px-2 text-sm text-texto-2 underline underline-offset-4"
    >
      Corregir
    </button>
  );
  const contenido = <Opciones golpe={golpe} nomenclatura={nomenclatura} onGuardar={guardar} guardando={corregir.isPending} />;

  if (escritorio) {
    return (
      <Popover open={abierta} onOpenChange={setAbierta}>
        <PopoverTrigger asChild>{boton}</PopoverTrigger>
        <PopoverContent className="w-72 bg-superficie" align="end">
          <p className="mb-3 font-display text-lg font-semibold">{titulo}</p>
          {contenido}
        </PopoverContent>
      </Popover>
    );
  }
  return (
    <>
      {boton}
      <Sheet open={abierta} onOpenChange={setAbierta}>
        <SheetContent side="bottom" className="h-auto max-h-[85dvh] bg-superficie pb-[env(safe-area-inset-bottom)]">
          <SheetHeader>
            <SheetTitle className="font-display text-2xl">{titulo}</SheetTitle>
            <SheetDescription className="text-texto-3">
              Tu corrección se guarda para entrenar y no cambia el conteo de golpes detectados.
            </SheetDescription>
          </SheetHeader>
          <div className="px-4 pb-6">{contenido}</div>
        </SheetContent>
      </Sheet>
    </>
  );
}
