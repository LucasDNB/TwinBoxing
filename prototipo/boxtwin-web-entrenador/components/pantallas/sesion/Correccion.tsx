"use client";

/*
 * Correccion del tipo de golpe (04 §04, punto 4; W7) y, con la bandera `objetivo`, de la
 * altura (W14): el par cabeza / cuerpo debajo del tipo. Hoja desde abajo en el celular y
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
import { useCorregir, useCorregirAltura } from "@/lib/api/consultas";
import { TIPOS_GOLPE, type Golpe, type TipoCorreccion } from "@/lib/api/esquemas";
import { tipoEfectivo } from "@/lib/fightcard";
import { mmss, nombreDeTipo } from "@/lib/formato";
import { useEsEscritorio } from "@/lib/medios";
import { cn } from "@/lib/utils";

type Altura = "cabeza" | "cuerpo";

function Opciones({ golpe, nomenclatura, conAltura, onGuardar, guardando }: {
  golpe: Golpe; nomenclatura: Record<string, string>; conAltura: boolean;
  onGuardar: (c: { tipo?: TipoCorreccion; objetivo?: Altura }) => void; guardando: boolean;
}) {
  const actual = golpe.descartado ? "no_golpe" : tipoEfectivo(golpe) ?? "";
  const [elegido, setElegido] = useState<string>(actual);
  const alturaActual = golpe.objetivo ?? "";
  const [altura, setAltura] = useState<string>(alturaActual);
  // Un golpe que no fue golpe no tiene altura: el par se esconde si se elige "No fue un golpe".
  const muestraAltura = conAltura && elegido !== "no_golpe";
  const cambiaTipo = !!elegido && elegido !== actual;
  const cambiaAltura = muestraAltura && !!altura && altura !== alturaActual;
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
      {muestraAltura && (
        <div role="radiogroup" aria-label="Altura del golpe" className="grid grid-cols-2 gap-2">
          {(["cabeza", "cuerpo"] as const).map((a) => (
            <button
              key={a}
              type="button"
              role="radio"
              aria-checked={altura === a}
              onClick={() => setAltura(a)}
              className={cn(
                "min-h-11 rounded-md border px-3 text-left text-[15px]",
                altura === a ? "border-texto bg-elevado" : "border-borde bg-superficie-2",
              )}
            >
              {a === "cabeza" ? "A la cabeza" : "Al cuerpo"}
            </button>
          ))}
        </div>
      )}
      <Button
        className="min-h-11"
        disabled={(!cambiaTipo && !cambiaAltura) || guardando}
        onClick={() => onGuardar({
          tipo: cambiaTipo ? (elegido as TipoCorreccion) : undefined,
          objetivo: cambiaAltura ? (altura as Altura) : undefined,
        })}
      >
        Guardar
      </Button>
    </div>
  );
}

export function Correccion({ sesionId, golpe, nomenclatura, conAltura = false }: {
  sesionId: string; golpe: Golpe; nomenclatura: Record<string, string>; conAltura?: boolean;
}) {
  const escritorio = useEsEscritorio();
  const [abierta, setAbierta] = useState(false);
  const corregir = useCorregir(sesionId);
  const corregirAltura = useCorregirAltura(sesionId);
  const avisos = {
    onSuccess: () => toast.success("Corrección guardada."),
    onError: (e: unknown) => toast.error(`No se pudo guardar la corrección: ${mensajeDeError(e)}`),
  };

  // La API toma una correccion por pedido (tipo u objetivo): si cambian los dos, van dos.
  function guardar({ tipo, objetivo }: { tipo?: TipoCorreccion; objetivo?: Altura }) {
    setAbierta(false);
    if (tipo) corregir.mutate({ golpe: golpe.id, tipo }, avisos);
    if (objetivo) corregirAltura.mutate({ golpe: golpe.id, objetivo }, avisos);
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
  const contenido = (
    <Opciones
      golpe={golpe}
      nomenclatura={nomenclatura}
      // Una sesion enriquecida trae el campo aunque sea null (sin dato): ahi tambien se puede marcar.
      conAltura={conAltura && golpe.objetivo !== undefined}
      onGuardar={guardar}
      guardando={corregir.isPending || corregirAltura.isPending}
    />
  );

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
