"use client";

/*
 * Alta y edicion de un peleador, en una hoja desde abajo (04, "Lista de peleadores"). La
 * usa tambien la nueva sesion ("+ Agregar peleador") con los campos cortos.
 */
import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/animate-ui/components/radix/sheet";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SelectNativo } from "@/components/boxtwin/Controles";
import { useGuardarBoxeador } from "@/lib/api/consultas";
import type { Boxeador } from "@/lib/api/esquemas";
import { mensajeDeError } from "@/components/boxtwin/Estados";

function numeroODecimal(t: string): number | null {
  const s = t.trim().replace(",", ".");
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}

export function HojaPeleador({
  abierta, onCerrar, inicial, corta = false, onGuardado,
}: {
  abierta: boolean;
  onCerrar: () => void;
  inicial?: Boxeador | null;
  corta?: boolean;
  onGuardado?: (b: Boxeador) => void;
}) {
  const guardar = useGuardarBoxeador();
  const [nombre, setNombre] = useState(inicial?.nombre ?? "");
  const [categoria, setCategoria] = useState(inicial?.categoria ?? "");
  const [guardia, setGuardia] = useState<string>(inicial?.guardia ?? "");
  const [altura, setAltura] = useState(inicial?.altura_m != null ? String(inicial.altura_m).replace(".", ",") : "");
  const [envergadura, setEnvergadura] = useState(inicial?.envergadura_m != null ? String(inicial.envergadura_m).replace(".", ",") : "");
  const [notas, setNotas] = useState(inicial?.notas ?? "");
  // B17 / W24 (cambios-14-10): cargados a mano. Viajan siempre, porque el PATCH reemplaza.
  const [distancia, setDistancia] = useState<string>(inicial?.distancia ?? "");
  const [estilo, setEstilo] = useState<string>(inicial?.estilo ?? "");
  const [error, setError] = useState<string | null>(null);

  async function enviar(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const a = numeroODecimal(altura);
    const v = numeroODecimal(envergadura);
    if (!nombre.trim()) return setError("Falta el nombre.");
    if (a != null && (Number.isNaN(a) || a < 1.2 || a > 2.2)) return setError("La altura va en metros, entre 1,20 y 2,20.");
    if (v != null && (Number.isNaN(v) || v < 1.2 || v > 2.4)) return setError("La envergadura va en metros, entre 1,20 y 2,40.");
    try {
      const b = await guardar.mutateAsync({
        id: inicial?.id,
        nombre: nombre.trim(),
        guardia: guardia === "ortodoxa" || guardia === "zurda" ? guardia : null,
        categoria: categoria.trim() || null,
        notas: notas.trim() || null,
        altura_m: a,
        envergadura_m: v,
        distancia: distancia === "larga" || distancia === "media" || distancia === "corta" ? distancia : null,
        estilo: estilo === "fajador" || estilo === "estilista" ? estilo : null,
      });
      toast.success(inicial ? "Peleador guardado." : `${b.nombre} agregado.`);
      onGuardado?.(b);
      onCerrar();
    } catch (err) {
      setError(mensajeDeError(err));
    }
  }

  return (
    <Sheet open={abierta} onOpenChange={(o) => !o && onCerrar()}>
      <SheetContent side="bottom" className="h-auto max-h-[90dvh] overflow-y-auto bg-superficie lg:mx-auto lg:max-w-lg lg:rounded-t-lg">
        <SheetHeader>
          <SheetTitle className="font-display text-2xl">{inicial ? "Editar peleador" : "Agregar peleador"}</SheetTitle>
          <SheetDescription className="text-texto-3">
            {corta ? "Después podés completar altura y envergadura." : "Altura y envergadura son opcionales; sirven para el Rival Fantasma."}
          </SheetDescription>
        </SheetHeader>
        <form onSubmit={enviar} className="flex flex-col gap-4 px-4 pb-6">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="p-nombre">Nombre</Label>
            <Input id="p-nombre" value={nombre} onChange={(e) => setNombre(e.target.value)} className="h-11 text-base" autoComplete="off" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="p-categoria">Categoría</Label>
              <Input id="p-categoria" placeholder="Welter" value={categoria} onChange={(e) => setCategoria(e.target.value)} className="h-11 text-base" />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="p-guardia">Guardia</Label>
              <SelectNativo id="p-guardia" value={guardia} onChange={(e) => setGuardia(e.target.value)}>
                <option value="">Sin cargar</option>
                <option value="ortodoxa">Ortodoxa</option>
                <option value="zurda">Zurda</option>
              </SelectNativo>
            </div>
          </div>
          {!corta && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="p-altura">Altura (m)</Label>
                  <Input id="p-altura" inputMode="decimal" placeholder="1,75" value={altura} onChange={(e) => setAltura(e.target.value)} className="h-11 text-base" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="p-envergadura">Envergadura (m)</Label>
                  <Input id="p-envergadura" inputMode="decimal" placeholder="1,80" value={envergadura} onChange={(e) => setEnvergadura(e.target.value)} className="h-11 text-base" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="p-distancia">Distancia de trabajo</Label>
                  <SelectNativo id="p-distancia" value={distancia} onChange={(e) => setDistancia(e.target.value)}>
                    <option value="">Sin cargar</option>
                    <option value="larga">Larga</option>
                    <option value="media">Media</option>
                    <option value="corta">Corta</option>
                  </SelectNativo>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="p-estilo">Estilo</Label>
                  <SelectNativo id="p-estilo" value={estilo} onChange={(e) => setEstilo(e.target.value)}>
                    <option value="">Sin cargar</option>
                    <option value="fajador">Fajador</option>
                    <option value="estilista">Estilista</option>
                  </SelectNativo>
                </div>
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="p-notas">Notas</Label>
                <Input id="p-notas" value={notas} onChange={(e) => setNotas(e.target.value)} className="h-11 text-base" />
              </div>
            </>
          )}
          {error && <p role="alert" className="text-sm text-error">{error}</p>}
          <Button type="submit" disabled={guardar.isPending} className="min-h-11 text-base">
            {guardar.isPending ? "Guardando…" : "Guardar"}
          </Button>
        </form>
      </SheetContent>
    </Sheet>
  );
}
