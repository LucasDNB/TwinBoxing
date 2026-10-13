"use client";

/*
 * El aviso "En construccion" (13-10): la funcion se muestra en el prototipo, y al intentar
 * usarla se avisa en vez de hacer algo a medias.
 */
import {
  AlertDialog, AlertDialogAction, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/animate-ui/components/radix/alert-dialog";
import { textos } from "@/lib/textos";

export function AvisoEnConstruccion({ abierto, onCerrar, texto }: { abierto: boolean; onCerrar: () => void; texto: string }) {
  return (
    <AlertDialog open={abierto} onOpenChange={(o) => !o && onCerrar()}>
      <AlertDialogContent className="bg-superficie">
        <AlertDialogHeader>
          <AlertDialogTitle>{textos.enConstruccion.titulo}</AlertDialogTitle>
          <AlertDialogDescription className="text-texto-2">{texto}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction className="min-h-11" onClick={onCerrar}>Entendido</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
