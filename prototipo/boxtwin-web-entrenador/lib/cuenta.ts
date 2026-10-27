"use client";

/*
 * La cuenta: version, rol y funciones (versiones 26-10, 03 §1, W27).
 *
 * POR QUE EXISTE
 *   Una version es el plan de la cuenta, no otra aplicacion. La web nunca decide que version
 *   es ni que funciones tiene: lo pregunta a GET /cuenta y dibuja, igual que con /producto.
 *   La API rechaza igual lo que la version no tiene (B22), asi que esto decide que se ve, no
 *   que se puede.
 */
import { useQuery } from "@tanstack/react-query";
import { pedirValidado } from "@/lib/api/cliente";
import { Cuenta, type Funciones, type Version } from "@/lib/api/esquemas";

export const CLAVE_CUENTA = ["cuenta"] as const;

export function useCuenta() {
  return useQuery({
    queryKey: CLAVE_CUENTA,
    queryFn: () => pedirValidado(Cuenta, "/cuenta"),
    staleTime: 30_000,
  });
}

export const NOMBRE_VERSION: Record<Version, string> = {
  individual: "Individual",
  entrenador: "Entrenador",
  gimnasio: "Gimnasio",
};

type FuncionBooleana = { [K in keyof Funciones]: Funciones[K] extends boolean ? K : never }[keyof Funciones];

/* Si la version de la cuenta tiene la funcion. Sin cuenta todavia, no. */
export function tiene(cuenta: Cuenta | null | undefined, funcion: FuncionBooleana): boolean {
  return !!cuenta?.funciones[funcion];
}

export const esIndividual = (c: Cuenta | null | undefined) => c?.version === "individual";
export const esGimnasio = (c: Cuenta | null | undefined) => c?.version === "gimnasio";
export const esTitular = (c: Cuenta | null | undefined) => c?.rol === "titular";

/* "Prueba · Gimnasio · entrenador" (03 §2). */
export function rotuloDemo(c: Cuenta | null | undefined): string | null {
  if (!c?.demo) return null;
  return `Prueba · ${c.perfil_demo?.nombre ?? NOMBRE_VERSION[c.version]}`;
}

/* Como se nombra a quien subio o corrigio algo: el nombre, y si no hay, el email. */
export function nombreDePersona(p: { nombre: string | null } | null | undefined): string | null {
  return p?.nombre ?? null;
}
