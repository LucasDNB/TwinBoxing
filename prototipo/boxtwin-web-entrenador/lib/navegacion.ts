/*
 * Destino seguro despues del ingreso: solo rutas propias. Un ?siguiente= que apunte afuera
 * ("//otro.com", "https://...") se ignora, para que el ingreso no sirva de redireccion
 * abierta.
 */
export function destinoSeguro(siguiente: string | null | undefined): string {
  if (!siguiente) return "/";
  if (!siguiente.startsWith("/") || siguiente.startsWith("//") || siguiente.startsWith("/\\")) return "/";
  if (siguiente.startsWith("/ingresar")) return "/";
  return siguiente;
}

/* Lee un parametro de la query en el cliente. */
export function param(nombre: string): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get(nombre);
}
