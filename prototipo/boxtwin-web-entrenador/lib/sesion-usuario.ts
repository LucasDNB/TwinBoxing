/*
 * El token y el usuario, en localStorage.
 *
 * POR QUE TOLERA ERRORES
 *   Safari en modo privado tira al leer o escribir localStorage. Sin persistencia la app
 *   pide ingresar otra vez, que es molesto pero no roto. Es la misma tolerancia que tiene
 *   boxtwin-web/src/api.js (02 §6).
 */
const CLAVE_TOKEN = "boxtwin.token";
const CLAVE_USUARIO = "boxtwin.usuario";

export function leerToken(): string {
  try {
    return localStorage.getItem(CLAVE_TOKEN) || "";
  } catch {
    return "";
  }
}

export function leerUsuario(): string {
  try {
    return localStorage.getItem(CLAVE_USUARIO) || "";
  } catch {
    return "";
  }
}

export function guardarSesion(token: string, email: string): void {
  try {
    localStorage.setItem(CLAVE_TOKEN, token);
    localStorage.setItem(CLAVE_USUARIO, email);
  } catch {
    /* sin persistencia: la sesion dura lo que dure la pestana */
  }
}

export function cerrarSesion(): void {
  try {
    localStorage.removeItem(CLAVE_TOKEN);
    localStorage.removeItem(CLAVE_USUARIO);
  } catch {
    /* nada que limpiar */
  }
}
