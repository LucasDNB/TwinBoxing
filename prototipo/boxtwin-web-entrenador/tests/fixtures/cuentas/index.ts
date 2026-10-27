/*
 * Una cuenta por version y rol (versiones 26-10, 05 §4), con la forma de GET /cuenta. La de
 * Entrenador es la que ven todas las pruebas anteriores: con ella la web se comporta igual que
 * antes de las versiones (compuerta de paridad).
 */
const horas = (incluidas: number, usadas = 1.2) => ({
  mes: "2026-10", incluidas, usadas, restantes: Math.max(incluidas - usadas, 0), sin_duracion: 0,
  modo: "informar" as "informar" | "bloquear",
});

const funciones = {
  individual: { horas_mes: 4, miembros_max: 1, peleadores: "propio", rival: false, invitaciones: false, plantel_compartido: false, consentimiento: "sesion", en_vivo: false },
  entrenador: { horas_mes: 15, miembros_max: 1, peleadores: "libre", rival: true, invitaciones: false, plantel_compartido: false, consentimiento: "sesion", en_vivo: false },
  gimnasio: { horas_mes: 50, miembros_max: null, peleadores: "libre", rival: true, invitaciones: true, plantel_compartido: true, consentimiento: "atleta", en_vivo: false },
} as const;

export function cuentaEntrenador() {
  return {
    id: "c-ent", nombre: "lucas@usal.edu.ar", version: "entrenador", rol: "titular",
    usuario: { id: "u-lucas", email: "lucas@usal.edu.ar", nombre: null },
    boxeador_propio: null, funciones: funciones.entrenador, horas: horas(15), demo: false, perfil_demo: null,
  };
}

export function cuentaIndividual() {
  return {
    id: "c-ind", nombre: "Individual de prueba", version: "individual", rol: "titular",
    usuario: { id: "u-ind", email: null, nombre: "Lucas" },
    boxeador_propio: { id: "propio", nombre: "Lucas" }, funciones: funciones.individual,
    horas: horas(4), demo: true, perfil_demo: { perfil: "individual", nombre: "Individual" },
  };
}

export function cuentaGimnasio(rol: "titular" | "entrenador" = "titular") {
  return {
    id: "c-gim", nombre: "Gimnasio de prueba", version: "gimnasio", rol,
    usuario: rol === "titular" ? { id: "u-tit", email: null, nombre: "Tita" } : { id: "u-ana", email: null, nombre: "Ana" },
    boxeador_propio: null, funciones: funciones.gimnasio, horas: horas(50, 12.4), demo: true,
    perfil_demo: { perfil: rol === "titular" ? "gimnasio_titular" : "gimnasio_entrenador", nombre: rol === "titular" ? "Gimnasio · titular" : "Gimnasio · entrenador" },
  };
}

export type CuentaDePrueba = ReturnType<typeof cuentaEntrenador> | ReturnType<typeof cuentaIndividual> | ReturnType<typeof cuentaGimnasio>;
