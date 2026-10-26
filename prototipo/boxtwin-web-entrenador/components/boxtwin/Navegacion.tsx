"use client";

/*
 * Navegacion (03 §3). Base y md: barra inferior fija de cinco destinos con "Subir" al centro
 * destacado, respetando el area segura del iPhone. lg en adelante: barra lateral de 232 px
 * con la marca arriba y el usuario abajo.
 *
 * Versiones 26-10 (03 §1, W27): los destinos salen de la cuenta. Individual tiene cuatro;
 * el titular de Gimnasio tiene seis en la barra lateral y cinco en la inferior, porque la
 * barra del celular no admite seis sin achicar los toques por debajo de 44 px: llega al Rival
 * Fantasma desde el perfil.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard } from "@/components/animate-ui/icons/layout-dashboard";
import { ChartColumn } from "@/components/animate-ui/icons/chart-column";
import { CloudUpload } from "@/components/animate-ui/icons/cloud-upload";
import { Users } from "@/components/animate-ui/icons/users";
import { User } from "@/components/animate-ui/icons/user";
import { AnimateIcon } from "@/components/animate-ui/icons/icon";
import { Marca } from "@/components/boxtwin/Marca";
import { UserCog } from "lucide-react";
import { cn } from "@/lib/utils";
import { cerrarSesion } from "@/lib/sesion-usuario";
import type { Cuenta } from "@/lib/api/esquemas";
import { NOMBRE_VERSION, rotuloDemo } from "@/lib/cuenta";

type Destino = {
  href: string;
  nombre: string;
  largo?: string;
  Icono: React.ComponentType<{ size?: number }>;
  activo: (p: string) => boolean;
};

const PANEL: Destino = { href: "/", nombre: "Panel", Icono: LayoutDashboard, activo: (p) => p === "/" };
const SESIONES: Destino = { href: "/sesiones/", nombre: "Sesiones", Icono: ChartColumn, activo: (p) => p.startsWith("/sesiones") && !p.startsWith("/sesiones/nueva") || p.startsWith("/sesion/") };
const SUBIR: Destino = { href: "/sesiones/nueva/", nombre: "Subir", largo: "Subir sesión", Icono: CloudUpload, activo: (p) => p.startsWith("/sesiones/nueva") };
const PELEADORES: Destino = { href: "/peleadores/", nombre: "Peleadores", Icono: Users, activo: (p) => p.startsWith("/peleador") };
const RIVAL: Destino = { href: "/rival/", nombre: "Rival", largo: "Rival Fantasma", Icono: User, activo: (p) => p.startsWith("/rival") };

export const DESTINOS = [PANEL, SESIONES, SUBIR, PELEADORES, RIVAL] as const;

/* Los destinos de cada version y rol (03 §1). `lateral` son los de la barra de escritorio. */
export function destinosDe(cuenta: Cuenta | null | undefined): { inferior: Destino[]; lateral: Destino[] } {
  if (!cuenta || cuenta.version === "entrenador") return { inferior: [...DESTINOS], lateral: [...DESTINOS] };
  if (cuenta.version === "individual") {
    const propio = cuenta.boxeador_propio?.id;
    const perfil: Destino = {
      href: propio ? `/peleador/?id=${propio}` : "/", nombre: "Mi perfil", Icono: User,
      activo: (p) => p.startsWith("/peleador"),
    };
    const inicio: Destino = { ...PANEL, nombre: "Inicio" };
    const d = [inicio, SESIONES, SUBIR, perfil];
    return { inferior: d, lateral: d };
  }
  const plantel: Destino = { ...PELEADORES, nombre: "Plantel" };
  if (cuenta.rol === "titular") {
    const equipo: Destino = { href: "/equipo/", nombre: "Equipo", Icono: UserCog, activo: (p) => p.startsWith("/equipo") };
    return { inferior: [PANEL, SESIONES, SUBIR, plantel, equipo], lateral: [PANEL, SESIONES, SUBIR, plantel, RIVAL, equipo] };
  }
  const d = [PANEL, SESIONES, SUBIR, plantel, RIVAL];
  return { inferior: d, lateral: d };
}

function salir(): void {
  cerrarSesion();
  // Recarga completa: que no quede en memoria nada de la cuenta que salio (CA-00.10).
  // eslint-disable-next-line @next/next/no-location-assign-relative-destination
  window.location.assign("/ingresar/");
}

/*
 * En la version de prueba, arriba de todo y en todos los tamaños: de que cuenta es y "Salir"
 * para volver a los botones del ingreso (03 §2). En el celular no hay barra lateral.
 */
export function FranjaDemo({ cuenta }: { cuenta: Cuenta | null | undefined }) {
  const rotulo = rotuloDemo(cuenta);
  if (!rotulo) return null;
  return (
    <div className="sticky top-0 z-40 flex min-h-11 items-center justify-between gap-3 border-b border-estimado-borde bg-estimado-fondo px-4 text-sm lg:pl-[248px]">
      <span className="rotulo text-estimado" data-testid="rotulo-demo">{rotulo}</span>
      <button type="button" className="min-h-11 px-2 underline underline-offset-4" onClick={salir}>
        Salir
      </button>
    </div>
  );
}

export function BarraInferior({ cuenta }: { cuenta?: Cuenta | null }) {
  const ruta = usePathname() || "/";
  const destinos = destinosDe(cuenta).inferior;
  return (
    <nav
      aria-label="Principal"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-borde-suave bg-nav/95 backdrop-blur lg:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <ul className={cn("mx-auto grid max-w-lg", destinos.length === 4 ? "grid-cols-4" : "grid-cols-5")}>
        {destinos.map(({ href, nombre, Icono, activo }) => {
          const esta = activo(ruta);
          const subir = nombre === "Subir";
          return (
            <li key={href} className="flex justify-center">
              <AnimateIcon animateOnHover asChild>
                <Link
                  href={href}
                  aria-current={esta ? "page" : undefined}
                  className={cn(
                    "flex min-h-14 min-w-14 flex-col items-center justify-center gap-0.5 px-1 text-[11px]",
                    esta ? "text-texto" : "text-tenue",
                  )}
                >
                  <span
                    className={cn(
                      "flex items-center justify-center",
                      subir && "-mt-4 size-12 rounded-full border border-borde-fuerte bg-texto text-fondo shadow-lg",
                    )}
                  >
                    <Icono size={subir ? 22 : 20} />
                  </span>
                  <span>{nombre}</span>
                </Link>
              </AnimateIcon>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export function BarraLateral({ usuario, cuenta }: { usuario: string; cuenta?: Cuenta | null }) {
  const ruta = usePathname() || "/";
  const destinos = destinosDe(cuenta).lateral;
  const quien = cuenta?.demo ? (cuenta.usuario.nombre ?? rotuloDemo(cuenta) ?? "") : usuario;
  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-[232px] flex-col border-r border-borde-suave bg-nav lg:flex">
      <div className="px-5 py-6">
        <Link href="/" aria-label="BoxTwin, ir al panel">
          <Marca />
        </Link>
        <p className="rotulo mt-2">versión {(cuenta ? NOMBRE_VERSION[cuenta.version] : "Entrenador").toLowerCase()}</p>
        {cuenta && cuenta.version === "gimnasio" && <p className="mt-1 truncate text-sm text-texto-2">{cuenta.nombre}</p>}
      </div>
      <nav aria-label="Principal" className="flex-1 px-3">
        <ul className="flex flex-col gap-1">
          {destinos.map(({ href, nombre, largo, Icono, activo }) => {
            const esta = activo(ruta);
            return (
              <li key={href}>
                <AnimateIcon animateOnHover asChild>
                  <Link
                    href={href}
                    aria-current={esta ? "page" : undefined}
                    className={cn(
                      "flex min-h-11 items-center gap-3 rounded-md px-3 text-[15px]",
                      esta ? "bg-elevado text-texto" : "text-texto-2 hover:bg-superficie",
                    )}
                  >
                    <Icono size={18} />
                    {largo ?? nombre}
                  </Link>
                </AnimateIcon>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="border-t border-borde-suave px-5 py-4">
        <p className="truncate text-sm text-texto-2" title={quien}>{quien}</p>
        <button
          type="button"
          className="mt-1 min-h-11 text-sm text-tenue underline-offset-4 hover:underline"
          onClick={salir}
        >
          Salir
        </button>
      </div>
    </aside>
  );
}
