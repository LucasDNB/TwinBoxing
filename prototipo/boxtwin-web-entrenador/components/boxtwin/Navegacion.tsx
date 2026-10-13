"use client";

/*
 * Navegacion (03 §3). Base y md: barra inferior fija de cinco destinos con "Subir" al centro
 * destacado, respetando el area segura del iPhone. lg en adelante: barra lateral de 232 px
 * con la marca arriba y el usuario abajo. El bloque del plan no existe (D4).
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
import { cn } from "@/lib/utils";
import { cerrarSesion } from "@/lib/sesion-usuario";

export const DESTINOS = [
  { href: "/", nombre: "Panel", Icono: LayoutDashboard, activo: (p: string) => p === "/" },
  { href: "/sesiones/", nombre: "Sesiones", Icono: ChartColumn, activo: (p: string) => p.startsWith("/sesiones") && !p.startsWith("/sesiones/nueva") || p.startsWith("/sesion/") },
  { href: "/sesiones/nueva/", nombre: "Subir", Icono: CloudUpload, activo: (p: string) => p.startsWith("/sesiones/nueva") },
  { href: "/peleadores/", nombre: "Peleadores", Icono: Users, activo: (p: string) => p.startsWith("/peleador") },
  { href: "/rival/", nombre: "Rival", Icono: User, activo: (p: string) => p.startsWith("/rival") },
] as const;

export function BarraInferior() {
  const ruta = usePathname() || "/";
  return (
    <nav
      aria-label="Principal"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-borde-suave bg-nav/95 backdrop-blur lg:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <ul className="mx-auto grid max-w-lg grid-cols-5">
        {DESTINOS.map(({ href, nombre, Icono, activo }) => {
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

export function BarraLateral({ usuario }: { usuario: string }) {
  const ruta = usePathname() || "/";
  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-[232px] flex-col border-r border-borde-suave bg-nav lg:flex">
      <div className="px-5 py-6">
        <Link href="/" aria-label="BoxTwin, ir al panel">
          <Marca />
        </Link>
        <p className="rotulo mt-2">versión entrenador</p>
      </div>
      <nav aria-label="Principal" className="flex-1 px-3">
        <ul className="flex flex-col gap-1">
          {DESTINOS.map(({ href, nombre, Icono, activo }) => {
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
                    {nombre === "Subir" ? "Subir sesión" : nombre === "Rival" ? "Rival Fantasma" : nombre}
                  </Link>
                </AnimateIcon>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="border-t border-borde-suave px-5 py-4">
        <p className="truncate text-sm text-texto-2" title={usuario}>{usuario}</p>
        <button
          type="button"
          className="mt-1 min-h-11 text-sm text-tenue underline-offset-4 hover:underline"
          onClick={() => {
            cerrarSesion();
            // Recarga completa: que no quede en memoria nada del usuario que salio.
            // eslint-disable-next-line @next/next/no-location-assign-relative-destination
            window.location.assign("/ingresar/");
          }}
        >
          Salir
        </button>
      </div>
    </aside>
  );
}
