/*
 * next/navigation sin el App Router: las pantallas corren en jsdom y la "navegacion" cambia
 * window.location con history, que es lo que leen useSearchParams y usePathname.
 */
import { vi } from "vitest";

export const navegacion = {
  historial: [] as string[],
};

function ir(url: string) {
  navegacion.historial.push(url);
  window.history.pushState({}, "", url);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

vi.mock("next/navigation", async () => {
  const React = await import("react");
  function useUbicacion() {
    const [, setN] = React.useState(0);
    React.useEffect(() => {
      const f = () => setN((n) => n + 1);
      window.addEventListener("popstate", f);
      return () => window.removeEventListener("popstate", f);
    }, []);
    return window.location;
  }
  return {
    useRouter: () => ({
      push: (u: string) => ir(u),
      replace: (u: string) => ir(u),
      back: () => window.history.back(),
      prefetch: () => {},
      refresh: () => {},
    }),
    usePathname: () => useUbicacion().pathname,
    useSearchParams: () => new URLSearchParams(useUbicacion().search),
  };
});

vi.mock("next/link", async () => {
  const React = await import("react");
  return {
    default: React.forwardRef<HTMLAnchorElement, React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }>(
      function Enlace({ href, children, onClick, ...resto }, ref) {
        return React.createElement(
          "a",
          {
            ...resto,
            ref,
            href,
            onClick: (e: React.MouseEvent<HTMLAnchorElement>) => {
              onClick?.(e);
              if (e.defaultPrevented) return;
              e.preventDefault();
              ir(href);
            },
          },
          children,
        );
      },
    ),
  };
});
