import type { Consejos as C } from "@/lib/api";

export function Consejos({ c }: { c: C }) {
  return (
    <section aria-labelledby="t-consejos" className="rounded-lg border border-borde bg-superficie p-3">
      <h2 id="t-consejos" className="rotulo mb-2">Consejos</h2>
      <ul className="space-y-2">
        {c.consejos.map((x) => <li key={x.clave} className="text-texto">{x.texto}</li>)}
      </ul>
      <p className="mt-3 text-xs text-tenue">{c.pie}</p>
    </section>
  );
}
