/*
 * Curvas de golpes por bloque de 10 s, superpuestas. SVG plano, sin libreria: dos o tres
 * series y nada mas. Cada serie lleva su nombre en la leyenda y en el aria-label.
 */
type Serie = { nombre: string; valores: number[]; color: string };

export function Curva({ series, titulo = "Golpes detectados cada 10 s" }: { series: Serie[]; titulo?: string }) {
  const n = Math.max(1, ...series.map((s) => s.valores.length));
  const max = Math.max(1, ...series.flatMap((s) => s.valores));
  const W = 320, H = 120, P = 4;
  const x = (i: number) => P + (i / Math.max(1, n - 1)) * (W - 2 * P);
  const y = (v: number) => H - P - (v / max) * (H - 2 * P);
  const desc = series.map((s) => `${s.nombre}: ${s.valores.reduce((a, b) => a + b, 0)} golpes`).join("; ");
  return (
    <figure className="rounded-lg border border-borde bg-superficie p-3">
      <figcaption className="rotulo mb-2">{titulo}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-32 w-full" role="img" aria-label={`${titulo}. ${desc}`}>
        <line x1={P} x2={W - P} y1={H - P} y2={H - P} stroke="var(--color-borde)" />
        {series.map((s) => (
          <polyline key={s.nombre} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round"
            points={s.valores.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
        ))}
      </svg>
      <ul className="mt-2 flex flex-wrap gap-3 text-xs text-texto-3">
        {series.map((s) => (
          <li key={s.nombre} className="flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-0.5 w-4" style={{ background: s.color }} />{s.nombre}
          </li>
        ))}
      </ul>
    </figure>
  );
}
