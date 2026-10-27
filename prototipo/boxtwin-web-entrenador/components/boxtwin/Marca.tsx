/* La marca de BoxTwin: dos cuadros, uno de cada esquina. */
export function Marca({ chica = false }: { chica?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2 font-display tracking-wide text-texto">
      <svg aria-hidden="true" width={chica ? 18 : 22} height={chica ? 18 : 22} viewBox="0 0 22 22">
        <rect x="1" y="1" width="9" height="20" rx="1.5" className="fill-a" />
        <circle cx="16.5" cy="11" r="4.5" className="fill-b" />
      </svg>
      <span className={chica ? "text-lg font-bold" : "text-2xl font-bold"}>BOXTWIN</span>
    </span>
  );
}
