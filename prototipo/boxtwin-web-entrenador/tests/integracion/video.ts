/*
 * Un <video> que jsdom no implementa: currentTime asignable, paused y play() que avisan.
 */
export const reproduccion = { reproducciones: 0 };

export function videoSimulado() {
  const tiempos = new WeakMap<HTMLMediaElement, number>();
  const pausado = new WeakMap<HTMLMediaElement, boolean>();
  Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
    configurable: true,
    get() { return tiempos.get(this) ?? 0; },
    set(v: number) { tiempos.set(this, v); },
  });
  Object.defineProperty(HTMLMediaElement.prototype, "paused", {
    configurable: true,
    get() { return pausado.get(this) ?? true; },
  });
  HTMLMediaElement.prototype.play = function () {
    pausado.set(this, false);
    reproduccion.reproducciones += 1;
    return Promise.resolve();
  };
  HTMLMediaElement.prototype.pause = function () { pausado.set(this, true); };
}
