import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// jsdom no trae lo que Motion y Radix piden para medir y observar.
class Observador {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}
const g = globalThis as unknown as Record<string, unknown>;
g.IntersectionObserver ??= Observador;
g.ResizeObserver ??= Observador;
if (typeof window !== "undefined") {
  window.matchMedia ??= ((q: string) => ({
    matches: false,
    media: q,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
  window.scrollTo ??= (() => {}) as typeof window.scrollTo;
  Element.prototype.scrollIntoView ??= function () {};
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.releasePointerCapture ??= () => {};
  HTMLMediaElement.prototype.play = function () {
    return Promise.resolve();
  };
  HTMLMediaElement.prototype.pause = function () {};
}
