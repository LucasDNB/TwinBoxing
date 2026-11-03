"use client";
import { useEffect } from "react";

/* Registra el service worker en produccion. En desarrollo no: cachearia el shell de Next dev. */
export function RegistroSW() {
  useEffect(() => {
    if (process.env.NODE_ENV === "production" && "serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => undefined);
    }
  }, []);
  return null;
}
