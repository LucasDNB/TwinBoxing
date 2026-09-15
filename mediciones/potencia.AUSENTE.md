# E-03 `potencia.csv` — no existe, y por qué

La especificación pide las muestras crudas de potencia. No hay archivo porque no hay
muestras, y un CSV con sólo el encabezado se leería después como una medición vacía.

`nvidia-smi` no arranca en esta máquina. El módulo del kernel quedó en 595.84 y la
biblioteca NVML en 595.91, tras una actualización de driver sin reiniciar. CUDA computa
igual: las 24 corridas de esta campaña usaron esa misma GPU. Pero NVML es la única vía de
telemetría de potencia, y verifiqué que no hay alternativa en esta máquina: no aparece un
sensor de la placa por `hwmon` ni por sysfs, sólo `nvme` y `coretemp`.

Se resuelve reiniciando. Para confirmarlo:

```bash
python scripts/medicion_potencia.py --verificar
```

Cuando diga `OK`, la campaña se vuelve a correr con el muestreador en paralelo y S-17
queda cerrado. El muestreador ya está escrito, toma muestras cada 0,2 s con marca de
tiempo en el mismo reloj que usa el arnés de etapas, y se niega a escribir nada si NVML
no responde.

Hasta entonces, los 250 W del Capítulo 5 siguen siendo un supuesto sin respaldo, y así
está escrito en el texto que entrego.
