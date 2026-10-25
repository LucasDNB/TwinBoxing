# Enmienda al pre-registro de E7 · modo académico en la app

**Estado: borrador del 25-10-2026, pendiente de aprobación de Lucas.** Se commitea aprobada,
sola, antes de escribir código del producto. No edita `docs/specs/2026-10-24-e7-4dhumans/01-preregistro.md`:
lo enmienda desde acá, con fecha y motivo.

## Qué enmienda

El pre-registro de E7, §2.1, dice: "E7 es académica. Ningún modelo ni feature de E7 entra al
producto. Si E7 pasa, llevarlo al producto es otro spec y empieza por la licencia comercial de
SMPL o por un estimador sin SMPL."

El 25-10, con E7 pasado y auditado, Lucas eligió la opción B: E7 entra al código de la app como
un modo académico, apagado por defecto, para mostrarlo en la presentación del 17-11. Este spec es
el "otro spec" que el §2.1 pedía. La licencia comercial de SMPL no se tramita: el modo se queda
dentro del uso no comercial.

## Condiciones (no se negocian)

1. **Apagado por defecto.** Solo se enciende con `BOXTWIN_CLASIFICADOR=e7`. Cualquier otro valor
   o la ausencia de la variable deja el comportamiento de hoy (E4).
2. **Nunca para usuarios.** No se enciende en la estación mientras haya sesiones de prueba,
   entrenadores u otros usuarios cargando videos. Solo en una instancia de demostración o en la
   presentación, con material propio.
3. **Licencia confirmada antes de encender.** Lucas pega en `TAREAS.md` el texto de la licencia
   de SMPLify que aceptó al bajar el modelo, y el modo solo se enciende si ese texto permite este
   uso (demostración académica de un PFI, sin cobro ni distribución del modelo).
4. **Tiempo medido antes de encender.** El modo se mide contra C5 (reloj hasta la Fight-Card,
   umbral 2x) en al menos dos sesiones. Si no entra en 2x, se puede encender igual para la demo,
   pero la Fight-Card y la tesis lo dicen, y C5 se informa con el modo apagado.
5. **La Fight-Card lo declara.** Método `geometrico-e7`, exactitud de familia 0,777 con una
   fuente afuera, y la leyenda "modo académico: 3D de 4D-Humans con SMPL, licencia de
   investigación".
6. **Nada de SMPL ni de HMR2 entra a git ni a una imagen del worker.** El modo usa el entorno
   `boxtwin_4dhumans` de la estación.
7. **El MVP no cambia.** Los criterios de aceptación de `docs/specs/MVP.md` se miden y se informan
   con E4.

## Lo que no se cambia

Los resultados de E7 y su lectura. Esta enmienda no toca ningún número ni ninguna regla de
lectura: solo autoriza que el código del modo académico exista en el producto, con las
condiciones de arriba.
