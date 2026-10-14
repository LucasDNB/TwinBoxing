# 02 · Rival Fantasma (pantalla 07)

Tareas: B18 (API), W25 y W26 (web). Depende de B16. Decisiones en `00-README.md`.

---

## 1. B18 · Ataque contra exposición (DC-8)

Módulo puro `boxtwin-api/src/boxtwin_api/rival.py` y endpoint `GET /rival?a={id}&b={id}`, que
arma los dos perfiles filtrados a sparring y combate y cruza sus bloques de B16. La regla vive
en Python y la web no la copia (como las combinaciones en W11).

### El cruce

Para cada sentido (A ataca a B, B ataca a A) y cada una de las cuatro zonas:

- `ataque_fraccion`: fracción corregida de los golpes del atacante dirigidos a esa zona.
- `exposicion_fraccion`: fracción corregida de lo que los rivales del receptor le dirigen a esa
  zona.
- `n_ataque`, `n_exposicion`: los conteos crudos.

El cruce es directo, zona por zona, sin espejar: la zona "cabeza por la mano izquierda del
rival" de la exposición de B es la misma a la que A tira con su mano izquierda (DC-2). Las dos
mitades usan la corrección por sesgo del detector (DC-4), para que el sesgo no aparezca como
diferencia entre peleadores.

### Las lecturas

Las tres reglas de zona de `ampliacion-13-10/05` §2, evaluadas en los dos sentidos:

1. Zona coincidente: el atacante dirige a la zona al menos 25% y al receptor le dirigen ahí al
   menos 25% → "{atacante} suele ir donde a {receptor} suelen buscarlo: {zona}."
2. Zona que el atacante trabaja más de lo que el receptor suele ver: la fracción del atacante
   supera en 10 puntos o más a la que el receptor ve → "{atacante} trabaja {zona} más de lo que
   {receptor} suele ver."
3. Cuerpo: el atacante dirige al cuerpo al menos 25% y al receptor le dirigen al cuerpo menos de
   15% → "{receptor} no suele recibir trabajo al cuerpo y {atacante} lo hace."

Cada lectura lleva `para` y `tipo`: si el atacante es A, es una oportunidad para A y una amenaza
para B, y al revés. La pantalla las muestra del lado de cada peleador. Cada lectura cita su zona y
sus dos fracciones.

Mínimo de datos: con menos de 50 golpes con altura en cualquiera de las cuatro mitades (ataque y
exposición de cada uno), no se dispara ninguna regla y `faltan` dice cuántos hay:
"Faltan sesiones para cruzar zonas: {n} y {m} golpes, hacen falta 50 de cada lado."

Las lecturas no dicen "va a conectar", "es vulnerable" ni dan probabilidades.

### Respuesta

```json
{
  "cruces": [{"atacante": "A", "zona": "cuerpo_mano_der", "ataque_fraccion": 0.24,
              "exposicion_fraccion": 0.31, "n_ataque": 88, "n_exposicion": 91}],
  "lecturas": [{"tipo": "oportunidad", "para": "A", "atacante": "A", "regla": "zona_coincidente",
                "texto": "…", "zona": "cuerpo_mano_der"}],
  "base": {"A": {"ataque": 412, "exposicion": 300}, "B": {"ataque": 233, "exposicion": 190}},
  "faltan": null,
  "sesgo_detector": {"recall_cabeza": 0.512, "recall_cuerpo": 0.348, "corregido": true}
}
```

Con la bandera de altura apagada, `cruces` y `lecturas` vacíos. El mismo peleador en A y B da 422.

### Pruebas (B18)

1. Bandera apagada: vacíos.
2. 49 golpes en una mitad: sin lecturas y `faltan` con los cuatro números.
3. Cada lectura cita una zona cuyos dos valores están en `cruces`.
4. Una regla con A atacante es oportunidad para A y amenaza para B (las dos filas).
5. Las tres reglas, cada una con un caso que la dispara y uno que no, en los umbrales justos
   (25% dispara; 24,9% no).
6. `a == b` da 422.

## 2. W26 · Bloque unificado

- "Oportunidades y amenazas" y "Dónde ataca uno y dónde recibe el otro" desaparecen como bloques.
  En su lugar, uno solo: "Ataque contra exposición", en estado estimación.
- Arriba, el selector "A ataca / B ataca" (pestañas de 44 px). Debajo, las dos siluetas del mapa
  (W22): la del atacante con sus cuatro zonas de ataque como flechas desde las manos, con grosor
  por fracción, que terminan antes de la silueta del otro (CA-16.3); la del receptor con el
  relleno de su exposición. En celular, una arriba de la otra.
- Tabla accesible de las ocho filas (cuatro zonas por sentido).
- Debajo, las lecturas en dos columnas, una por peleador: oportunidades en verde y amenazas en
  rojo, cada una con su zona y sus dos porcentajes. Sin lecturas: "Sin diferencias de zona que
  superen los umbrales." Con datos de menos: el texto de `faltan`.
- Frase fija: "Lectura cualitativa entre perfiles ya construidos. No es un pronóstico ni una
  probabilidad de victoria."
- `lib/rival.ts` y sus pruebas se borran: las reglas de métricas de W12 ya no se muestran (DC-8).
  La tabla "Frente a frente" queda como está, más dos filas cargadas a mano: distancia y estilo.

## 3. W25 · Dimensiones reales con avatares 3D (DC-9)

- Reemplaza `Siluetas` (de palitos) por dos `Avatar3D` enfrentados en el mismo lienzo, A a la
  izquierda mirando a la derecha y B al revés, cada uno a escala de sus medidas y con su guardia.
- Por avatar, dos controles deslizantes (brazo adelantado y atrasado), de 0% (en guardia) a 100%
  (extendido), con el alcance en metros al lado: hombro a puño, calculado con la envergadura.
- "Lado a lado" y "Superpuestas" se mantienen: superpuestas pone los dos en el mismo lugar,
  semitransparentes, para comparar talla y alcance.
- Distancia entre los dos: la suma de los dos alcances con los brazos extendidos, rotulada
  "distancia de presentación, no medida".
- Rótulo "ilustrativo, no reconstruido del video". Sin WebGL, la tabla de medidas.

## 4. Criterios de aceptación

- CA-25.1. Mover un control cambia el alcance escrito y no mueve al otro avatar.
- CA-25.2. La ruta del Rival sigue debajo de 380 KB de JS inicial.
- CA-26.1. Con la bandera de altura apagada, el bloque dice "No disponible" y no aparece ninguna
  zona.
- CA-26.2. Ninguna lectura aparece con menos de 50 golpes por mitad.
- CA-26.3. Ninguna lectura usa "conectar", "vulnerable" ni da una probabilidad.
- CA-26.4. Elegir el mismo peleador en A y B sigue sin estar permitido (CA-07.2).
