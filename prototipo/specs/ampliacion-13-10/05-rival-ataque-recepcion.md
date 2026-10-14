# 05 · Rival Fantasma: dónde ataca uno y dónde recibe el otro

Pedido (13-10): en el Rival Fantasma, cruzar dónde ataca un peleador con dónde recibe el otro.

Tareas: B13 (API), W16 (web). Esfuerzo: 1,5 días. Depende de B11 y B12.

---

## 1. Qué se cruza

La definición del Rival Fantasma en la tesis cruza "la distribución de golpes de un peleador contra
el mapa de exposición del otro". Con `03` y `04` existen las dos mitades, con las mismas cuatro
zonas:

- Ataque de A: a qué zonas dirige A sus golpes (sus golpes detectados, por altura y por la mano
  con que tira).
- Exposición de B: a qué zonas le dirigen los golpes sus rivales (`04`).

"Recibe" se lee como en `04`: le dirigen, no le entra. El bloque lo dice.

Hay un problema de lado que hay que resolver antes de cruzar. El ataque de A está en manos de A
(su izquierda, su derecha). La exposición de B está en manos de los rivales de B. Una zona de la
exposición de B, "cabeza por la mano izquierda del rival", es la misma zona a la que A tira con su
mano izquierda. Así que el cruce es directo: zona por zona, la fracción de A contra la fracción de
B, sin espejar nada. Esto vale porque DA-4 se tomó "por mano del rival"; con "por lado del cuerpo" habría que
espejar según las guardias.

## 2. B13 · API

Sin endpoint nuevo: la web pide los dos perfiles, como hoy. El cálculo de las lecturas vive en un
módulo puro, `boxtwin-api/src/boxtwin_api/rival.py`, expuesto en
`GET /rival?a={id}&b={id}`, para que las reglas se prueben en Python y la web no las copie (como
se decidió para las combinaciones en W11). La regla vieja de W12 que está en `lib/rival.ts` se
mueve a ese módulo en la misma tarea.

```json
{
  "cruces": [
    {"atacante": "A", "zona": "cuerpo_mano_der",
     "ataque_fraccion": 0.24, "exposicion_fraccion_rival": 0.31,
     "n_ataque": 88, "n_exposicion": 91}
  ],
  "lecturas": [
    {"tipo": "oportunidad", "para": "A", "regla": "zona_coincidente",
     "texto": "…", "cita": ["cuerpo_mano_der"]}
  ],
  "base": {"A": {"sesiones": 6, "golpes": 412}, "B": {"sesiones": 4, "golpes": 233}}
}
```

Cruces en los dos sentidos (A sobre B y B sobre A), cuatro zonas cada uno. Solo sparring y combate.

### Reglas de lectura (textos a validar por Lucas, como las de W12)

Se escriben ahora con sus umbrales, y quedan "En construcción" hasta que Lucas valide el texto,
igual que las cinco reglas de W12 (decisión del 13-10).

1. Zona coincidente: A dirige a una zona al menos el 25% de sus golpes y a B le dirigen sus
   rivales a esa misma zona al menos el 25% → "A suele ir donde a B suelen buscarlo: {zona}".
   Cita las dos fracciones.
2. Zona descuidada por el rival de B que A trabaja: A dirige a una zona 10 puntos más que el
   promedio de los rivales de B → "A trabaja {zona} más de lo que B suele ver".
3. Cuerpo: A dirige al cuerpo al menos 25% y a B le dirigen al cuerpo menos de 15% → "B no suele
   recibir trabajo al cuerpo y A lo hace".
4. Mínimo de datos: ninguna regla se dispara con menos de 50 golpes en cualquiera de las dos
   mitades. En ese caso: "Faltan sesiones para cruzar zonas: {n} y {m} golpes, hacen falta 50 de
   cada lado."

Las reglas no dicen "va a conectar", "es vulnerable" ni dan probabilidades. La frase fija de la
pantalla sigue: "Lectura cualitativa entre perfiles ya construidos. No es un pronóstico ni una
probabilidad de victoria."

### Pruebas (B13)

1. Con la bandera de C8 apagada, `cruces` y `lecturas` vienen vacíos.
2. Con 49 golpes de un lado no se dispara ninguna regla.
3. Cada lectura cita al menos una zona y sus dos fracciones están en `cruces`.
4. Las reglas de W12 dan lo mismo desde Python que desde `lib/rival.ts` antes de borrar la copia
   (se corre la batería de `rival.test.ts` contra el endpoint).

## 3. W16 · Web (pantalla 07)

- El bloque plegado "Dónde ataca uno y dónde recibe el otro" deja de estar "No disponible".
- Dos siluetas frontales enfrentadas, la de A a la izquierda y la de B a la derecha, con las cuatro
  zonas de `04`. Un selector "A ataca / B ataca" elige el sentido. En el sentido "A ataca", las
  zonas de A se dibujan como flechas desde sus manos, con grosor por fracción, y las zonas de B
  con el relleno de su exposición. Las flechas no tocan la silueta del otro: terminan antes, para
  no dibujar un contacto que nadie midió.
- En celular, las dos siluetas una arriba de la otra y el selector como pestañas de 44 px.
- Debajo, las lecturas de §2 con su cita, o "En construcción" mientras Lucas no valide los textos.
- Tabla alternativa con las ocho filas (cuatro zonas por sentido) para lector de pantalla.

## 4. Criterios de aceptación

- CA-16.1. Con C8 apagado, el bloque sigue "No disponible" y no aparece ninguna zona.
- CA-16.2. Ninguna lectura aparece con menos de 50 golpes por lado.
- CA-16.3. Ninguna flecha toca la silueta del rival.
- CA-16.4. Elegir el mismo peleador en A y B sigue sin estar permitido (CA-07.2).
