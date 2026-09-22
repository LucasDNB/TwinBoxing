# boxtwin-web

El frontend. Seis pasos: entrar, subir el video, esperar, decir quién es quién, ver la
Fight-Card, tocar un evento y verlo en el video.

React + Vite, sin router y sin librería de estado. El flujo son dos pantallas —la lista de
sesiones y una sesión— y las dos entran en un hash. Cuando haya más, se agrega.

## Tres decisiones que no son estéticas

**El margen va arriba de todo.** Con recall medido 0,484 el conteo está por debajo del real.
Un número grande sin su margen se lee como un dato exacto, así que el margen es lo primero
que aparece en la Fight-Card, en amarillo, y el texto sale del propio documento: si el
detector mejora, el número de la pantalla mejora con él, en vez de quedar una constante
vieja escrita en la interfaz.

**El video es la función principal, no un extra.** Lo que convierte un conteo dudoso en una
herramienta de revisión es que el entrenador toque un evento, vea el golpe y decida él. Por
eso el reproductor queda pegado arriba mientras se recorre la línea de tiempo, y el salto cae
medio segundo antes del evento: el golpe empieza donde arranca el movimiento, que es antes de
que se haga evidente.

**Lo medido y lo estimado se ven distinto.** El volumen y la guardia salen de medición; el
tipo de golpe sale de un clasificador que no generaliza a fuentes nuevas. El tipo va en
itálica y con su confianza al lado; una corrección del entrenador va en otro color. Mezclarlos
en la misma tipografía sería mentir prolijo.

## El video y el token

Un elemento `<video>` no puede mandar un header, así que no hay forma de pasarle el
`Authorization`. La salida habitual es poner el token en la query, y el token de sesión en una
query es lo peor de los dos mundos: dura días, sirve para todo y queda escrito en los logs, en
el historial y en el Referer.

La API emite un **ticket**: media hora, un solo video, nada más. Eso es lo que va en la URL.

## Correr

```bash
npm install
npm run dev        # http://localhost:5173, proxea la API de localhost:8000
npm run build
npm test
```

El proxy de vite hace que el frontend hable siempre con rutas relativas, así que no hay una
URL de API que configurar ni que equivocar al desplegar: en producción los dos salen por el
mismo túnel.

## Pensado para un celular en un gimnasio

Oscuro por defecto, porque el material de sparring es oscuro y una interfaz blanca alrededor
obliga a la pupila a trabajar dos veces. Botones de 44 px, que es el ancho de un dedo. Inputs
de 16 px, que es el mínimo para que iOS no haga zoom al enfocar. El input de archivo lleva
`capture`, así que en el celular abre la cámara directo.

## Lo que no hay

No hay avatar 3D, no hay modo en vivo y no hay nada que hable de conexión, puntuación o
veredicto. Lo primero y lo segundo están fuera del alcance del MVP; lo tercero está fuera del
alcance del producto, porque un sistema monocular no establece contacto físico.
