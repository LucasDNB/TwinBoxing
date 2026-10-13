# 02 · Arquitectura y stack

Sigue la vista en capas que pide la Clase 8 (presentación, API, lógica, datos) y responde
dónde se aloja cada parte.

---

## 1. Decisión principal: Next.js exportado, servido por la API

```
celular / escritorio
   │ HTTPS, un solo túnel, un solo origen
   ▼
boxtwin-api (FastAPI, estación de desarrollo)
   ├── rutas de la API (/auth, /jobs, /videos, /fightcards, /boxeadores, /producto, /panel…)
   └── "/" → StaticFiles sobre web-entrenador/out  (HTML, JS y CSS que generó `next build`)
   │
   ├── PostgreSQL (usuarios, sesiones, trabajos, boxeadores, correcciones)
   └── worker (cola en tabla, dos entornos conda) → sesiones/<id>/fightcard.json, procesado.mp4
```

Next.js se usa con `output: "export"`: el build genera archivos estáticos y en producción no
corre ningún servidor de Node. La API ya sirve el frontend desde su mismo origen
(`app.mount("/", StaticFiles(..., html=True))`, al final de `app.py`), así que el cambio de
despliegue se reduce a apuntar `BOXTWIN_WEB` a la carpeta `out/`.

Por qué así y no con el servidor de Next:

| Opción | A favor | En contra | Decisión |
|---|---|---|---|
| A. Exportación estática servida por FastAPI | Un origen, un túnel, sin CORS, sin un proceso más que vigilar en un servidor de un nodo (RNF4). El ticket de video sigue funcionando igual | Sin rutas dinámicas en el servidor ni componentes de servidor: todo se pide desde el cliente | **Elegida** |
| B. `next start` detrás del túnel con reescrituras a la API | Rutas dinámicas, componentes de servidor | Un servicio systemd más, dos procesos que coordinar en el despliegue, un túnel más o un proxy | Descartada |
| C. Vercel para la web y túnel para la API | Hosting gratis de la web | Dos orígenes, CORS, el token cruzando dominios, y el video igual sale por el túnel | Descartada |

Consecuencia de la opción A: las pantallas que dependen de un identificador lo reciben por
la query (`/sesion/?id=…`) y no por un segmento de ruta (`/sesion/[id]`), porque la
exportación estática no genera páginas para identificadores que no conoce en el build.

Lo que cambia en la tesis: el Cap 4 (stack, frontend React + Vite) y el Anexo I (cómo
ejecutarlo) quedan desactualizados cuando la web nueva reemplace a la vieja. Se corrigen en la
entrega acumulativa, no antes. El Cap 7 del 13-10 informa la web que está corriendo hoy.

---

## 2. Versiones

Fijadas el 12-10-2026 con `npm view`. Se fija versión exacta en `package.json` para lo que
tiene historia de romper (`motion`, `next`), y rango menor para el resto.

| Paquete | Versión | Nota |
|---|---|---|
| Node.js | 22 LTS | Next 16 pide ≥ 20.9 |
| next | 16.4.x | App Router, `output: "export"`, `trailingSlash: true` |
| react, react-dom | 19.x | |
| typescript | 5.x | `strict: true` |
| tailwindcss | 4.3.x | configuración en CSS (`@theme`), sin `tailwind.config.js` |
| shadcn (CLI) | 4.x | solo para instalar componentes; no es dependencia de runtime |
| motion | **12.43.0 exacta** | Animate UI está escrito sobre la 12; la 14 salió el 02-10-2026 |
| radix-ui | 1.x | lo traen los componentes de Animate UI de la familia Radix |
| lucide-react | 1.x | íconos estáticos; los animados vienen de Animate UI |
| @tanstack/react-query | 5.x | estado del servidor, sondeo de trabajos |
| zod | 4.x | validación de formularios y de respuestas de la API |
| vitest + @testing-library/react | 5.x | pruebas de componentes e integración |
| msw | 2.x | API simulada en pruebas de integración |
| @playwright/test | 1.6x | e2e en viewport de celular y de escritorio |
| @axe-core/playwright | 4.x | accesibilidad en e2e |

No se agregan: Redux o Zustand (el estado es del servidor y lo maneja TanStack Query; el
estado de la interfaz cabe en el componente), una librería de gráficos (los cuatro gráficos
del diseño son barras y líneas simples en SVG propio, ver `03` §5), Three.js (el avatar 3D está
fuera).

---

## 3. Estructura de carpetas

El paquete nuevo se llama `boxtwin-web-entrenador` y vive al lado de `boxtwin-web` hasta el
reemplazo. Después del reemplazo, `boxtwin-web` se borra en un commit propio.

```
boxtwin-web-entrenador/
├── app/
│   ├── layout.tsx              fuentes, tema oscuro, MotionConfig, QueryClientProvider
│   ├── globals.css             Tailwind 4 + tokens (@theme) de 03 §2
│   ├── ingresar/page.tsx       ingreso y registro por invitación
│   ├── (app)/layout.tsx        guarda de sesión + navegación (barra inferior / lateral)
│   ├── (app)/page.tsx          01 Panel
│   ├── (app)/sesiones/page.tsx           lista completa de sesiones
│   ├── (app)/sesiones/nueva/page.tsx     02 / 08 Nueva sesión
│   ├── (app)/sesion/page.tsx             04 / 05 Sesión (?id=…&vista=video|fightcard|perfiles)
│   ├── (app)/sesion/identificar/page.tsx 03 Identificación (?id=…)
│   ├── (app)/peleadores/page.tsx         lista y alta
│   ├── (app)/peleador/page.tsx           06 Perfil (?id=…&contexto=…)
│   └── (app)/rival/page.tsx              07 Rival Fantasma (?a=…&b=…)
├── components/
│   ├── animate-ui/             lo que instala el CLI de shadcn desde @animate-ui (no se edita a mano
│   │                           salvo para tokens; cada cambio queda anotado en NOTAS.md de la carpeta)
│   ├── ui/                     base de shadcn sin equivalente en Animate UI (input, label, card, badge, skeleton, sonner)
│   ├── boxtwin/                piezas propias: Marca, Esquina (A/B), EstadoBloque, Margen, Rotulo, Tick…
│   └── pantallas/              una carpeta por pantalla con sus subcomponentes
├── lib/
│   ├── api/cliente.ts          fetch con token, errores tipados, subida con XHR
│   ├── api/esquemas.ts         zod de cada respuesta (sesión, trabajo, fightcard 0.1/0.2, perfil, producto)
│   ├── api/consultas.ts        hooks de TanStack Query, claves y sondeo
│   ├── fightcard.ts            agregados de pantalla: conteos por tipo, por round, combinaciones
│   ├── formato.ts              tiempo, porcentajes, decimales con coma (es-AR), nombres de golpe
│   ├── banderas.ts             lectura de /producto y helper `estadoDe(bloque)`
│   └── sesion-usuario.ts       token y usuario en localStorage con tolerancia a Safari privado
├── tests/                      integración (vitest + msw) y e2e (playwright)
├── public/                     manifest.webmanifest, íconos (P2)
├── components.json             config de shadcn con el registro @animate-ui
├── next.config.ts
└── package.json
```

`lib/formato.ts` porta `boxtwin-web/src/formato.js` con sus pruebas: esa lógica ya está
probada y no se reescribe desde cero.

---

## 4. Rutas de la web y rutas de la API

La web y la API comparten el origen. Starlette prueba primero las rutas de la API y después el
montaje estático, así que una página con el mismo prefijo que una ruta GET de la API queda
tapada. Prefijos reservados para la API, que ninguna página puede usar:

`/auth`, `/jobs`, `/videos`, `/fightcards`, `/boxeadores`, `/salud`, `/producto`, `/panel`,
`/docs`, `/openapi.json`.

Por eso la lista de peleadores es `/peleadores/` y no `/boxeadores/`. `PUT /sesiones/{id}/boxeadores`
convive con la página `/sesiones/` porque los métodos y la profundidad son distintos; una
prueba de B0 lo verifica.

| Página | Archivo exportado | Query |
|---|---|---|
| `/ingresar/` | `ingresar/index.html` | `?siguiente=` |
| `/` | `index.html` | |
| `/sesiones/` | `sesiones/index.html` | `?estado=` |
| `/sesiones/nueva/` | `sesiones/nueva/index.html` | |
| `/sesion/` | `sesion/index.html` | `?id=`, `?vista=video\|fightcard\|perfiles`, `?t=` (segundo inicial), `?round=` |
| `/sesion/identificar/` | `sesion/identificar/index.html` | `?id=` |
| `/peleadores/` | `peleadores/index.html` | |
| `/peleador/` | `peleador/index.html` | `?id=`, `?contexto=` |
| `/rival/` | `rival/index.html` | `?a=`, `?b=` |

El estado que el usuario elige y que tiene sentido compartir (round, vista, contexto) va en la
query, para que el botón atrás del celular haga lo esperable y un enlace abra la misma vista.

---

## 5. Datos y estado

- Todo dato del servidor pasa por TanStack Query. Claves: `["sesiones"]`, `["sesion", id]`,
  `["fightcard", id]`, `["boxeadores"]`, `["perfil", id, contexto]`, `["panel"]`,
  `["producto"]`.
- Cada respuesta se valida con zod antes de llegar a un componente. Si la Fight-Card no valida,
  la pantalla muestra el error con la versión recibida y no intenta dibujarla a medias.
- Sondeo del estado de una sesión: cada 3 s mientras está en `en_cola`, `procesando` o
  `completando`; se detiene en `espera_siembra`, `listo` y `fallo`; se pausa con la pestaña
  oculta (`refetchIntervalInBackground: false`). El panel sondea cada 10 s solo si alguna
  sesión visible está en curso.
- Las correcciones de tipo son optimistas: la lista cambia al tocar "Guardar" y vuelve atrás si
  la API responde error, con un aviso.
- Fight-Card 0.1 y 0.2 conviven: el esquema acepta las dos y `lib/fightcard.ts` completa con
  `null` lo que la 0.1 no trae (ver `05` §4).

## 6. Autenticación

Se mantiene el esquema de hoy: `POST /auth/login` devuelve un token bearer, que la web guarda en
`localStorage` con la misma tolerancia a Safari en modo privado que tiene `boxtwin-web/src/api.js`.
Un 401 en cualquier pedido borra el token y lleva a `/ingresar/?siguiente=<ruta actual>`.

La guarda de sesión vive en `(app)/layout.tsx` y corre en el cliente: con exportación estática
no hay middleware de servidor. Mientras resuelve, se muestra el esqueleto de la pantalla, no una
pantalla en blanco.

## 7. Video, recortes y subida

- `<video>` e `<img>` no mandan el header `Authorization`. La web pide
  `GET /videos/{id}/ticket` (30 minutos, un solo video) y lo pone en la query
  (`?t=`) de `/videos/{id}/stream`, `/videos/{id}/procesado` y
  `/jobs/{id}/candidatos/{track}`. El ticket se renueva a los 25 minutos si la pantalla sigue
  abierta.
- El video procesado se pregunta con `HEAD /videos/{id}/procesado`. Si no está, se usa el
  original y se ofrece pedirlo con `POST /jobs/{id}/render`.
- La subida usa `XMLHttpRequest` y no `fetch`, porque `fetch` no informa el progreso de subida
  en los navegadores de celular. Se puede cancelar (`xhr.abort()`). No es reanudable: un corte
  de red obliga a subir de nuevo y la pantalla lo dice. La subida reanudable (tus) queda como
  línea futura y se nombra en el Cap 8 si la prueba con usuarios muestra cortes.
- Antes de subir, la web lee del archivo lo que el navegador puede leer sin decodificar todo:
  duración y resolución (un `<video>` oculto con `preload="metadata"`). Los fps y la tasa
  constante los informa el servidor después de `procesar` (`duracion_s`, `fps` de la sesión),
  porque el navegador no los expone de forma confiable.
- Tamaño y formato se validan en el cliente con los mismos límites que la API
  (`.mp4`, `.mov`, `.m4v`; `max_bytes` leído de `GET /producto`), para no descubrir el 413
  después de subir un gigabyte.

## 8. Seguridad

- Las tres cosas que ya resuelve la API se mantienen: cada sesión solo la ve su dueño (RNF3),
  el ticket de video, el registro cerrado por invitación.
- La web no usa `dangerouslySetInnerHTML`. El export HTML de la Fight-Card se abre en una
  pestaña nueva desde la API, no se inyecta.
- Content-Security-Policy desde la API para la carpeta estática: `default-src 'self'`,
  `img-src 'self' blob: data:`, `media-src 'self' blob:`, `style-src 'self' 'unsafe-inline'`
  (Motion escribe estilos en línea), `font-src 'self'`. Las fuentes se sirven desde el propio
  build (`next/font`), así que no hace falta abrir Google Fonts.
- Ningún secreto en la web: no hay variables `NEXT_PUBLIC_` con claves. La URL de la API es
  relativa.

## 9. Desarrollo local

```bash
cd boxtwin-web-entrenador
npm install
npm run dev      # http://localhost:3000, reescribe /auth, /jobs, /videos… a localhost:8000
npm run build    # genera out/
npm test         # vitest
npm run e2e      # playwright contra el build servido por la API de prueba
```

`next.config.ts` arma la configuración según el modo: en desarrollo declara `rewrites` hacia
`localhost:8000` para los prefijos reservados del §4 y no declara `output`; en el build
declara `output: "export"` y no declara `rewrites`, porque Next no las admite en una
exportación estática y avisa si las encuentra. En producción no hacen falta: el origen es uno.
