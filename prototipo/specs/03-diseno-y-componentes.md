# 03 · Sistema de diseño, componentes y movimiento

Cubre lo que la Clase 8 pide del apartado de UX/UI (decisiones de color, navegación y
jerarquía, usabilidad y accesibilidad) y lo que pide el Prompt 10 de la Clase 5 (Tailwind con
componentes listos, responsive mobile first, WCAG AA, jerarquía visual clara).

---

## 1. Principios que vienen del diseño y no se tocan

1. Oscuro por defecto y único tema. El material de sparring es oscuro y una interfaz clara
   alrededor del video obliga a la pupila a adaptarse dos veces. No hay selector de tema: el
   Prompt 10 lo sugiere como opción, y acá se descarta con este motivo.
2. Peleador A es rojo y cuadrado; peleador B es azul y círculo. La forma y la letra van siempre
   junto al color, para que la distinción no dependa solo del tono.
3. Lo medido, lo estimado y lo no disponible se ven distinto (§6). Es la decisión que
   atraviesa todo el diseño y la que viene de la posición epistemológica del proyecto.
4. Todo número de golpes lleva su margen cerca: "detectados", nunca "lanzados".
5. Un objetivo táctil mide 44 px como mínimo. Un campo de texto usa 16 px de fuente, el mínimo
   para que iOS no agrande la pantalla al enfocarlo.

## 2. Tokens

Salen de los colores del diseño. Se declaran una vez en `app/globals.css` con `@theme` de
Tailwind 4 y en ningún componente se escribe un hexadecimal.

```css
@theme {
  --color-fondo: #0E1113;
  --color-nav: #121518;
  --color-superficie: #15191C;
  --color-superficie-2: #111517;      /* tarjetas dentro de tarjetas */
  --color-elevado: #1F262B;           /* ítem activo */
  --color-borde: #2A3137;
  --color-borde-suave: #232A2F;
  --color-borde-fuerte: #3A4349;
  --color-texto: #ECEAE4;
  --color-texto-2: #C9D0D6;
  --color-texto-3: #B7BFC6;
  --color-tenue: #A3ACB4;             /* rótulos; 7,1:1 sobre superficie */
  --color-a: #C8412B;  --color-a-trazo: #E5563F;
  --color-b: #2F74C8;  --color-b-trazo: #4A90E2;
  --color-estimado: #F0C572; --color-estimado-borde: #E0A93B; --color-estimado-fondo: #261F11;
  --color-ok: #9FD6B2;  --color-ok-borde: #2F5A3D;  --color-ok-fondo: #18261D;
  --color-curso: #A9CBF3; --color-curso-borde: #2D4D73; --color-curso-fondo: #152231;
  --color-error: #F2A3A3; --color-error-borde: #7A3434; --color-error-fondo: #2A1515;

  --font-display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --font-sans: "IBM Plex Sans", "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", Consolas, monospace;

  --radius-sm: 3px; --radius: 6px; --radius-md: 8px; --radius-lg: 10px;
}
```

Las variables que esperan los componentes de shadcn y Animate UI (`--background`,
`--foreground`, `--card`, `--primary`, `--muted`, `--border`, `--ring`…) se mapean a estos
tokens en el mismo archivo, de modo que un componente instalado hereda la paleta sin
editarlo. `--primary` es `--color-texto` (el botón primario del diseño es claro sobre fondo
oscuro) y `--ring` es `--color-estimado-borde` para que el foco se vea.

Contraste verificado en la tarea W2 con un script que recorre los pares texto/fondo usados;
ninguno puede quedar bajo 4,5:1 (3:1 para texto de 24 px o más).

## 3. Tipografía y grilla, desde el celular

Fuentes cargadas con `next/font/google` (se sirven desde el build, sin pedir a Google en
producción): Barlow Condensed 500/600/700, IBM Plex Sans 400/500/600, IBM Plex Mono 400/500.

| Uso | Celular (base) | ≥ 1024 px |
|---|---|---|
| Título de pantalla (display 700) | 32 px / 1,0 | 46 px |
| Título de bloque (display 600) | 20 px | 24 px |
| Número grande (display 600) | 34 px | 42 px |
| Cuerpo (sans 400) | 15 px / 1,5 | 15 px |
| Rótulo (mono 11, mayúsculas, tracking 0,14em) | 11 px | 11 px |
| Dato tabular (mono 400/500) | 13 px | 13 px |

Puntos de corte (los de Tailwind): base 360 a 767, `md` 768, `lg` 1024, `xl` 1280. Las pantallas
se escriben para la base y se amplían con `md:` y `lg:`; nunca al revés.

- Margen lateral: 16 px en la base, 24 en `md`, 40 en `lg`.
- Ancho máximo del contenido: 1200 px, centrado, desde `xl`.
- Sin scroll horizontal de la página en ningún ancho. Las tablas anchas (frente a frente,
  contextos) hacen scroll dentro de su propia caja.

### Navegación

- Base y `md`: barra inferior fija de cinco destinos (Panel, Sesiones, Subir, Peleadores, Rival),
  con "Subir" al centro como botón destacado. Respeta `env(safe-area-inset-bottom)`.
- `lg` en adelante: barra lateral de 232 px como en el diseño, con la marca arriba y el usuario
  abajo. El bloque del plan no existe (D4).
- Encabezado de pantalla en la base: título y una acción a la derecha; la flecha atrás aparece en
  las pantallas de detalle (sesión, peleador, identificación).

## 4. Mapa de componentes de Animate UI

Se instalan con el CLI de shadcn desde el registro `@animate-ui`. Se prefieren las variantes de
la familia Radix porque traen el comportamiento accesible (foco, teclado, roles) resuelto.

```bash
npx shadcn@latest init
npx shadcn@latest add @animate-ui/components-radix-sidebar @animate-ui/components-radix-tabs \
  @animate-ui/components-radix-toggle-group @animate-ui/components-radix-progress \
  @animate-ui/components-radix-sheet @animate-ui/components-radix-popover \
  @animate-ui/components-radix-alert-dialog @animate-ui/components-radix-accordion \
  @animate-ui/components-radix-checkbox @animate-ui/components-radix-switch \
  @animate-ui/components-radix-dropdown-menu @animate-ui/components-radix-tooltip \
  @animate-ui/components-buttons-button @animate-ui/primitives-texts-counting-number \
  @animate-ui/primitives-effects-fade @animate-ui/primitives-effects-auto-height
npx shadcn@latest add input label card badge skeleton sonner
```

Si el CLI no resuelve el espacio de nombres `@animate-ui`, se declara en `components.json`:
`"registries": { "@animate-ui": "https://animate-ui.com/r/{name}.json" }`.
Después de instalar, `motion` se fija en 12.43.0 (el CLI instala la última).

| Patrón del diseño | Componente | Dónde |
|---|---|---|
| Navegación lateral en escritorio | `components-radix-sidebar` (en celular se reemplaza por la barra inferior propia, no por su hoja) | layout |
| Barra inferior | propia (`components/boxtwin/BarraInferior`), con íconos animados | layout |
| Botones | `components-buttons-button` | todas |
| Vistas de una sesión (Video y golpes / Fight-Card / Perfiles) | `components-radix-tabs` | 04, 05 |
| Sesión / R1 / R2 / R3 | `components-radix-tabs` (lista desplazable en celular) | 05, 09 |
| Contexto, A / B / Ninguno, velocidad 0,25× / 0,5× / 1×, filtros de contexto | `components-radix-toggle-group` | 02, 03, 04, 06 |
| Progreso de subida y de procesamiento | `components-radix-progress` | 01, 02, 08 |
| Corrección de tipo en celular | `components-radix-sheet` desde abajo | 04 |
| Corrección de tipo en escritorio | `components-radix-popover` | 04 |
| Confirmar siembra, cancelar subida | `components-radix-alert-dialog` | 02, 03 |
| "Cómo leer esta tarjeta", línea de tiempo plegable, bloques no disponibles | `components-radix-accordion` | 04, 05, 06 |
| Checklist de filmación y consentimiento | `components-radix-checkbox` | 02, 08 |
| Mostrar esqueleto | `components-radix-switch` | 04 |
| Exportar (PDF, CSV, JSON) | `components-radix-dropdown-menu` | 05 |
| Explicación de una métrica | `components-radix-tooltip` en escritorio; en celular el mismo contenido en `popover` al tocar, porque no hay hover | 05, 06 |
| Números grandes de los indicadores | `primitives-texts-counting-number` | 01, 05 |
| Entrada de listas y tarjetas | `primitives-effects-fade` | 01, 03 |
| Bloques que cambian de alto (detalle de golpe, tramos) | `primitives-effects-auto-height` | 03, 04 |
| Íconos | `@animate-ui/icons-*`: `upload`, `cloud-upload`, `play`, `pause`, `check`, `circle-check`, `loader-circle`, `x`, `users`, `user`, `chart-column`, `layout-dashboard`, `download`, `refresh-cw`, `sliders-horizontal`, `plus`, `clock`, `timer`, `arrow-left`, `chevron-down`, `chevron-right` | todas |
| Campos de texto y número | `ui/input`, `ui/label` de shadcn (Animate UI no tiene) | 02, peleadores |
| Selector de peleador | `<select>` nativo estilizado: en el celular abre el selector del sistema, que es más usable que un menú propio | 02, 08, 07 |
| Avisos breves | `ui/sonner` | todas |
| Esqueletos de carga | `ui/skeleton` | todas |

No se usan los fondos animados ni los botones decorativos del registro (liquid, ripple, fireworks
y similares): agregan movimiento sin información y peso en un celular de gama media.

## 5. Gráficos

Los gráficos del diseño son cuatro: barras de golpes por round (05), barras horizontales A
contra B (05, 09), línea de evolución entre sesiones (06) y siluetas a escala (07). Se dibujan en
SVG propio (`components/boxtwin/graficos/`), sin librería, porque son pocos, simples y tienen
que respetar las reglas de §6. Cada gráfico:

- lleva `role="img"` y un `aria-label` con la lectura en una frase ("De R2 a R3, Ríos cae de 44
  a 31 golpes detectados");
- tiene al lado o debajo la tabla con los mismos números, plegada en celular;
- no anima la escala ni los ejes, solo la entrada de las barras, y nada si el usuario pidió
  menos movimiento.

## 6. Cómo se ve cada estado de bloque

`components/boxtwin/EstadoBloque.tsx` envuelve cualquier bloque y recibe uno de los estados
de `01-producto.md` §4.

| Estado | Aspecto | Texto obligatorio |
|---|---|---|
| Medido | Número en `--color-texto`, sin adornos | El margen del detector cerca (precisión y recall de la Fight-Card, en lenguaje llano: "acierta 9 de cada 10 que marca, se le escapa la mitad") |
| Medido como estimación | Número en `--color-estimado` con "≈", borde punteado `--color-estimado-borde`, rótulo "estimación" | Versión del modelo y, si existe, su exactitud sobre fuente no vista |
| Condicionado, apagado | El bloque no muestra números. Caja con borde `--color-borde-fuerte` y rótulo "en validación" | Qué criterio falta ("se muestra cuando el indicador pase la validación contra marcado manual") |
| No disponible | Caja plegada (acordeón) con rótulo "no disponible en esta versión" | Qué pieza falta y por qué ("una cámara sola no establece contacto: hace falta la sub-hipótesis 2") |
| Corregido por el entrenador | Tipo en `--color-ok` con rótulo "corregido" | El tipo original tachado al lado |

El texto del margen sale del documento (`fightcard.detector.precision_medida` y
`recall_medido`), no de una constante de la web: si el detector mejora, la pantalla mejora sola.
Regla de redondeo para el lenguaje llano: `round(valor × 10)` de cada 10.

## 7. Movimiento

Motion se usa para que el usuario entienda qué cambió, no para decorar.

- Se envuelve la app en `<MotionConfig reducedMotion="user">`. Con
  `prefers-reduced-motion: reduce` no hay desplazamientos ni conteos animados: los números
  aparecen con su valor final.
- Duraciones: 150 ms para cambios de estado de un control, 200 a 250 ms para entrada de un
  bloque, nunca más de 300 ms. Curva `easeOut` para entrar, `easeIn` para salir.
- Se anima: el cambio de pestaña, la entrada de la lista de golpes, la apertura de la hoja de
  corrección, el progreso, el número de un indicador cuando cambia, el paso de "procesando" a
  "listo".
- No se anima: el video, la línea de tiempo mientras corre el video (se mueve con el tiempo, no
  con una animación), los ejes de los gráficos, nada en bucle salvo el ícono de carga.
- Las animaciones usan `transform` y `opacity`; nada que fuerce reflujo en cada cuadro.

## 8. Accesibilidad (WCAG 2.1 AA)

- Controles reales: `<button>`, `<a href>`, `<input>` con su `<label>`. Ningún `div` con
  `onClick`.
- Foco visible en todo, con el anillo de `--ring`.
- La línea de tiempo es un grupo de botones navegable con flechas (izquierda y derecha: golpe
  anterior y siguiente del mismo peleador; arriba y abajo: cambiar de peleador), y cada tick
  tiene `aria-label` "Golpe detectado de Ríos, jab estimado, 1 minuto 47".
- El video tiene los controles nativos además de los propios, y los atajos propios no capturan
  teclas globales.
- Avisos de estado (subida terminada, análisis listo, corrección guardada) en una región
  `aria-live="polite"`.
- Idioma `lang="es-AR"`; números con coma decimal (`Intl.NumberFormat("es-AR")`).
- La accesibilidad pensada para distintos niveles de familiaridad con la tecnología que menciona
  la Clase 8: un solo camino principal por pantalla, un botón primario por pantalla, textos de
  ayuda en lenguaje del gimnasio y no técnico ("tramo" y no "track", "marcá" y no "sembrá").

## 9. Presupuesto de rendimiento en un celular de gama media

Medido con Lighthouse en modo móvil (Moto G Power simulado, 4G lenta) sobre el build servido por
la API:

- LCP ≤ 2,5 s en Panel y en Fight-Card.
- JavaScript inicial por ruta ≤ 380 KB comprimido. Modificado el 13-10 (era 200 KB): con
  React DOM, el runtime de Next y Motion, que pide Animate UI, el piso medido es 192 KB y las
  pantallas del prototipo van de 281 a 360 KB. El límite queda por encima de lo medido con
  margen, y lo que decide si la carga es aceptable en un celular de gama media es el LCP y el
  INP de abajo, no el peso.
- INP ≤ 200 ms al tocar un golpe de la línea de tiempo.
- CLS ≤ 0,1 (las imágenes de los recortes y el video llevan sus dimensiones reservadas).
- Puntaje de accesibilidad de Lighthouse ≥ 95 en todas las rutas.
