# Spec del MVP — BoxTwin

Estado: aprobado con ajustes · 21-09-2026 (decisiones 1 a 3 de Lucas) · base: `main` en `8fa07eb` (PR #7 mergeado)
Ubicación sugerida en el repo: `docs/specs/MVP.md`
Plazo: 8 semanas, del 22-09 al 16-11. Entrega final y presentación el 17-11-2026.

Formato: sigue el Prompt 9 de la Clase 5 (diez funcionalidades ranqueadas por valor y esfuerzo,
cuáles entran y cuáles no, flujo en 5 a 7 pasos, stack para 8 semanas, tres riesgos de
ejecución) y la distinción de la Clase 8 entre prototipo y MVP: el MVP tiene que entregar valor
y poder probarse con usuarios reales, no simular funciones.

---

## 1. Punto de partida medido (qué hay y qué no)

Lo que existe y está medido:

- Pose y seguimiento: YOLOv8l-pose + BoT-SORT, media precisión, 2,3× tiempo real sobre la
  2080 Super para una pelea de 37,9 min.
- Identidad automática (nuevo, 18 al 22-09): altura + guante + color del guante.
  82,4% de la partición sobre seis fuentes de gimnasio (84/102 tracks), 95,7% sobre
  transmisión (44/46). Con perfiles sembrados desde tracks con rol conocido, el color asigna
  112 de 113 tracks (99,1%). La diferencia entre 82,4% y 99,1% es toda de la siembra.
  No adivina cuando los guantes son del mismo color: se abstiene.
- Detector de golpes (TCN, por brazo): sobre fuente no vista, precisión 0,885, recall 0,484.
- Clasificador de familia (PoseConv3D, entorno `boxtwin_mmaction`): no generaliza a fuentes
  nuevas. El 0,745 está inflado.

Lo que no existe:

- Un comando de inferencia que vaya de un video crudo a un resultado sin pasar por un proyecto
  de anotación. `tools/pipeline.py` entrena el ensamble por fold y lee `*.det.npz` exportados
  de proyectos anotados; `demo_vivo.py` es la única pieza que usa un checkpoint congelado.
- API, base de datos, cola, frontend, Docker. Nada del stack web del Cap 4 está en el repo.
- Heurísticas de conexión, bloqueo y esquiva (sub-hipótesis 2).

Consecuencia de diseño: el MVP se apoya en las piezas que generalizan (pose, identidad,
detector) para lo que muestra como medición, e incluye el tipo de golpe como estimación
versionada que mejora con cada modelo nuevo, sin que el resto de la Fight-Card dependa de él.

---

## 2. Propuesta de valor

Usuario objetivo del MVP: entrenador de gimnasio y boxeador que hace sparring filmado con el
celular. Es el segmento con el que hay material propio y acceso para probar.

Problema: hoy el análisis se hace a ojo, en el rincón, durante el round (entrevista a Suárez,
Apéndice A). No queda registro objetivo de cuánto trabajó cada uno ni de cuándo se abrió la
guardia.

Lo que el MVP entrega: subís el video del sparring, marcás cuál de los dos sos, y recibís por
peleador y por round el volumen de golpes detectados por brazo, su evolución a lo largo de la
sesión y los descuidos de guardia, con cada evento enlazado al momento exacto del video para
que el entrenador lo verifique.

Por qué alguien pagaría por esto y no por menos: el número solo no alcanza, porque con recall
de 0,48 el conteo absoluto está por debajo de lo real. Lo que sí sostiene la medición es la
comparación dentro de la sesión (round 1 contra round 3, A contra B) y la lectura de guardia,
que es lo que Suárez pone primero. La línea de tiempo enlazada al video es lo que convierte un
número dudoso en una herramienta de revisión: el entrenador ve cada evento y decide.

---

## 3. Funcionalidades ranqueadas (matriz valor / esfuerzo)

| # | Funcionalidad | Valor | Esfuerzo | MVP |
|---|---|---|---|---|
| F1 | Subida de video desde navegador móvil o de escritorio, procesamiento en cola | Alto | Medio | Sí |
| F2 | Procesamiento automático en servidor: pose, identidad, detector | Alto | Medio | Sí |
| F3 | Confirmación de identidad en un paso: elegir "este soy yo / este es el rival" entre recortes | Alto | Bajo | Sí |
| F4 | Fight-Card de volumen: golpes detectados por brazo, por round y por minuto, caída entre rounds | Alto | Bajo | Sí |
| F5 | Indicadores de guardia: retorno de la mano al mentón y mano opuesta caída durante el golpe | Alto | Medio | Sí, condicionado a validación (sección 7) |
| F6 | Línea de tiempo navegable: click en un evento salta al video | Alto | Bajo | Sí |
| F7 | Exportar la Fight-Card a PDF o CSV | Medio | Bajo | Sí |
| F8 | Historial de sesiones de un mismo boxeador | Medio | Bajo | Solo si sobra tiempo |
| F9 | Tipo de golpe estimado (familia), con confianza y versión del modelo | Alto | Medio | Sí, como estimación que se ajusta con cada modelo |
| F14 | Corregir el tipo de golpe desde la línea de tiempo; la corrección se guarda como etiqueta | Medio | Bajo | Sí |
| F10 | Conectados, bloqueados, esquivados, mapa de exposición | Alto | Alto | No: sub-hipótesis 2 sin implementar |
| F11 | Avatar 3D con Three.js, Rival Fantasma | Medio | Alto | No |
| F12 | Modo en vivo con overlay | Medio | Muy alto | No: sub-hipótesis 3 se mide con el benchmark, no con el MVP |
| F13 | Planes, pagos, multi-organización | Bajo para el PFI | Medio | No |

F9 entra aunque el clasificador no generalice todavía, por decisión de producto: que
clasifique como le salga y se ajuste a medida que mejore el modelo. Para que eso sea
honesto y medible, cada etiqueta lleva su confianza y la versión del checkpoint que la
produjo, y la Fight-Card separa lo medido (volumen, guardia) de lo estimado (tipo). F14
cierra el ciclo: cada corrección del entrenador es una etiqueta nueva sobre material que el
modelo no vio, que es exactamente lo que le falta para generalizar.

Costo operativo: el worker necesita los dos entornos conda (`twinboxing_env` y
`boxtwin_mmaction`), que no se mezclan. Van en la misma imagen como dos entornos separados y
se comunican por el archivo JSON de segmentos, como ya lo hace el pipeline.

---

## 4. Flujo del usuario (6 pasos)

1. Entra al sitio desde el celular y se loguea (cuenta simple, email y contraseña).
2. Sube el video del sparring e indica duración de round y descanso (o "sin rounds").
3. Espera: el sistema muestra el estado del trabajo (en cola, procesando, listo).
4. Ve dos recortes candidatos y marca cuál es él y cuál el rival. Si el sistema no pudo
   separar dos candidatos, le muestra más y elige.
5. Recibe la Fight-Card: volumen por brazo y por round, curva de trabajo, eventos de guardia,
   y la distribución de tipos de golpe marcada como estimada.
6. Toca cualquier evento y el video salta a ese instante. Si el tipo está mal, lo corrige
   ahí mismo. Puede exportar.

El paso 4 es la única intervención humana y es deliberada: reemplaza la siembra automática,
que es donde se pierde la diferencia entre 82,4% y 99,1%.

---

## 5. Requisitos

### Funcionales

- RF1. CUANDO el usuario sube un video mp4 o mov de hasta 30 minutos, EL SISTEMA DEBE
  encolarlo y devolver un identificador de trabajo sin esperar el procesamiento.
- RF2. EL SISTEMA DEBE procesar el video sin intervención humana hasta el punto de siembra
  de identidad, incluyendo filtros de altura y guante.
- RF3. CUANDO el procesamiento previo termina, EL SISTEMA DEBE proponer los dos tracks
  candidatos (los dos que más coexisten separados) con un recorte de cada uno.
- RF4. CUANDO el usuario confirma la siembra, EL SISTEMA DEBE completar la asignación de
  identidad por color, correr el detector y generar la Fight-Card.
- RF5. SI los perfiles de color quedan por debajo del piso de separación, EL SISTEMA DEBE
  abstenerse en esos tracks y decirlo en la Fight-Card (porcentaje del tiempo sin
  asignar), en lugar de asignar.
- RF6. La Fight-Card DEBE mostrar "golpes detectados", nunca "golpes lanzados", con la
  precisión y el recall medidos del detector visibles en la misma pantalla.
- RF7. Cada evento de la Fight-Card DEBE tener marca de tiempo y enlazar al video.
- RF8. EL SISTEMA NO DEBE mostrar conexión, puntuación de round ni veredicto.
- RF9. Cada golpe DEBE mostrar su tipo estimado con la confianza del clasificador; la
  Fight-Card DEBE indicar la versión del modelo y su exactitud medida sobre fuente no vista,
  y rotular la distribución de tipos como estimación.
- RF10. CUANDO el usuario corrige un tipo, EL SISTEMA DEBE guardar la corrección con el
  segmento, el video, la etiqueta original y la versión del modelo, sin pisar la original.
- RF11. La nomenclatura en pantalla DEBE seguir el uso rioplatense (cross es el gancho),
  resuelto en la capa de presentación sin tocar el esquema de anotación.
- RF12. CUANDO se publica un checkpoint nuevo, EL SISTEMA DEBE poder reclasificar sesiones
  existentes desde los segmentos guardados, sin repetir pose ni detección.

### No funcionales

- RNF1. Tiempo de procesamiento no mayor a 2× la duración del video en la estación de
  desarrollo (hoy la cadena completa con clasificador mide 1,447 h por hora de video).
- RNF2. El worker es una imagen Docker con runtime NVIDIA, la misma que se usa en el PoC de
  nube. Ningún engine de TensorRT se copia entre máquinas.
- RNF3. Los videos y resultados de un usuario solo son visibles para ese usuario.
- RNF4. Servidor de un solo nodo: la estación de desarrollo, como declara el Cap 4.

---

## 6. Diseño técnico

### Componentes

```
navegador (React + Vite)
   │  HTTPS (túnel hacia la estación de desarrollo)
   ▼
API (FastAPI + Pydantic 2) ──── PostgreSQL (usuarios, videos, trabajos, fightcards)
   │ tabla jobs, SELECT ... FOR UPDATE SKIP LOCKED
   ▼
worker (Docker, GPU, dos entornos conda en la misma imagen)
   [twinboxing_env]     boxtwin procesar  → pose + tracking + filtros de identidad → ESPERA_SIEMBRA
   [twinboxing_env]     boxtwin completar → color + detector + guardia → segmentos.json
   [boxtwin_mmaction]   clasificar        → tipo + confianza por segmento → fightcard.json
```

La cola se resuelve con una tabla de trabajos en PostgreSQL y no con Redis y Celery: una
pieza menos que instalar, monitorear y justificar, y alcanza para un nodo y pocos usuarios.

### Comando de inferencia (la pieza que falta y la primera que se construye)

Dos subcomandos nuevos en el paquete del anotador, que ya tiene preproceso e identidad:

- `boxtwin procesar <video> --out <dir>`: preproceso, cache de pose, filtros de altura y
  guante, propuesta de dos semillas con sus recortes. Guarda la evidencia con
  `--guardar-evidencia` para no redecodificar.
- `boxtwin completar <dir> --semilla-a <track> --semilla-b <track>`: asignación por color
  desde esas semillas, features del detector (20 por cuadro, mismas que el entrenamiento),
  inferencia con checkpoint congelado, decodificación a segmentos, indicadores, JSON.
- Etapa de clasificación en `boxtwin_mmaction`: adaptar `tools/clasificar_segmentos.py` para
  que lea `segmentos.json` de una sesión, clasifique con el checkpoint PoseConv3D vigente y
  escriba tipo y confianza. Es la misma etapa que reclasifica cuando cambia el modelo.

Checkpoint de producción: ensamble entrenado sobre las siete fuentes. Con eso no queda
ninguna fuente propia sin ver, así que la validación del MVP se hace sobre sesiones nuevas
(sección 7).

### Contrato `fightcard.json` (versión 0.1)

```json
{
  "version": "0.1",
  "video": {"duracion_s": 0, "fps": 0, "rounds": [{"inicio_s": 0, "fin_s": 0}]},
  "identidad": {"cobertura_A": 0.0, "cobertura_B": 0.0, "sin_asignar": 0.0},
  "detector": {"checkpoint": "", "precision_medida": 0.885, "recall_medido": 0.484},
  "clasificador": {"checkpoint": "", "exactitud_familia_fuente_no_vista": null, "estimado": true},
  "peleadores": {
    "A": {
      "golpes": [{"id": "", "t_inicio": 0.0, "t_fin": 0.0, "brazo": "izq|der", "score": 0.0,
                  "tipo": "jab|cross|hook|uppercut", "confianza_tipo": 0.0, "corregido": null}],
      "por_round": [{"round": 1, "izq": 0, "der": 0}],
      "guardia": [{"t": 0.0, "tipo": "retorno_lento|mano_opuesta_caida", "brazo": "", "valor": 0.0}]
    },
    "B": {}
  }
}
```

### Endpoints

- `POST /auth/registro`, `POST /auth/login`
- `POST /videos` (multipart) → `{job_id}`
- `GET /jobs/{id}` → estado y, si corresponde, candidatos de siembra con URLs de recortes
- `POST /jobs/{id}/siembra` → `{track_a, track_b}`
- `GET /fightcards/{id}`, `GET /fightcards/{id}/export?formato=pdf|csv`
- `PATCH /golpes/{id}` → `{tipo}` (corrección, se guarda aparte de la original)
- `GET /videos/{id}/stream` (rango HTTP, para saltar al instante)

---

## 7. Criterios de aceptación pre-registrados

Se fijan ahora, antes de medir, siguiendo el principio de protocolo antes de medir.

| Criterio | Cómo se mide | Umbral | Si no se cumple |
|---|---|---|---|
| C1. Identidad con siembra humana | Tracks bien asignados sobre las seis fuentes de gimnasio, sembrando con los dos tracks que elegiría el usuario | ≥ 95% | Se agrega corrección manual por track en la web |
| C2. Estabilidad del recall del detector | Recall por round sobre `sparring-3`, fold sin esa fuente | Variación entre rounds ≤ 0,10 | La curva de trabajo se muestra solo por sesión, no por round |
| C3. Validez del indicador de guardia | 100 golpes de `sparring-3` marcados a mano por Lucas (volvió a guardia sí/no, mano opuesta caída sí/no), comparados contra el indicador | Acuerdo ≥ 80% en cada uno | F5 sale del MVP y se declara como línea futura |
| C4. Punta a punta sin intervención | Sesiones nuevas desde la subida hasta la Fight-Card con solo el paso 4 | 100% de las sesiones de prueba | Bloqueante |
| C5. Tiempo de procesamiento | Mediana sobre las sesiones de prueba en la estación | ≤ 2× duración | Se baja a YOLOv8m-pose y se mide la pérdida |
| C7. Línea de base del clasificador en el producto | Exactitud de familia sobre una muestra anotada de las sesiones de prueba, y sobre las correcciones de F14 | Sin umbral: F9 entra igual. Se registra por versión de checkpoint para comparar las siguientes | Se reporta tal cual en el Cap 7 como evidencia de la sub-hipótesis 1 |
| C6. Prueba con usuarios | 3 a 5 sesiones nuevas de gimnasio, con Suárez o un entrenador del gimnasio revisando la Fight-Card | Cualitativo, guion corto escrito antes | Se reporta tal cual en el Cap 7 |

Definición operativa de guardia para C3 (a confirmar por Lucas como criterio de dominio):
muñeca a menos de 0,6 anchos de hombro de la nariz. Retorno lento: más de N cuadros entre el
fin del golpe y la vuelta a esa zona, con N fijado sobre la mediana de `sparring-3` antes de
correr la comparación. Monocular: la profundidad no se ve, así que el indicador se declara como
estimación.

---

## 8. Plan de 8 semanas (con las entregas de tesis encima)

| Semana | Fechas | Desarrollo | Tesis |
|---|---|---|---|
| 1 | 22-09 a 28-09 | Aprobar esta spec. `boxtwin procesar` y `completar` por línea de comandos. Medir C1 | Cap 6 (vence 29-09) |
| 2 | 29-09 a 05-10 | Dockerfile del worker con los dos entornos y la etapa de clasificación. PoC en nube con la misma imagen (Experimento 1). Medir C2 | |
| 3 | 06-10 a 12-10 | API, PostgreSQL, tabla de trabajos. Indicador de guardia y C3 | Cap 7 (vence 13-10) |
| 4 | 13-10 a 19-10 | Frontend: subida, estado, siembra, Fight-Card | Cap 8 (vence 20-10) |
| 5 | 20-10 a 26-10 | Línea de tiempo con salto al video y corrección de tipo (F14), export, túnel, cuentas | |
| 6 | 27-10 a 02-11 | Sesiones de prueba con usuarios (C4, C5, C6) | |
| 7 | 03-11 a 09-11 | Correcciones, métricas finales, integración al documento | Entrega acumulativa |
| 8 | 10-11 a 16-11 | Congelamiento, ensayo de demo, plan B grabado | Presentación 17-11 |

Semanas 1, 3 y 4 cargan capítulo y desarrollo a la vez. Si algo se corre, se corre el
desarrollo, no la entrega.

---

## 9. Tres riesgos de ejecución

1. La siembra no alcanza sobre material nuevo. El amateur ya mostró el caso: guantes casi del
   mismo color (separación 0,133 contra un piso de 0,55) y grafo de coexistencia fragmentado.
   Mitigación: la abstención ya existe; se suma corrección manual por track como plan B de C1.
2. El indicador de guardia no valida. Es el único indicador nuevo y el que conecta con lo que
   Suárez mira primero. Mitigación: C3 se mide en la semana 3, con tiempo para sacarlo sin
   romper el resto; la Fight-Card de volumen y la línea de tiempo sostienen el MVP solas.
3. El calendario de tesis y la imagen Docker se comen el desarrollo. Además de tres
   capítulos en cuatro semanas, `boxtwin_mmaction` tiene mmcv pinneado contra CUDA 11.8, y
   compilarlo dentro de una imagen es donde suelen irse días. Mitigación adicional: si la
   imagen con los dos entornos no sale en la semana 2, la clasificación corre fuera del
   contenedor, en el entorno conda local, leyendo el mismo JSON.
   Sobre el calendario, la mitigación es el orden de construcción pone primero lo que también sirve a la
   tesis (el comando de inferencia alimenta Cap 7, el Docker alimenta el PoC del Cap 5 y el
   Cap 6).

---

## 10. Fuera de alcance, dicho explícitamente

Conexión y bloqueo, mapa de exposición, avatar 3D, Rival
Fantasma, modo en vivo, pagos, multi-organización, despliegue en nube como infraestructura.
Todo sigue en el alcance del producto evaluado en el Cap 5; nada de eso entra en el MVP.

## 11. Tareas de higiene del repo (semana 1)

- Actualizar `CLAUDE.md` secciones 4 a 6 y `docs/PROXIMOS_PASOS.md`: siguen en el estado del
  11-08 y no mencionan la identidad automática.
- Documentar en `docs/experiments/` el resultado de 82,4% de la partición sobre seis fuentes,
  que hoy vive solo en el mensaje del commit `0d50f34`.
