# 06 · Plan de construcción

Del 28-10 al 03-11, que es el congelamiento de funciones del Cap 6. Después del 03-11 solo se
corrige. La regla del MVP sigue: si algo se corre, se corre el desarrollo y no la entrega.

Encima de esto, en el mismo calendario:

- Las sesiones de prueba de la semana del 27-10 (C4, C5, C6), que no se repiten y van primero.
- El corte de la sub-hipótesis 1 el 31-10 (R-01) y el benchmark de perfiles (R-11), que siguen
  pendientes y usan la GPU.
- La práctica de presentación del martes 03-11 y la integración de todo al documento para el 17-11.

Esfuerzo en días de una persona con Claude Code, en la misma escala que `prototipo/specs/07`. Esa
escala sumaba 24,5 días para W0 a W12 y B0 a B8, que se construyeron el 12 y el 13-10 (CLAUDE.md
§5), así que es probable que estos números estén igual de altos. Lo que no se achica es el tiempo de Lucas para revisar, filmar y
probar en el celular, que es el que de verdad limita.

---

## 1. Regla de despliegue

Nada de estos specs se despliega en la estación mientras haya sesiones de prueba por delante. El
desarrollo va en una rama (`versiones-26-10`) con PR contra `main`, y cada tanda se despliega una
sola vez, después de su compuerta, con la base respaldada (R-12) y fuera del horario de una sesión
o de una medición. Si la última sesión de prueba cae después del 30-10, el despliegue de la tanda 1
espera.

---

## 2. Tareas en orden

### Tanda 0 · Specs (26 y 27-10)

| Id | Tarea | Hecho cuando |
|---|---|---|
| — | Lucas revisa estos specs y decide DV-1 a DV-11 de `00` | Decisiones anotadas en `00`, con fecha. Las que no decida quedan con la recomendación, y se dice |

### Tanda 1 · Cuentas e Individual (28 al 30-10)

| Id | Tarea | Lee | Días | Hecho cuando |
|---|---|---|---|---|
| B19 | Cuentas, miembros y acceso; migración | `02` §3, §4, §8; `04` B19 | 1 | Plan aprobado antes del código; I1, prueba estructural y CV2 |
| B20 | Horas del mes | `02` §6; `04` B20 | 0,5 | I5 |
| B22 | Reglas de Individual | `04` B22 | 0,5 | I4 |
| B29 | Ingreso de prueba: ruta, perfiles, `demo.py`, `demo.yaml.ejemplo` | `02` §11; `04` B29 | 1 | I3 y la tercera parte de la prueba estructural |
| W27 | Cuenta y navegación por versión | `03` §1 | 0,5 | CA-27.1, CA-27.2 |
| W38 | Ingreso con un botón por versión, rótulo de prueba y "Salir" | `03` §2 | 0,5 | CA-00.7 a CA-00.10 |
| W29 | Mi progreso | `03` I-01 | 1 | CA-I01.1 a CA-I01.3 |
| W30 | Subir y marcarse en Individual | `03` I-02, I-03 | 0,5 | CA-I02.1 a CA-I03.2 |
| W31 | Sesión, Fight-Card y perfil en segunda persona | `03` I-04 a I-06 | 0,5 | CA-I05.1, CA-I06.1 |

Compuerta de la tanda 1 (viernes 30-10): CV1, CV2, I3, I4, I5 y E1. Si pasa: respaldo de la base,
migración y despliegue a la noche, con `BOXTWIN_DEMO` apagada. Después, Lucas arma
`~/boxtwin/demo.yaml` con las sesiones de DV-9 y corre `sembrar`. CV6 y CV3 al día siguiente, con
la bandera encendida solo para eso. Los dos botones de Gimnasio quedan en "En construcción".

Medición de CV4 (DV-6), en paralelo: Lucas filma los dos videos en su gimnasio el 28 o el 29-10;
se procesan y se revisan el 30 o el 31-10, con la GPU libre. Cerca de una hora de Lucas.

### Tanda 2 · Gimnasio (31-10 al 02-11), solo si la tanda 1 pasó su compuerta

| Id | Tarea | Lee | Días | Hecho cuando |
|---|---|---|---|---|
| B24 | Plantel compartido, autoría y lista de miembros | `04` B24 | 0,5 | I2 |
| B25 | Consentimiento por atleta | `04` B25 | 1 | Pruebas de B25. No arranca sin el texto (DV-5) |
| W32 | Equipo, en su forma de la versión de prueba | `03` G-09 | 0,25 | CA-G09.0 |
| W33 | Plantel, consentimiento y subida en Gimnasio | `03` G-02, G-07, G-08 | 1 | CA-G02.1, CA-G07.1, CA-G08.1 |
| W34 | Panel del gimnasio y autoría en la sesión | `03` G-01, §4 sesión | 0,5 | CA-G01.1, CA-G01.2, CA-G04.1 |

Compuerta de la tanda 2 (lunes 02-11 al mediodía): CV1, CV2, I2, I3, E2 y E3. Si pasa: respaldo,
despliegue esa tarde con los dos perfiles de Gimnasio disponibles, `reiniciar` para sembrar el
gimnasio con sus consentimientos, y CV5 en la estación. Si no pasa al mediodía del 02-11, no se
despliega: el 03-11 hay práctica de presentación y no conviene llegar con un despliegue de la
noche anterior. Los botones de Gimnasio siguen en "En construcción".

### P1, si sobra tiempo antes del 03-11

| Id | Tarea | Lee | Días |
|---|---|---|---|
| B27 | Posición en la cola | `04` B27 | 0,5 |
| W35 | Posición en la cola en la espera de la sesión | `03` G-02 | 0,25 |

### P2, especificadas y sin construir

| Id | Tarea | Lee |
|---|---|---|
| B26 | Supresión de un atleta | `04` B26 |
| W36 | Pantalla de supresión en la ficha del atleta | `04` B26 |
| B28 | Cambio de Individual a Entrenador | `04` B28 |
| W37 | Pasar a Entrenador desde Mi perfil | `04` B28 |

### Postergadas por la versión de prueba (pedido del 26-10)

Vuelven cuando vuelvan el email y las invitaciones. No tienen fecha.

| Id | Tarea | Lee |
|---|---|---|
| B21 | Registro por versión | `04` B21 |
| W28 | Registro por versión y entrada por invitación | `03` §2, registro e invitación |
| B23 | Invitaciones y bajas de miembros | `02` §7; `04` B23 |
| W32 (resto) | Invitar, revocar invitaciones y dar de baja en Equipo | `03` G-09 |

---

## 3. Si no alcanza, qué se recorta y en qué orden

1. La tanda 2 entera, antes de tocar nada de la tanda 1. Gimnasio queda especificado y sin
   construir, que es lo que el Cap 6 preveía para lo post-MVP.
2. Dentro de la tanda 2, si hay que recortar: primero la actividad por entrenador del panel,
   después el entrenador a cargo, después el CSV de consentimientos. B25 entero puede caer y
   Gimnasio quedar con el consentimiento por sesión, igual que Entrenador; se declara.
3. Lo que no se recorta: CV1 y CV2 antes que cualquier despliegue, e I3 y CV6 antes de encender la
   versión de prueba en la estación. Una versión con una fuga entre cuentas no se presenta, y un
   ingreso sin credenciales que alcance una cuenta real tampoco.
4. Si B29 no llega, el ingreso queda como hoy, con email y clave, y las versiones se muestran
   entrando con cuentas creadas a mano. Es la contingencia, no el plan.
5. Dentro de la tanda 1, la tarjeta de tendencia de Mi progreso puede quedar como la lista de
   sesiones con sus golpes por minuto. El resto de la tanda es chico y no tiene recorte útil.

---

## 4. Qué va a cada capítulo de la tesis

Todo en la entrega final del 17-11, con el formato de edición de siempre (ANTES / DESPUÉS / QUÉ
CAMBIÓ) y solo si se construye. Si no se construye, lo que va es una línea en el Cap 8.

| Capítulo | Qué cambia |
|---|---|
| 4, §2.7 (MVP) | F13 entra en parte: planes y multi-organización sin pagos. Las dos versiones nuevas, con fecha y motivo, como segunda ampliación después de la del 13-10 |
| 4, arquitectura y seguridad | Modelo de cuentas, control de acceso y la tabla de seguridad de `02` §9, que responde a lo que la Clase 8 pide (autenticación, control de accesos, riesgos que cubre cada medida) |
| 4, Anexo I | `BOXTWIN_CUPO`, `BOXTWIN_DEMO` y `BOXTWIN_DEMO_SUBIR`, cómo sembrar y reiniciar las cuentas de prueba, la migración y el respaldo previo |
| 5 | Ninguna cifra cambia (decisión del 15-09). A lo sumo una oración en la nota al cierre: tres de los cuatro planes de la Tabla 5.7 quedaron implementados en el prototipo sin pagos ni modo en vivo, y parte de la línea "backend multiusuario" de la Tabla 5.2 se adelantó en el PFI. No se descuenta del flujo porque el flujo se rehace con la alternativa B |
| 6 | R-03 revisado: segunda ampliación a ocho días del congelamiento. R-04: el ingreso con un botón suma a la contramedida, porque frente al tribunal no se tipea ninguna clave. R-09: la declaración de mayoría de edad y el material con permiso para mostrarse (DV-9) suman a la contramedida. Cuatro riesgos nuevos, propuestos abajo |
| 7 | Requisitos RF-V, RF-D, RNF-V y RNF-D (`01` §5), diseño (`02`), construcción (`prototipo/TAREAS.md`), pruebas técnicas (CV1, CV2, CV6, I1 a I5) y de aceptación (CV3 a CV5), con fecha y commit. Que lo que se presenta es una versión de prueba con cuentas compartidas, que el registro y las invitaciones quedaron especificados y sin construir, y que ninguna de las dos versiones tuvo prueba con usuarios. Sin capturas de código ni de pantalla, como pide la Clase 12 |
| 8 | Implicancia: el registro transferible entre entrenadores del Cap 5 §5.2 ya existe como software. Líneas futuras: acceso del atleta a su perfil, consentimiento de representantes para menores, supresión, varias sedes por usuario, modo en vivo |
| Declaración de uso de IA | Estos specs los redactó Claude el 26-10 a partir de los dos pedidos de Lucas de ese día. Qué decisiones DV tomó Lucas y cuáles quedaron con la recomendación por defecto |

Riesgos propuestos para el registro del Cap 6, con la redacción de la Clase 11. R-21 a R-23 ya
están propuestos desde el 13-10, así que siguen R-24 en adelante:

| Id | Debido a... podría ocurrir... lo que provocaría... | Naturaleza y etapa | P | I | Exp. | Respuesta | Contingencia |
|---|---|---|---|---|---|---|---|
| R-24 | Debido a que B19 cambia el control de acceso de todas las rutas a una semana del congelamiento, podría ocurrir que una sesión quede a la vista de otra cuenta o deje de verse para la suya, lo que provocaría exponer datos personales de atletas o romper la demo | Datos y cumplimiento. Pasaje a producción | 2 | 5 | 10, media | Mitigar: CV1, CV2, respaldo previo | Volver al commit anterior con la base respaldada |
| R-25 | Debido a que la versión Individual se apoya en bolsa y sombra, que no están validadas (B8), podría ocurrir que cuente mal justo en su caso principal, lo que provocaría presentar una versión cuyo número central no está sostenido | Técnico. Durante | 3 | 3 | 9, media | Mitigar: medir CV4 | Rótulo de función sin validar y declaración en el Cap 7 |
| R-26 | Debido a que la versión Gimnasio no tiene prueba con usuarios antes del 17-11, podría ocurrir que el plantel compartido no se ajuste a cómo trabaja un gimnasio, lo que provocaría que su valor quede sin evidencia | Alcance. Operación | 3 | 2 | 6, media | Aceptar, con responsable el autor | Declararla como probada técnicamente y sin usuarios, y llevar la validación a líneas futuras |
| R-27 | Debido a que la versión de prueba deja entrar sin credenciales por un túnel público, podría ocurrir que alguien con la URL suba videos o mire el material cargado durante una ventana abierta, lo que provocaría llenar la cola de la única GPU en plena demo o mostrar datos de personas filmadas | Datos y cumplimiento. Pasaje a producción | 2 | 4 | 8, media | Mitigar: bandera apagada por defecto y encendida solo en ventanas (DV-10), cupo que bloquea, solo material con permiso y seudónimos (DV-9), CV6 | `BOXTWIN_DEMO_SUBIR=0` y, si hace falta, apagar la bandera y presentar con la demo local en la notebook (contingencia de R-04) |
