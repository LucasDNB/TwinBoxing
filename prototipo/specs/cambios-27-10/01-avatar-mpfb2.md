# 01 · Avatar MPFB2 y mapa de calor sobre la piel

Pedido (27-10): reemplazar el maniquí de primitivas (W23, W25) por una malla humana con rig,
exportada desde Blender con MPFB2, y pintar sobre la piel los golpes que le dirigen al peleador.
Decisión de producto de Lucas: se busca impacto visual, aunque la imagen sugiera más precisión de
la que el sistema tiene.

## 1. Qué cambia y qué no

- La pose sigue siendo la guardia ilustrativa de `lib/maniqui.ts` (DC-6). No hay keypoints 3D:
  B14 (`elevar_3d`) sigue bloqueado, y la spec pegada asumía un `poseToRigSolver` alimentado
  por pose 3D que no existe. El solver gira los huesos del rig hacia los segmentos del maniquí.
- El calor no es un dato nuevo: es `perfil.exposicion.zonas` (B16, W22), los golpes que le
  dirigen sus rivales por zona, con la corrección del sesgo del detector. Se pinta solo si el
  bloque `06.exposicion` es visible. No dice dónde pegaron: el sistema no establece contacto.
- El rótulo "ilustrativo, no reconstruido del video" queda.
- Sin `public/models/boxer_mpfb2.glb`, o si el GLB no trae los huesos, la escena cae al maniquí
  de primitivas, con el mismo calor.

## 2. Archivos

La spec pegada proponía `src/components/canvas/`, `src/systems/` y `src/shaders/`. Se adaptó a la
estructura del repo:

| Spec pegada | Acá |
|---|---|
| `BoxerScene.tsx` + `BoxerAvatarMPFB2.tsx` | `components/boxtwin/avatar/escena.ts` (Three.js, chunk aparte) y `Avatar3D.tsx` (montaje) |
| `HeatmapLegendOverlay.tsx` | `components/boxtwin/avatar/LeyendaCalor.tsx` |
| `poseToRigSolver.ts` | `lib/avatar/solverRig.ts` |
| `exposureCalculator.ts` | `lib/avatar/exposicion.ts` |
| `skinHeatmapShader.ts` | `lib/avatar/shaderCalor.ts` |
| utilidad de validación | `scripts/validar-glb.mjs` |

## 3. Contrato del GLB

`node scripts/validar-glb.mjs public/models/boxer_mpfb2.glb` lo revisa entero.

- Malla del cuerpo `Boxer_Body_Mesh` con skin (error si falta). Guantes `Boxer_Gloves_Mesh`
  (aviso si falta; toman el color del lado A o B).
- Huesos del rig "Game engine" de MPFB2: `pelvis`, `spine_01` a `spine_03`, `neck_01`, `head`,
  `clavicle_*`, `upperarm_*`, `lowerarm_*`, `hand_*`, `thigh_*`, `calf_*`, `foot_*` (`_l`, `_r`).
  Se aceptan con prefijo de armadura (`Armature_upperarm_l`).
- Material del cuerpo PBR: BaseColor, NormalMap y rugosidad (avisos si faltan).
- 15.000 a 35.000 triángulos (aviso fuera del rango, error sobre 60.000).
- Peso: aviso sobre 6 MB, error sobre 16 MB. El celular es el caso base.
- Sin Draco ni Meshopt: el GLTFLoader se usa sin decodificadores.
- La orientación no importa: el solver lee la derecha de los hombros y escala a la altura cargada.

## 4. Export desde Blender (a confirmar por Lucas al exportar)

1. MPFB2: nuevo humano, cuerpo de peleador. Borrar helpers.
2. Rig: "Game engine". Sin animaciones.
3. Piel: material de textura de imagen (el procedural no exporta a glTF). Texturas a 1024 o 2048.
4. Guantes: malla aparte pesada a `hand_l` y `hand_r`, nombrada `Boxer_Gloves_Mesh`.
5. Exportar glTF binario, +Y arriba, con skinning, sin compresión de malla.
6. Copiar a `prototipo/boxtwin-web-entrenador/public/models/boxer_mpfb2.glb` y correr el validador.

Los assets de MakeHuman son CC0: el GLB puede ir al repo y a la tesis.

## 5. Shader

Inyección en `onBeforeCompile` de `MeshStandardMaterial`. Se agregan líneas después de
`skinning_vertex` (posición deformada, llevada al espacio de la figura), `map_fragment` (el calor
se mezcla en el albedo) y `emissivemap_fragment` (un poco de emisión). Normal map, rugosidad y luz
siguen intactos. Cuatro zonas con meseta y caída suave; se pisan con `max`. Rampa térmica de azul
a rojo, la misma de la leyenda. La leyenda deja apagarlo y mover intensidad y umbral sin rearmar
la escena.

## 6. Pruebas

- `tests/unit/avatar-calor.test.ts` (13): zonas y lados, inyección sobre el shader real de Three
  0.170, solver sobre un rig en T mirando a +z, validador sobre GLB sintéticos.
- `perfil-tactico.test.tsx`: rótulo nuevo y leyenda con la bandera de exposición.
- Visto en Chromium (SwiftShader) con un arnés fuera del repo: compila con y sin skinning, y cae
  al maniquí sin el GLB. Con un GLB real de MPFB2 no se probó: no hay ninguno todavía.
