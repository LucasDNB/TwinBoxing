"""
BoxTwinZ - Los textos que la spec fija palabra por palabra (secciones 3, 8 y 12).

Viven en un solo lugar para que el test de textos obligatorios (CZ7) y la API lean lo mismo.
Cambiar un consentimiento es cambiar su version en boxtwinz.toml ademas del texto.
"""

CONSENTIMIENTO_USO = (
    "Tengo 18 años o más y cuento con la autorización de las personas que aparecen en los "
    "videos que cargo. No cargo videos con menores de edad."
)
CONSENTIMIENTO_REENTRENAMIENTO = (
    "Autorizo a usar mis videos y valoraciones para reentrenar el modelo. Lo que ya se usó en "
    "un modelo entrenado no puede retirarse de ese modelo."
)
PAGO_SIMULADO = "Pago simulado — versión de demostración"
PERFIL_SIMULADO = "Perfil simulado"
EN_EVALUACION = "Reconocimiento en evaluación"
ROUNDS_SIN_VALIDAR = "Comparación entre rounds sin validar"
GUARDIA_SUPUESTA = "guardia supuesta"
