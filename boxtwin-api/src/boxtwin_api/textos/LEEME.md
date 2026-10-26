# Textos del consentimiento por atleta (B25, DV-5)

Acá va el texto que firma cada atleta del plantel en la versión Gimnasio, en un archivo por
versión: `consentimiento-<version>.md` (por ejemplo `consentimiento-2026-10-08.md`). La API toma
como vigente el de versión más alta en orden alfabético, y cada consentimiento registrado guarda
la versión que se aceptó.

**Este directorio no trae ningún texto a propósito.** El spec dice que B25 no inventa un texto
legal: tiene que ser el del 08-10 para C6, si Lucas lo confirma, u otro que él decida. Mientras
no haya ninguno, `GET /cuenta/consentimiento` y `POST /boxeadores/{id}/consentimiento` responden
409 con este camino, y la web dice que falta el texto. Los consentimientos que carga la versión de
prueba (`demo.py`) no dependen de esto: van como semilla, con la versión que diga `demo.yaml`.

Este archivo no cuenta como texto: el patrón es `consentimiento-*.md`.
