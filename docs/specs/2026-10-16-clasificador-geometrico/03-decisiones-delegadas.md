# Decisiones delegadas para la noche

Lucas no está para contestar. Estas son las respuestas a lo que puede pasar. Cada vez que se
aplica una, se escribe en la bitácora (`<trabajo>/NOCHE.md`) con hora, qué pasó y cuál se
aplicó. Lo que no está acá y no se puede resolver sin decidir algo de fondo se anota como
pregunta abierta en `TAREAS.md` y se sigue con la próxima tarea independiente.

| Id | Si pasa esto | Se hace esto |
|---|---|---|
| D1 | Una tarea no pasa sus pruebas después de tres intentos de arreglo | Se marca `bloqueada` con el error y se sigue con las que no dependen de ella. |
| D2 | `huellas.py verificar` falla | Se frena todo. Nada más corre. Se escribe en `NOCHE.md` y en el resumen final. |
| D3 | El conjunto de golpes (1.047, conteos por fuente, `frame_dir`) no coincide con el de E1 | Se frena la parte de E4 a E5 que dependa de eso. No se ajusta el pre-registro: se documenta la diferencia. E6 sigue. |
| D4 | Faltan las sesiones de C8 nivel 2 o los `pred.pkl` de E3 | E6 o F quedan `no realizado`, con lo que falta. El resto sigue. |
| D5 | Hay otro proceso de cómputo en la GPU | E4, E4a y E5 entrenan en CPU. La elevación 3D espera de a 10 minutos hasta 2 horas; si sigue ocupada, corre en CPU. |
| D6 | MMPose no importa, la config no está o los pesos no se pueden bajar | E5 queda `no realizado` con el error exacto. No se instala nada ni se busca otro modelo. |
| D7 | La elevación 3D corre pero tarda más de 3 horas estimadas | Se mide la tasa en 50 golpes, se extrapola y, si pasa de 3 horas, se elevan solo las cuatro fuentes de C8 y se documenta. E5 se evalúa sobre esas cuatro, con la regla del pre-registro aplicada a los 4 folds (≥ 0,03 y 3 de 4). |
| D8 | Se encuentra un bug en la evaluación o en los datos después de ver números | Se arregla con prueba, se borran los resultados afectados, se corre de nuevo y el documento cuenta las dos corridas. |
| D9 | Un número sale sospechosamente alto (familia ≥ 0,90 o ocho clases ≥ 0,80) | Antes de escribirlo, se busca fuga: se verifica que la estandarización y el entrenamiento no vieron la fuente de prueba y se corre una corrida con etiquetas barajadas, que tiene que dar cerca del azar. Se documenta. |
| D10 | Algo del diseño es ambiguo | Se elige lo más simple que respete el pre-registro y se anota la elección en `TAREAS.md` como decisión, con el motivo. El pre-registro no se reinterpreta: si la ambigüedad está ahí, gana la lectura más estricta. |
| D11 | Tentación de mejorar después de ver: más épocas, otra arquitectura, otras features | No. Se anota como idea en "Próximos pasos" del documento. |
| D12 | Se termina todo antes de la mañana | Se corre `auditor-medicion` sobre los resultados, se escriben los documentos y las notas de tesis, y se para. No se empiezan experimentos nuevos. |
| D13 | Algo pide `git push`, borrar archivos o instalar paquetes | No se hace. Está denegado y no se busca otra vía. |
