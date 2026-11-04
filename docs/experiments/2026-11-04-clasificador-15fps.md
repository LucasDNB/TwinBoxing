# Clasificador con pose a 15 fps

4-11-2026 · pre-registro: `docs/specs/2026-11-04-cierre/00-preregistro.md` §2 · datos:
`docs/experiments/datos/2026-11-04-clasificador-15fps.json` · código: `scripts/mejoras/cierre_15fps.sh` y
`features2d.diezmar_serie`

## Veredicto

**El perfil B es viable para el clasificador, por la regla.** La exactitud de familia del ensamble cae
de 0,732 a 0,718: 0,013, contra un máximo de 0,03.

Hay una salvedad: el promedio esconde al uppercut, cuyo recall baja de 0,541 a 0,412.

## Resultados

E4 completo, 35 corridas, 1047 golpes, siete fuentes dejando una afuera.

| | Base 30 fps | Muestreo 15 fps |
|---|---|---|
| Familia, exactitud | 0,732 (Wilson 0,704 a 0,758) | 0,718 (0,690 a 0,745) |
| Familia, balanceada | 0,654 | 0,617 |
| Recall straight (n 662) | 0,813 | 0,789 |
| Recall hook (n 300) | 0,607 | 0,650 |
| Recall uppercut (n 85) | 0,541 | 0,412 |
| Pasos usables | 0,842 | 0,839 |

| Fuente | Base | 15 fps |
|---|---|---|
| Sparring | 0,605 | 0,518 |
| 01-sparring | 0,656 | 0,688 |
| 02-sparring | 0,790 | 0,823 |
| 03-sparring | 0,767 | 0,736 |
| 04-sparring | 0,833 | 0,758 |
| sparring-3-rounds | 0,727 | 0,730 |
| pacquiao_margarito | 0,752 | 0,772 |

Los cambios por fuente van de −0,088 a +0,032. Es más ruido del que indica la caída total.

## Limitaciones

1. **Simulado.** Se descartan cuadros de la pose a 30 fps y se interpola. Una corrida real a 15 fps con
   tracker podría perder más.
2. **Solo E4.** No se midió el detector de golpes, que es lo que encuentra los golpes que E4 clasifica.
   Para decidir B entero falta eso. Tampoco identidad.
3. **E4 se reentrenó con la pose diezmada.** Un modelo de 30 fps servido a 15 fps sin reentrenar es otra
   medición.
4. **El uppercut tiene n 85.** La caída es grande, pero la muestra es chica. La regla no la mira.
5. **Sin consecuencia en la estación.** R-11 dio que B colapsa en A: esto importa si una máquina más
   débil, como la notebook sin medir, necesita 15 fps.
