# Reentrenar con 3,7× más datos movió la familia 0,001 — el hook no es un problema de datos

**07-09-2026 · 7 fuentes, 779/268 · prueba de prior contra separabilidad.**

## Qué se preguntaba

La propuesta era filmar sparring con mayoría de hooks para balancear las clases. La motivaba
una asimetría real: en la corrida de punta a punta, **29 de 68 hooks se llamaron straight** y
sólo 12 al revés. Esa firma es compatible con un problema de **prior de clase**, que se
arregla con datos o con pesos.

Pero también es compatible con un problema de **separabilidad**, que no se arregla con
ninguna de las dos. Se distinguen midiendo.

## Dos cosas que se hicieron primero, gratis

El clasificador en uso se había entrenado con **285 muestras de una sola fuente**, y con las
guardias mal puestas: 175 eventos tenían jab y cross intercambiados porque `arm_role` se
deriva de la guardia. Antes de conclusiones sobre el dominio, había que sacar eso del medio.

| | Clasificador anterior | Reentrenado |
|---|---|---|
| Fuentes | 1 | **7** |
| Train / val | 285 / 96 | **779 / 268** |
| Guardias | 175 eventos invertidos | corregidas |

## El resultado que decide

Dos configuraciones idénticas salvo por los pesos de clase — misma partición, misma semilla,
mismos hiperparámetros.

**Sin pesos**, acierto por familia **0,746**:

| Real | n | → straight | → hook | → uppercut | recall |
|---|---|---|---|---|---|
| straight | 167 | **158** | 6 | 3 | 0,946 |
| hook | 76 | **38** | 37 | 1 | 0,487 |
| uppercut | 25 | 12 | 8 | 5 | 0,200 |

**Con pesos** (inverso de la frecuencia), acierto por familia **0,694**:

| Real | n | → straight | → hook | → uppercut | recall |
|---|---|---|---|---|---|
| straight | 167 | 151 | 9 | 7 | 0,904 |
| hook | 76 | **45** | 22 | 9 | **0,289** |
| uppercut | 25 | 6 | 6 | **13** | **0,520** |

## Lo que dice, y se parte por clase

**El hook NO es un problema de prior.** Subirle el peso 2,6 veces respecto del jab **empeoró**
su recall, de 0,487 a 0,289. Si fuera prior, subir el peso lo habría movido en la dirección
contraria. La asimetría hook→straight sobrevive intacta: 38 contra 6 sin pesos, 45 contra 9
con pesos, razón 6,3 y 5,0.

**El uppercut SÍ lo era.** Los pesos solos lo llevaron de 0,200 a 0,520, sin un dato nuevo.
Ahí el desbalance de 7,79:1 era el problema.

## Y el número que cierra el caso

El acierto por familia del clasificador anterior, medido en la corrida de punta a punta, fue
**0,745**. El reentrenado con 3,7 veces más datos, siete fuentes en lugar de una y 175
etiquetas corregidas da **0,746**.

**Una milésima.**

Triplicar los datos, multiplicar por siete la diversidad de fuentes y arreglar el 17% de las
etiquetas no movió la familia del golpe. Es la evidencia más fuerte hasta ahora de que el
techo no está en el dataset.

## Contra qué hay que leerlo

Cuatro mediciones independientes, ahora cinco, sobre el mismo eje:

1. El anotador distingue hook de straight con kappa 0,881.
2. Cinco descriptores 2D, dos de ellos codificando la definición escrita del anotador: mejor
   AUC 0,638.
3. La matriz del clasificador entrenado en distribución.
4. La corrida de punta a punta: 29 de 68 hooks llamados straight.
5. **Reentrenar con 3,7× datos: +0,001.**

La distinción existe y un humano la hace. No está en la pose monocular.

## Consecuencia práctica

**No conviene filmar sparring con mayoría de hooks.** La evidencia dice que el problema no es
cuántos hooks hay, y montar drills además sería material *staged*, que es exactamente lo que
este proyecto rechazó de BoxingVI.

**Sí conviene aplicar los pesos de clase**, que son gratis y arreglan el uppercut. Y si se
filma algo, que sea material con uppercuts, que es donde el desbalance sí pesa.

**Y el hook queda como pregunta de sensor, no de dataset.** La explicación en pie sigue siendo
la proyección: un hook es un arco alrededor del eje vertical, y si ese arco cae en un plano que
contiene al eje óptico, en la imagen se ve como una recta. Eso no lo arregla más data ni otra
arquitectura sobre la misma entrada — lo arreglaría una segunda cámara.
