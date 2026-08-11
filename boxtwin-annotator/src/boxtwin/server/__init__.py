"""
Servidor HTTP del anotador.

La aplicacion se usa desde el navegador de otra maquina, contra un puerto de la maquina que
tiene el cache de pose. No hay interfaz de escritorio: el equipo de trabajo es headless y
se accede por red.

Depende de core y de opencv. No depende de torch ni de ultralytics: anotar no necesita GPU.
"""
