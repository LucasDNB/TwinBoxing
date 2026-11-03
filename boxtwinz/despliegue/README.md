# Desplegar BoxTwinZ en la estación

BoxTwinZ corre al lado de A, sin tocarlo: otro puerto (8003), otra base, otro directorio de
datos, otro archivo de entorno y otros servicios. Comparte la GPU (riesgo R1 en `TAREAS.md`).

| | A (producción) | BoxTwinZ |
|---|---|---|
| Puerto | 8000 | 8003 |
| Entorno | `/etc/boxtwin.env` | `/etc/boxtwinz.env` |
| Datos | `~/boxtwin/datos` | `~/boxtwinz-datos` |
| Servicios | `boxtwin-api`, `boxtwin-worker` | `boxtwinz-api`, `boxtwinz-worker` |
| Túnel | Funnel en 443 | Funnel en 10000 |

## Los pasos

```bash
cd ~/Proyectos/TwinBoxing          # con la rama de BoxTwinZ mergeada

# 0. CZ2: la huella de A antes de tocar nada
bash boxtwinz/despliegue/verificar_a.sh > ~/boxtwinz-datos/a-antes.txt

# 1. la web exportada
cd boxtwinz/web && npm ci && npm run build && cd ../..

# 2. datos y configuración (los valores [PROVISIONAL] van en el toml)
mkdir -p ~/boxtwinz-datos
cp boxtwinz/boxtwinz.toml.example ~/boxtwinz-datos/boxtwinz.toml
editor ~/boxtwinz-datos/boxtwinz.toml        # rutas de modelos: las mismas que A
sudo install -m 600 -o lucasb -g lucasb boxtwinz/despliegue/boxtwinz.env.ejemplo /etc/boxtwinz.env
sudo -u lucasb editor /etc/boxtwinz.env      # BOXTWINZ_SECRETO largo y BOXTWINZ_ADMINS

# 3. los servicios
sudo cp boxtwinz/despliegue/boxtwinz-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now boxtwinz-api boxtwinz-worker
curl -s localhost:8003/api/salud

# 4. datos de demo (Z18). Con los videos propios: --videos <carpeta con sombra.mp4, bolsa.mp4, sparring.mp4>
cd boxtwinz/api && BOXTWINZ_CONFIG=~/boxtwinz-datos/boxtwinz.toml \
  ~/miniforge3/envs/twinboxing_env/bin/python -m boxtwinz_api.demo sembrar --clave '<clave de demo>' && cd ../..

# 5. el túnel: Funnel solo admite 443, 8443 y 10000; 443 es de A y 8443 de B
tailscale funnel --bg --https=10000 8003
tailscale funnel status                      # https://desarrollo-lucas.tail52f6aa.ts.net:10000

# 6. CZ2: A no cambió
bash boxtwinz/despliegue/verificar_a.sh > ~/boxtwinz-datos/a-despues.txt
diff ~/boxtwinz-datos/a-antes.txt ~/boxtwinz-datos/a-despues.txt && echo "A igual"
```

No se usa una ruta (`/z`) dentro del dominio de A: la web exportada pide sus archivos desde la
raíz y habría que reconstruirla con `basePath`, y además pasaría por el proxy de A.

## Vuelta atrás

BoxTwinZ no modifica nada de A, así que volver es apagarlo:

```bash
tailscale funnel --https=10000 off
sudo systemctl disable --now boxtwinz-api boxtwinz-worker
sudo rm /etc/systemd/system/boxtwinz-*.service && sudo systemctl daemon-reload
```

Los datos quedan en `~/boxtwinz-datos` (base y videos) hasta que se borren a mano. Para volver
a una versión anterior de BoxTwinZ: `git checkout <commit>` del directorio `boxtwinz/`,
`npm run build` y `sudo systemctl restart boxtwinz-api boxtwinz-worker`. Antes de cambiar de
versión con datos reales, respaldar la base:
`cp ~/boxtwinz-datos/boxtwinz.db ~/boxtwinz-datos/boxtwinz.db.antes-de-<fecha>.bak`.

## Línea de base de A (3-11, antes de cualquier despliegue de BoxTwinZ)

```
83ec9468a5b0080c1dd378fa01dcf471124212689796f0c73596d56748155764  /etc/boxtwin.env
5dab81b50ef667d21e73fcf5f833d972487493ab638bb652fbb0917d9b191f19  /etc/systemd/system/boxtwin-api.service
6cc52efda647b9c6c722a70042ee3c0323e8a8f42251d02ee3fdf43d069b4e4e  /etc/systemd/system/boxtwin-worker.service
salud_A_commit ed54f1a
boxtwin-api active
boxtwin-worker active
```

Tomada después de la corrida real de Z4 (que usó un servidor temporal en 8003 y la GPU): A
siguió en el mismo commit y con sus servicios activos.
