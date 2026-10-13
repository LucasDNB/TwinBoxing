-- Migracion de la web entrenador (prototipo/specs/05-api-y-datos.md, B1, B3 y B6).
--
-- POR QUE EXISTE
--   create_all no altera tablas existentes. db.py agrega solo las columnas nulas que faltan
--   (ADD COLUMN sin default ni indice), que alcanza para arrancar, pero no deja el valor por
--   defecto de contexto ni los indices. Esto deja la base de la estacion igual a la que
--   crearia el modelo desde cero.
--
-- USO
--   psql "$BOXTWIN_DB" -f despliegue/migraciones/2026-10-web-entrenador.sql
--   Es idempotente: se puede correr de nuevo sin romper nada.

BEGIN;

-- B1: contexto de la sesion y consentimiento
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS contexto VARCHAR(16);
ALTER TABLE sesiones ALTER COLUMN contexto SET DEFAULT 'sparring';
UPDATE sesiones SET contexto = 'sparring' WHERE contexto IS NULL;
CREATE INDEX IF NOT EXISTS ix_sesiones_contexto ON sesiones (contexto);
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS practica VARCHAR(24);
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS rounds INTEGER;
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS consentimiento_en TIMESTAMP WITH TIME ZONE;
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS consentimiento_por VARCHAR(320);
-- Las sesiones existentes quedan con consentimiento nulo: no se declaro y no se inventa.

-- B6: datos del peleador
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS categoria VARCHAR(40);
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS altura_m DOUBLE PRECISION;
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS envergadura_m DOUBLE PRECISION;

-- B3: asignaciones manuales de tramos (la crea create_all si no existe; va aca para que la
-- migracion sea la foto completa)
CREATE TABLE IF NOT EXISTS asignaciones_manuales (
    id VARCHAR(32) PRIMARY KEY,
    sesion_id VARCHAR(32) NOT NULL REFERENCES sesiones (id),
    track INTEGER NOT NULL,
    rol VARCHAR(8) NOT NULL,
    usuario_id VARCHAR(32) NOT NULL REFERENCES usuarios (id),
    creada TIMESTAMP WITH TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_asignaciones_manuales_sesion_id ON asignaciones_manuales (sesion_id);

COMMIT;
