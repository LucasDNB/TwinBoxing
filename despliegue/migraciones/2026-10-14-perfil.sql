-- Cambios del 14-10 (prototipo/specs/cambios-14-10/01 §3, B17). Idempotente. La API agrega
-- sola las columnas nulas al arrancar (db.py); esto es la foto completa para PostgreSQL.
BEGIN;

-- Distancia de trabajo y estilo, cargados a mano en el boxeador.
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS distancia VARCHAR(8);
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS estilo VARCHAR(16);

COMMIT;
