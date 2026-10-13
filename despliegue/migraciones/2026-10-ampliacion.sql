-- Ampliacion del 13-10 (prototipo/specs/ampliacion-13-10). Idempotente: se puede correr dos
-- veces. La API agrega sola las columnas nulas al arrancar (db.py); esto es la foto completa
-- para PostgreSQL.
BEGIN;

-- B11: la correccion puede ser de tipo o de altura (objetivo). Las filas anteriores son de tipo.
ALTER TABLE correcciones ADD COLUMN IF NOT EXISTS campo VARCHAR(16);
UPDATE correcciones SET campo = 'tipo' WHERE campo IS NULL;

COMMIT;
