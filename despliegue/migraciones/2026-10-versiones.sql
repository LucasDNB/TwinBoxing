-- Versiones Individual y Gimnasio (prototipo/specs/versiones-26-10, 02 §8). Idempotente: se
-- puede correr dos veces. La API hace lo mismo sola al arrancar (db.py: columnas nulas,
-- create_all de las tablas nuevas y acceso.migrar para el indice y el relleno); esto es la
-- foto completa para PostgreSQL. En la estacion la base es SQLite y alcanza con reiniciar la
-- API, con la base respaldada antes en boxtwin.db.antes-de-versiones.bak (R-12).
--
-- Aditiva: el codigo anterior corre sobre la base migrada e ignora lo nuevo, asi que volver
-- atras es volver de commit y reiniciar.
BEGIN;

-- B19: cuentas y miembros. B29: cuentas.demo.
CREATE TABLE IF NOT EXISTS cuentas (
    id VARCHAR(32) PRIMARY KEY,
    nombre VARCHAR(120) NOT NULL,
    version VARCHAR(16) NOT NULL DEFAULT 'entrenador',
    boxeador_propio_id VARCHAR(32),
    demo BOOLEAN NOT NULL DEFAULT FALSE,
    creada TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_cuentas_version ON cuentas (version);
CREATE INDEX IF NOT EXISTS ix_cuentas_demo ON cuentas (demo);

CREATE TABLE IF NOT EXISTS miembros (
    id VARCHAR(32) PRIMARY KEY,
    cuenta_id VARCHAR(32) NOT NULL REFERENCES cuentas (id),
    usuario_id VARCHAR(32) NOT NULL UNIQUE REFERENCES usuarios (id),
    rol VARCHAR(16) NOT NULL DEFAULT 'titular',
    desde TIMESTAMPTZ NOT NULL DEFAULT now(),
    baja_en TIMESTAMPTZ,
    baja_por VARCHAR(32)
);
CREATE INDEX IF NOT EXISTS ix_miembros_cuenta_id ON miembros (cuenta_id);

-- B25: consentimiento por atleta; semilla es de B29.
CREATE TABLE IF NOT EXISTS consentimientos (
    id VARCHAR(32) PRIMARY KEY,
    cuenta_id VARCHAR(32) NOT NULL REFERENCES cuentas (id),
    boxeador_id VARCHAR(32) NOT NULL REFERENCES boxeadores (id),
    texto_version VARCHAR(40) NOT NULL,
    mayor_de_edad BOOLEAN NOT NULL DEFAULT TRUE,
    nota VARCHAR(200),
    registrado_por VARCHAR(32) NOT NULL REFERENCES usuarios (id),
    registrado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
    revocado_en TIMESTAMPTZ,
    revocado_por VARCHAR(32),
    semilla BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS ix_consentimientos_cuenta_id ON consentimientos (cuenta_id);
CREATE INDEX IF NOT EXISTS ix_consentimientos_boxeador_id ON consentimientos (boxeador_id);

-- Columnas nuevas, todas nulas.
ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS nombre VARCHAR(80);
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS cuenta_id VARCHAR(32) REFERENCES cuentas (id);
ALTER TABLE sesiones ADD COLUMN IF NOT EXISTS duracion_declarada_s DOUBLE PRECISION;
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS cuenta_id VARCHAR(32) REFERENCES cuentas (id);
ALTER TABLE boxeadores ADD COLUMN IF NOT EXISTS entrenador_id VARCHAR(32) REFERENCES usuarios (id);
ALTER TABLE correcciones ADD COLUMN IF NOT EXISTS usuario_id VARCHAR(32) REFERENCES usuarios (id);
CREATE INDEX IF NOT EXISTS ix_sesiones_cuenta_id ON sesiones (cuenta_id);
CREATE INDEX IF NOT EXISTS ix_boxeadores_cuenta_id ON boxeadores (cuenta_id);

-- Relleno: una cuenta Entrenador por usuario sin miembro, con su email como nombre. El id de la
-- cuenta sale del id del usuario (los dos son hex de 32), asi el relleno es determinista.
INSERT INTO cuentas (id, nombre, version, demo)
SELECT md5('cuenta-' || u.id), left(u.email, 120), 'entrenador', FALSE
FROM usuarios u
WHERE NOT EXISTS (SELECT 1 FROM miembros m WHERE m.usuario_id = u.id)
ON CONFLICT (id) DO NOTHING;

INSERT INTO miembros (id, cuenta_id, usuario_id, rol)
SELECT md5('miembro-' || u.id), md5('cuenta-' || u.id), u.id, 'titular'
FROM usuarios u
WHERE NOT EXISTS (SELECT 1 FROM miembros m WHERE m.usuario_id = u.id);

UPDATE sesiones s SET cuenta_id = m.cuenta_id
FROM miembros m WHERE s.cuenta_id IS NULL AND m.usuario_id = s.usuario_id;
UPDATE boxeadores b SET cuenta_id = m.cuenta_id
FROM miembros m WHERE b.cuenta_id IS NULL AND m.usuario_id = b.usuario_id;

-- B19: el nombre del peleador es unico por cuenta. El indice viejo (usuario_id, nombre) queda.
CREATE UNIQUE INDEX IF NOT EXISTS uq_boxeador_cuenta_nombre ON boxeadores (cuenta_id, nombre);

-- B20: la duracion de las sesiones existentes la copia la API al arrancar desde sesion.json
-- (acceso.completar_cuentas): esta en disco y no en la base, asi que SQL no la alcanza.

COMMIT;
