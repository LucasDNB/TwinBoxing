"""
BoxTwin API - Las tablas.

POR QUE EXISTE ESTA COLA Y NO REDIS CON CELERY
  El MVP corre en un solo nodo, la estacion de desarrollo, con pocos usuarios y trabajos
  que duran minutos u horas. Una tabla con SELECT ... FOR UPDATE SKIP LOCKED alcanza y
  evita instalar, monitorear y justificar dos piezas mas. La transaccionalidad sale gratis:
  encolar el trabajo y crear la sesion pasan en el mismo commit, asi que no existe el
  estado "hay sesion y no hay trabajo".

QUE HACE
  Usuarios, sesiones de analisis, trabajos y correcciones. Los archivos pesados -video,
  cache de pose, Fight-Card- viven en disco y no en la base: son de la sesion, los escribe
  el worker y los lee la API del mismo filesystem, que en un nodo unico es el mismo.

  La correccion de tipo es la unica tabla que existe por una razon que no es operativa: es
  un dataset. Cada fila es una etiqueta sobre material que el modelo no vio, hecha por
  alguien que sabe, y es exactamente lo que le falta al clasificador para generalizar.

USO
  from boxtwin_api.modelos import Sesion, Trabajo, Usuario
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "Correccion",
    "ESTADOS_SESION",
    "ESTADOS_TRABAJO",
    "ETAPAS",
    "Sesion",
    "Trabajo",
    "Usuario",
    "ahora",
    "nuevo_id",
]

# Los mismos estados que escribe el comando de linea en sesion.json. Estan repetidos a
# proposito y no importados: la API tiene que poder correr sin el paquete del anotador
# instalado, y la copia se verifica en un test contra el original.
ESTADOS_SESION = (
    "subida", "en_cola", "procesando", "espera_siembra", "completando", "listo", "fallo",
)
ESTADOS_TRABAJO = ("en_cola", "tomado", "listo", "fallo")
ETAPAS = ("procesar", "completar", "clasificar")


def ahora() -> datetime:
    return datetime.now(timezone.utc)


def nuevo_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Usuario(Base):
    __tablename__ = "usuarios"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=nuevo_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hash_clave: Mapped[str] = mapped_column(String(255))
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)

    sesiones: Mapped[list[Sesion]] = relationship(back_populates="usuario")


class Sesion(Base):
    """Un video subido y lo que el sistema saco de el."""

    __tablename__ = "sesiones"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=nuevo_id)
    usuario_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id"), index=True)
    nombre: Mapped[str] = mapped_column(String(255))
    estado: Mapped[str] = mapped_column(String(32), default="subida", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    video_nombre: Mapped[str] = mapped_column(String(255))
    duracion_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    round_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    descanso_s: Mapped[float] = mapped_column(Float, default=60.0)

    # Lo que el usuario contesto en el paso 4. Se guarda en la base y no solo en el
    # directorio porque es la unica decision humana del flujo y tiene que quedar auditable
    # aunque alguien borre la sesion de disco.
    semilla_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    semilla_b: Mapped[int | None] = mapped_column(Integer, nullable=True)

    creada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
    actualizada: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=ahora, onupdate=ahora
    )

    usuario: Mapped[Usuario] = relationship(back_populates="sesiones")
    trabajos: Mapped[list[Trabajo]] = relationship(
        back_populates="sesion", cascade="all, delete-orphan"
    )


class Trabajo(Base):
    """
    Una etapa a ejecutar. La cola.

    `tomado_en` no es telemetria: es lo que permite devolver a la cola un trabajo cuyo
    worker se murio. Sin eso, un kill en medio del preproceso deja la sesion colgada y
    nadie se entera hasta que el usuario pregunta.
    """

    __tablename__ = "trabajos"
    __table_args__ = (
        Index("ix_trabajos_cola", "estado", "creado"),
        UniqueConstraint("sesion_id", "etapa", "intento", name="uq_trabajo_intento"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=nuevo_id)
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id"), index=True)
    etapa: Mapped[str] = mapped_column(String(32))
    estado: Mapped[str] = mapped_column(String(16), default="en_cola", index=True)
    intento: Mapped[int] = mapped_column(Integer, default=1)
    parametros: Mapped[dict] = mapped_column(JSON, default=dict)

    tomado_por: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tomado_en: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminado_en: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)

    sesion: Mapped[Sesion] = relationship(back_populates="trabajos")


class Correccion(Base):
    """
    Una correccion de tipo hecha por el entrenador. F14 y RF10.

    No pisa nada: la etiqueta original queda en `tipo_original` junto con el checkpoint que
    la produjo, porque una etiqueta sin saber contra que modelo se midio no entra a un
    dataset.
    """

    __tablename__ = "correcciones"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=nuevo_id)
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id"), index=True)
    golpe: Mapped[str] = mapped_column(String(64))
    tipo: Mapped[str] = mapped_column(String(32))
    tipo_original: Mapped[str | None] = mapped_column(String(32), nullable=True)
    checkpoint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    datos: Mapped[dict] = mapped_column(JSON, default=dict)
    creada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
