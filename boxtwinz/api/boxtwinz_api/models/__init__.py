"""
BoxTwinZ API - Las tablas de la seccion 4 de la spec.

POR QUE EXISTE
  Un solo modulo para las 18 tablas: son chicas y se referencian entre si, y partirlas en
  archivos obliga a importar en un orden que SQLAlchemy no perdona.

QUE HACE
  Declara el modelo con SQLAlchemy 2. Todas llevan id (UUID en texto), creado y actualizado.

USO
  from boxtwinz_api.models import Base, Usuario, Perfil, Sesion
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "Base", "ahora", "Usuario", "Perfil", "Arquetipo", "Sesion", "Golpe", "MetricaSesion",
    "ExposicionSesion", "Valoracion", "Correccion", "Amistad", "Denuncia", "AccionModeracion",
    "TandaModeracion", "Trabajo", "UsoGpuDiario", "NotificacionAdmin", "EventoPublicidad",
    "UsoMinutos", "MODALIDADES", "ESTILOS", "ESTADOS_SESION", "ETAPAS",
]

MODALIDADES = ("sombra", "bolsa", "sparring")
ESTILOS = ("propio", "fajador", "estilista", "hibrido")
ESTADOS_SESION = ("subida", "procesando", "esperando_siembra", "completando", "lista", "error")
ETAPAS = ("procesar", "completar")


def ahora() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class _Comun:
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
    actualizado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora, onupdate=ahora)


class Usuario(_Comun, Base):
    __tablename__ = "usuarios"
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hash: Mapped[str] = mapped_column(String(255))
    nombre: Mapped[str] = mapped_column(String(120))
    plan: Mapped[str] = mapped_column(String(10), default="free")
    rol: Mapped[str] = mapped_column(String(10), default="usuario")
    consentimiento_uso_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consentimiento_uso_version: Mapped[str] = mapped_column(String(40))
    consentimiento_uso_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    consentimiento_reent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consentimiento_reent_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    suspendido: Mapped[bool] = mapped_column(Boolean, default=False)
    premium_intencion_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Perfil(_Comun, Base):
    __tablename__ = "perfiles"
    __table_args__ = (UniqueConstraint("usuario_id", "ranura"),)
    usuario_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id", ondelete="CASCADE"), index=True)
    ranura: Mapped[int] = mapped_column(Integer)
    nombre: Mapped[str] = mapped_column(String(120))
    estilo: Mapped[str] = mapped_column(String(12))
    guardia: Mapped[str | None] = mapped_column(String(10), nullable=True)
    publico: Mapped[bool] = mapped_column(Boolean, default=False)
    arquetipo_id: Mapped[str | None] = mapped_column(ForeignKey("arquetipos.id"), nullable=True)
    customizado: Mapped[bool] = mapped_column(Boolean, default=False)
    # Moderacion (spec 6.4): "el perfil no fue sacado del ranking".
    fuera_de_ranking: Mapped[bool] = mapped_column(Boolean, default=False)


class Arquetipo(_Comun, Base):
    __tablename__ = "arquetipos"
    clave: Mapped[str] = mapped_column(String(40), unique=True)
    nombre: Mapped[str] = mapped_column(String(120))
    estilo: Mapped[str] = mapped_column(String(12))
    variante: Mapped[str | None] = mapped_column(String(40), nullable=True)
    solo_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    metricas: Mapped[dict] = mapped_column(JSON, default=dict)
    exposicion: Mapped[dict] = mapped_column(JSON, default=dict)
    curva_ritmo: Mapped[list] = mapped_column(JSON, default=list)
    # Copia custom de un usuario premium (POST /arquetipos/custom). Null = del catalogo.
    usuario_id: Mapped[str | None] = mapped_column(ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=True)


class Sesion(_Comun, Base):
    __tablename__ = "sesiones"
    usuario_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id", ondelete="CASCADE"), index=True)
    perfil_id: Mapped[str] = mapped_column(ForeignKey("perfiles.id", ondelete="CASCADE"), index=True)
    modalidad: Mapped[str] = mapped_column(String(10))
    rival_perfil_id: Mapped[str | None] = mapped_column(ForeignKey("perfiles.id", ondelete="SET NULL"), nullable=True)
    video_ruta: Mapped[str] = mapped_column(String(512))
    video_duracion_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_ancho: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_alto: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estado: Mapped[str] = mapped_column(String(20), default="subida")
    error_etapa: Mapped[str | None] = mapped_column(String(20), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    rounds_declarados: Mapped[int | None] = mapped_column(Integer, nullable=True)
    apta_ranking: Mapped[bool] = mapped_column(Boolean, default=False)
    calidad_pose: Mapped[float | None] = mapped_column(Float, nullable=True)
    semilla_usuario: Mapped[int | None] = mapped_column(Integer, nullable=True)
    semilla_rival: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Lo que se desconto del cupo al encolar (spec 9), para poder devolverlo exacto.
    segundos_cupo: Mapped[float] = mapped_column(Float, default=0.0)
    # Proxy de "golpes sin identidad" para el panel (desvio D6): fraccion de cuadros sin asignar.
    fraccion_sin_identidad: Mapped[float | None] = mapped_column(Float, nullable=True)
    segundos_gpu: Mapped[float] = mapped_column(Float, default=0.0)
    vista_resultado: Mapped[bool] = mapped_column(Boolean, default=False)


class Golpe(_Comun, Base):
    __tablename__ = "golpes"
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id", ondelete="CASCADE"), index=True)
    quien: Mapped[str] = mapped_column(String(8))
    mano: Mapped[str] = mapped_column(String(10))
    t_inicio: Mapped[float] = mapped_column(Float)
    t_fin: Mapped[float] = mapped_column(Float)
    confianza: Mapped[float] = mapped_column(Float)
    altura_estimada: Mapped[str | None] = mapped_column(String(10), nullable=True)
    id_motor: Mapped[str | None] = mapped_column(String(40), nullable=True)


class MetricaSesion(_Comun, Base):
    __tablename__ = "metricas_sesion"
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id", ondelete="CASCADE"), index=True)
    quien: Mapped[str] = mapped_column(String(8))
    segmento: Mapped[int | None] = mapped_column(Integer, nullable=True)
    volumen: Mapped[float] = mapped_column(Float)
    golpes: Mapped[int] = mapped_column(Integer, default=0)
    pct_izquierda: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_mano_adelantada: Mapped[float | None] = mapped_column(Float, nullable=True)
    combinaciones_min: Mapped[float] = mapped_column(Float, default=0.0)
    pct_golpes_en_combinacion: Mapped[float | None] = mapped_column(Float, nullable=True)
    sosten_ritmo: Mapped[float | None] = mapped_column(Float, nullable=True)
    iniciativa: Mapped[float | None] = mapped_column(Float, nullable=True)
    curva: Mapped[list] = mapped_column(JSON, default=list)


class ExposicionSesion(_Comun, Base):
    __tablename__ = "exposicion_sesion"
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id", ondelete="CASCADE"), unique=True)
    rival_izquierda: Mapped[int] = mapped_column(Integer, default=0)
    rival_derecha: Mapped[int] = mapped_column(Integer, default=0)
    lado_izquierdo_usuario: Mapped[int] = mapped_column(Integer, default=0)
    lado_derecho_usuario: Mapped[int] = mapped_column(Integer, default=0)


class Valoracion(_Comun, Base):
    __tablename__ = "valoraciones"
    sesion_id: Mapped[str | None] = mapped_column(ForeignKey("sesiones.id", ondelete="SET NULL"), nullable=True, index=True)
    estrellas: Mapped[int] = mapped_column(Integer)
    comentario: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Para que sobreviva anonimizada a la baja (spec 3) sin perder lo que el panel agrega.
    modalidad: Mapped[str | None] = mapped_column(String(10), nullable=True)
    usada_en_modelo: Mapped[bool] = mapped_column(Boolean, default=False)


class Correccion(_Comun, Base):
    __tablename__ = "correcciones"
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id", ondelete="CASCADE"), index=True)
    golpe_id: Mapped[str | None] = mapped_column(ForeignKey("golpes.id", ondelete="CASCADE"), nullable=True)
    tipo: Mapped[str] = mapped_column(String(16))
    t: Mapped[float | None] = mapped_column(Float, nullable=True)


class Amistad(_Comun, Base):
    __tablename__ = "amistades"
    __table_args__ = (UniqueConstraint("solicitante_id", "destinatario_id"),)
    solicitante_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id", ondelete="CASCADE"))
    destinatario_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id", ondelete="CASCADE"))
    estado: Mapped[str] = mapped_column(String(10), default="pendiente")


class Denuncia(_Comun, Base):
    __tablename__ = "denuncias"
    denunciante_id: Mapped[str | None] = mapped_column(ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True)
    perfil_id: Mapped[str] = mapped_column(ForeignKey("perfiles.id", ondelete="CASCADE"))
    motivo: Mapped[str] = mapped_column(String(24))
    detalle: Mapped[str | None] = mapped_column(Text, nullable=True)
    estado: Mapped[str] = mapped_column(String(10), default="abierta")
    tanda_id: Mapped[str | None] = mapped_column(ForeignKey("tandas_moderacion.id"), nullable=True)


class AccionModeracion(_Comun, Base):
    __tablename__ = "acciones_moderacion"
    denuncia_id: Mapped[str] = mapped_column(ForeignKey("denuncias.id", ondelete="CASCADE"))
    admin_id: Mapped[str | None] = mapped_column(ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True)
    accion: Mapped[str] = mapped_column(String(16))
    nota: Mapped[str | None] = mapped_column(Text, nullable=True)


class TandaModeracion(_Comun, Base):
    __tablename__ = "tandas_moderacion"
    admin_id: Mapped[str | None] = mapped_column(ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True)
    inicio: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
    fin: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cantidad_revisada: Mapped[int] = mapped_column(Integer, default=0)


class Trabajo(_Comun, Base):
    __tablename__ = "trabajos"
    sesion_id: Mapped[str] = mapped_column(ForeignKey("sesiones.id", ondelete="CASCADE"), index=True)
    etapa: Mapped[str] = mapped_column(String(12))
    estado: Mapped[str] = mapped_column(String(10), default="en_cola", index=True)
    intento: Mapped[int] = mapped_column(Integer, default=1)
    parametros: Mapped[dict] = mapped_column(JSON, default=dict)
    tomado_por: Mapped[str | None] = mapped_column(String(120), nullable=True)
    tomado_en: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    inicio_gpu: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fin_gpu: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminado_en: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    log: Mapped[str | None] = mapped_column(Text, nullable=True)


class UsoGpuDiario(_Comun, Base):
    __tablename__ = "uso_gpu_diario"
    fecha: Mapped[date] = mapped_column(Date, unique=True)
    segundos_ocupados: Mapped[float] = mapped_column(Float, default=0.0)
    utilizacion: Mapped[float | None] = mapped_column(Float, nullable=True)


class NotificacionAdmin(_Comun, Base):
    __tablename__ = "notificaciones_admin"
    tipo: Mapped[str] = mapped_column(String(20))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    leida: Mapped[bool] = mapped_column(Boolean, default=False)


class EventoPublicidad(_Comun, Base):
    __tablename__ = "eventos_publicidad"
    usuario_id: Mapped[str | None] = mapped_column(ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True)
    ubicacion: Mapped[str] = mapped_column(String(40))
    tipo: Mapped[str] = mapped_column(String(12), default="impresion")


class UsoMinutos(_Comun, Base):
    __tablename__ = "uso_minutos"
    __table_args__ = (UniqueConstraint("usuario_id", "mes"),)
    usuario_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id", ondelete="CASCADE"))
    mes: Mapped[str] = mapped_column(String(7))
    segundos_procesados: Mapped[float] = mapped_column(Float, default=0.0)
