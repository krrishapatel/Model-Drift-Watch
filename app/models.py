from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base


class Batch(Base):
    __tablename__ = "batches"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    schema_version = Column(String, nullable=False)
    schema_json = Column(JSON, nullable=False)
    is_reference = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    quality_metrics = relationship("QualityMetric", back_populates="batch")
    drift_metrics = relationship("DriftMetric", back_populates="batch")
    model_metrics = relationship("ModelMetric", back_populates="batch")
    alerts = relationship("Alert", back_populates="batch")


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    type = Column(String, nullable=False)


class QualityMetric(Base):
    __tablename__ = "quality_metrics"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    feature_id = Column(Integer, ForeignKey("features.id"), nullable=True)
    metric = Column(String, nullable=False)
    value = Column(Float, nullable=False)

    batch = relationship("Batch", back_populates="quality_metrics")


class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    feature_id = Column(Integer, ForeignKey("features.id"), nullable=False)
    metric = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    comparison = Column(String, nullable=False)  # reference or rolling

    batch = relationship("Batch", back_populates="drift_metrics")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    metric = Column(String, nullable=False)
    value = Column(Float, nullable=False)

    batch = relationship("Batch", back_populates="model_metrics")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    status = Column(String, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    batch = relationship("Batch", back_populates="alerts")


class ReferenceStat(Base):
    __tablename__ = "reference_stats"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    feature_id = Column(Integer, ForeignKey("features.id"), nullable=False)
    stats_json = Column(JSON, nullable=False)


class RollingStat(Base):
    __tablename__ = "rolling_stats"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    feature_id = Column(Integer, ForeignKey("features.id"), nullable=False)
    stats_json = Column(JSON, nullable=False)
