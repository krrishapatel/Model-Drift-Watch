from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class BatchOut(BaseModel):
    id: int
    timestamp: datetime
    source: str
    row_count: int
    schema_version: str
    is_reference: bool

    class Config:
        from_attributes = True


class DriftMetricOut(BaseModel):
    batch_id: int
    feature: str
    metric: str
    value: float
    comparison: str


class QualityMetricOut(BaseModel):
    batch_id: int
    feature: Optional[str]
    metric: str
    value: float


class ModelMetricOut(BaseModel):
    batch_id: int
    metric: str
    value: float
