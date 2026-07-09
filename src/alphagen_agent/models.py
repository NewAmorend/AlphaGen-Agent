from __future__ import annotations

import enum
from datetime import datetime
from pydantic import BaseModel, Field


class AlphaStatus(str, enum.Enum):
    GENERATED = "generated"
    BACKTESTING = "backtesting"
    EVALUATED = "evaluated"
    HIGH_QUALITY = "high_quality"
    SUBMITTED = "submitted"
    FAILED = "failed"


class GenerationStrategy(str, enum.Enum):
    LLM = "llm"
    TEMPLATE = "template"
    FACTOR_MINING = "factor_mining"


class QualityGrade(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECT = "reject"


class AlphaRecord(BaseModel):
    id: int | None = None
    expression: str
    strategy: GenerationStrategy = GenerationStrategy.LLM
    llm_model: str | None = None
    status: AlphaStatus = AlphaStatus.GENERATED
    created_at: datetime = Field(default_factory=datetime.now)
    submitted_at: datetime | None = None


class BacktestResult(BaseModel):
    id: int | None = None
    alpha_id: int
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "INDUSTRY"
    sharpe: float | None = None
    turnover: float | None = None
    fitness: float | None = None
    returns: float | None = None
    drawdown: float | None = None
    grade: QualityGrade | None = None
    checks: list[dict] | None = None
    wq_alpha_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class SimulationSettings(BaseModel):
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "INDUSTRY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unit_handling: str = "VERIFY"
    nan_handling: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False


class SimulationRequest(BaseModel):
    type: str = "REGULAR"
    settings: SimulationSettings = Field(default_factory=SimulationSettings)
    regular: str


class WQDataField(BaseModel):
    id: str
    description: str = ""
    dataset: str | None = None
    type: str | None = None


class WQOperator(BaseModel):
    name: str
    category: str = ""
    type: str = "SCALAR"
    definition: str = ""
    description: str = ""
