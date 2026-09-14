from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("An explicit timezone is required for information availability.")
    return value.astimezone(timezone.utc)


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DocumentRecord(StrictRecord):
    document_id: str
    source: str
    source_type: Literal["filing", "official_release", "official_speech", "news_discovery", "position_report"]
    title: str
    entities: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    asset_tags: list[str] = Field(default_factory=list)
    cik: int | None = None
    form: str | None = None
    items: list[str] = Field(default_factory=list)
    section: str | None = None
    accession: str | None = None
    report_period: str | None = None
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    retrieved_at: datetime
    source_url: str
    raw_path: str
    text: str = Field(min_length=30)
    text_hash: str
    duplicate_group: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", "accepted_at", "available_at", "retrieved_at", mode="before")
    @classmethod
    def timestamps(cls, value):
        return utc(value) if value is not None else value

    @model_validator(mode="after")
    def chronology(self):
        known = [v for v in [self.published_at, self.accepted_at] if v is not None]
        if known and self.available_at < max(known):
            raise ValueError("available_at precedes publication or acceptance.")
        if self.available_at > self.retrieved_at:
            raise ValueError("A document cannot be retrieved before it is available.")
        return self


class DocumentChunk(StrictRecord):
    chunk_id: str
    document_id: str
    source: str
    content_date: str | None = None
    title: str
    entities: list[str]
    tickers: list[str]
    form: str | None
    section: str
    subsection: str = ""
    available_at: datetime
    order: int
    token_count: int
    token_method: str
    text: str
    text_hash: str
    duplicate_group: str

    @field_validator("available_at", mode="before")
    @classmethod
    def timestamp(cls, value):
        return utc(value)


class ContextSnapshot(StrictRecord):
    name: str
    as_of: datetime
    latest_data_at: datetime | None
    available_at: datetime | None
    freshness: Literal["current", "stale", "unavailable"]
    measures: dict[str, Any]
    dependencies: list[str]
    version: str = "context-v1"
    notes: list[str] = Field(default_factory=list)

    @field_validator("as_of", "latest_data_at", "available_at", mode="before")
    @classmethod
    def timestamp(cls, value):
        return utc(value) if value is not None else value

    @model_validator(mode="after")
    def availability(self):
        if self.available_at is not None and self.available_at > self.as_of:
            raise ValueError("Context contains future information.")
        return self


class EventRecord(StrictRecord):
    event_id: str
    available_at: datetime
    family: str
    event_type: str
    entities: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    asset_tags: list[str] = Field(default_factory=list)
    what_happened: str
    what_changed: str
    direction: str | None = None
    importance: Literal["low", "medium", "high"]
    document_ids: list[str]
    evidence_ids: list[str]
    task_version: str = "event-v1"
    model_version: str | None = None
    prompt_version: str = "analyst-v1"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("available_at", mode="before")
    @classmethod
    def timestamp(cls, value):
        return utc(value)


class Claim(StrictRecord):
    statement: str
    evidence_ids: list[str] = Field(min_length=1)
    kind: Literal["fact", "interpretation", "uncertainty"]


class QueryPlan(StrictRecord):
    entities: list[str] = Field(default_factory=list, max_length=4)
    sources: list[Literal["sec", "fed", "bls", "bea", "eia", "cftc", "gdelt"]] = Field(default_factory=list)
    contexts: list[Literal["market", "risk", "volatility", "rates", "financial_conditions", "factors",
                           "cross_asset", "fundamentals", "credit", "macro"]] = Field(default_factory=list)
    queries: list[str] = Field(min_length=1, max_length=5)
    sections: list[str] = Field(default_factory=list, max_length=5)
    lookback_days: int = Field(default=90, ge=1, le=730)


class AnalysisTarget(StrictRecord):
    conclusion: str
    materiality: Literal["low", "medium", "high", "uncertain"]
    claims: list[Claim] = Field(min_length=1)
    what_changed: str
    why_it_matters: str
    uncertainty: list[str] = Field(min_length=1)


class TrainingExample(StrictRecord):
    example_id: str
    task: Literal["event", "sec_change", "macro", "reconciliation", "market"]
    group_id: str
    cutoff: datetime
    entities: list[str]
    source_ids: list[str]
    source_hashes: dict[str, str]
    template_version: str = "analyst-v1"
    quality_status: Literal["candidate", "review", "accepted", "rejected"] = "candidate"
    creation_method: str
    anchor: bool = False
    review: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, str]]

    @field_validator("cutoff", mode="before")
    @classmethod
    def timestamp(cls, value):
        return utc(value)

    @model_validator(mode="after")
    def conversation(self):
        if [x.get("role") for x in self.messages] != ["system", "user", "assistant"]:
            raise ValueError("Expected one system, user and assistant message.")
        AnalysisTarget.model_validate_json(self.messages[-1]["content"])
        return self
