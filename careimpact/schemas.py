"""Typed schemas shared by ingestion, diffing, training, inference and the API."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChangeClass(str, Enum):
    COVERAGE = "coverage"
    CLINICAL_CRITERIA = "clinical_criteria"
    DOCUMENTATION = "documentation"
    CODING_BILLING = "coding_billing"
    PROVIDER_REQUIREMENTS = "provider_requirements"
    RETIREMENT_REPLACEMENT = "retirement_replacement"
    ADMINISTRATIVE = "administrative"
    NO_SUBSTANTIVE_CHANGE = "no_substantive_change"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


SUBSTANTIVE_CLASSES = {
    ChangeClass.COVERAGE,
    ChangeClass.CLINICAL_CRITERIA,
    ChangeClass.DOCUMENTATION,
    ChangeClass.CODING_BILLING,
    ChangeClass.PROVIDER_REQUIREMENTS,
    ChangeClass.RETIREMENT_REPLACEMENT,
}

# When a comparison has several findings, the overall classification is the
# highest-priority category present.
CLASS_PRIORITY = [
    ChangeClass.RETIREMENT_REPLACEMENT,
    ChangeClass.COVERAGE,
    ChangeClass.CLINICAL_CRITERIA,
    ChangeClass.CODING_BILLING,
    ChangeClass.DOCUMENTATION,
    ChangeClass.PROVIDER_REQUIREMENTS,
    ChangeClass.ADMINISTRATIVE,
    ChangeClass.NO_SUBSTANTIVE_CHANGE,
]


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    REWORDED = "reworded"


class ImpactArea(str, Enum):
    PATIENT_ELIGIBILITY = "patient_eligibility"
    PROVIDER_WORKFLOW = "provider_workflow"
    DOCUMENTATION = "documentation"
    BILLING = "billing"
    COMPLIANCE = "compliance"
    IMPLEMENTATION_TIMELINE = "implementation_timeline"


Confidence = Literal["low", "medium", "high"]
DataOrigin = Literal["real", "synthetic", "user"]
VerificationStatus = Literal["verified", "unverified", "unavailable"]


# ---------------------------------------------------------------- documents
class Section(BaseModel):
    heading: str
    text: str


class PolicyDocument(BaseModel):
    doc_id: str
    title: str
    version: Optional[str] = None
    doc_type: str = "LCD"
    status: Optional[str] = None
    contractor: Optional[str] = None
    original_effective_date: Optional[str] = None
    revision_effective_date: Optional[str] = None
    revision_ending_date: Optional[str] = None
    retirement_date: Optional[str] = None
    source_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    data_origin: DataOrigin = "real"
    verification_status: VerificationStatus = "unverified"
    verification_notes: list[str] = Field(default_factory=list)
    related_documents: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    content_hash: Optional[str] = None


# -------------------------------------------------------------------- diffs
class DiffOp(BaseModel):
    op: Literal["equal", "insert", "delete", "replace"]
    old: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)


class SectionDiff(BaseModel):
    key: str
    heading_old: Optional[str] = None
    heading_new: Optional[str] = None
    status: Literal["unchanged", "modified", "added", "removed", "formatting_only"]
    similarity: float
    old_text: str = ""
    new_text: str = ""
    ops: list[DiffOp] = Field(default_factory=list)

    @property
    def heading(self) -> str:
        return self.heading_new or self.heading_old or self.key


class DocumentDiff(BaseModel):
    sections: list[SectionDiff]

    @property
    def changed(self) -> list[SectionDiff]:
        return [s for s in self.sections if s.status != "unchanged"]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sections:
            out[s.status] = out.get(s.status, 0) + 1
        return out


# ----------------------------------------------------------------- analysis
class Finding(BaseModel):
    section: str
    change_type: ChangeType
    category: ChangeClass
    what_changed: str
    evidence_old: Optional[str] = None
    evidence_new: Optional[str] = None
    impact_areas: list[ImpactArea] = Field(default_factory=list)
    potential_operational_impact: str
    recommended_action: str
    confidence: Confidence
    requires_verification: bool = True


class PolicyAnalysis(BaseModel):
    document_id: str
    change_classification: ChangeClass
    is_substantive: bool
    change_summary: str
    changed_sections: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    abstain: bool = False
    limitations: list[str] = Field(default_factory=list)


# ------------------------------------------------------------ training data
class ExampleRecord(BaseModel):
    """A real or synthetic comparison with its target analysis and provenance."""

    example_id: str
    data_origin: DataOrigin
    family: str  # policy family used for leakage-safe splitting
    scenario: Optional[str] = None
    old: PolicyDocument
    new: PolicyDocument
    target: PolicyAnalysis
    target_source: Literal["scenario_spec", "llm_draft", "baseline_draft", "human"]
    generator_version: Optional[str] = None
    seed: Optional[int] = None
    validation_status: Literal["auto_validated", "pending_review", "approved", "rejected", "failed_checks"]
    validation_errors: list[str] = Field(default_factory=list)
    training_eligible: bool = False
    created_at: Optional[str] = None
