"""Programmatic synthetic policy-pair generator.

Design:
* A fictional policy is held as structured state (sections -> items).
* A scenario mutates a copy of that state into the "new" version and emits
  ground-truth findings derived from the mutation itself (never from an LLM).
* Both versions are rendered to text; every evidence quote is an item string,
  so it is an exact substring of the rendered text. This is re-checked after
  rendering together with diff/label consistency checks.
"""
from __future__ import annotations

import copy
import datetime as dt
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from careimpact.analysis.prompts import build_user_message
from careimpact.diffing import diff_documents, quote_in_source
from careimpact.schemas import (
    CLASS_PRIORITY,
    SUBSTANTIVE_CLASSES,
    ChangeClass,
    ChangeType,
    ExampleRecord,
    Finding,
    ImpactArea,
    PolicyAnalysis,
    PolicyDocument,
    Section,
)
from careimpact.synthetic import topics as T

GENERATOR_VERSION = "synth-1.0"
BASE_LIMITATION = "Operational impact is inferred from the supplied text and has not been verified against official CMS guidance."
UNVERIFIED_LIMITATION = "The source versions have not been verified against official records, so the comparison may not reflect actual policy history."

PE, WF, DOC, BILL, COMP, TIME = (
    ImpactArea.PATIENT_ELIGIBILITY, ImpactArea.PROVIDER_WORKFLOW, ImpactArea.DOCUMENTATION,
    ImpactArea.BILLING, ImpactArea.COMPLIANCE, ImpactArea.IMPLEMENTATION_TIMELINE,
)

SECTION_ORDER = [
    "retirement_notice", "coverage_indications", "eligibility", "limitations", "utilization_guidelines",
    "evidence_requirements", "documentation_requirements", "provider_qualifications", "coding", "billing",
    "summary_of_evidence", "effective_dates", "sources", "contacts",
]


@dataclass
class Sec:
    key: str
    heading: str
    intro: Optional[str]
    items: list[str]
    style: str  # numbered | bullet | prose


@dataclass
class GT:
    key: str
    change_type: ChangeType
    category: ChangeClass
    what: str
    old: Optional[str]
    new: Optional[str]
    impacts: list[ImpactArea]
    impact: str
    action: str
    confidence: str
    verify: bool = True


@dataclass
class Outcome:
    findings: list[GT] = field(default_factory=list)
    abstain: bool = False
    limitations: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    reworded_ok: bool = False  # changed sections without findings are expected
    unexplained_ok: set = field(default_factory=set)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _fmt_date(d: dt.date) -> str:
    return d.strftime("%m/%d/%Y")


def _fmt_num(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# Real post-2019 LCDs group several topics under one CMS heading; the
# "consolidated" layout mirrors that so the heading alone does not reveal the
# change category.
CONSOLIDATED = {
    "Coverage Indications, Limitations, and/or Medical Necessity":
        ["coverage_indications", "eligibility", "limitations", "evidence_requirements", "provider_qualifications"],
    "Associated Information": ["documentation_requirements", "utilization_guidelines"],
}
CONSOLIDATED_LAYOUT_P = 0.5


def _section_lines(sec: Sec) -> list[str]:
    lines: list[str] = []
    if sec.intro:
        lines.append(sec.intro)
    if sec.style == "prose" or len(sec.items) == 1:
        lines.append(" ".join(sec.items))
    elif sec.style == "numbered":
        lines.extend(f"{i}. {it}" for i, it in enumerate(sec.items, 1))
    else:
        lines.extend(f"- {it}" for it in sec.items)
    return lines


def render_state(state: list[Sec], layout: str) -> tuple[list[Section], dict[str, str]]:
    """Render sections; returns (sections, key -> rendered heading)."""
    if layout != "consolidated":
        return [render_section(s) for s in state], {s.key: s.heading for s in state}
    member = {k: h for h, keys in CONSOLIDATED.items() for k in keys}
    out: list[Section] = []
    headings: dict[str, str] = {}
    done: set[str] = set()
    for sec in state:
        group = member.get(sec.key)
        if group is None:
            out.append(render_section(sec))
            headings[sec.key] = sec.heading
            continue
        if group in done:
            continue
        done.add(group)
        lines: list[str] = []
        for sub in (s for s in state if member.get(s.key) == group):
            lines.append(sub.heading)
            lines.extend(_section_lines(sub))
            headings[sub.key] = group
        out.append(Section(heading=group, text="\n".join(lines)))
    return out, headings


def render_section(sec: Sec) -> Section:
    lines: list[str] = []
    if sec.intro:
        lines.append(sec.intro)
    if sec.style == "prose" or len(sec.items) == 1:
        lines.append(" ".join(sec.items))
    elif sec.style == "numbered":
        lines.extend(f"{i}. {it}" for i, it in enumerate(sec.items, 1))
    else:
        lines.extend(f"- {it}" for it in sec.items)
    return Section(heading=sec.heading, text="\n".join(lines))


class PolicyPairGenerator:
    def __init__(self, topic: T.Topic, rng: random.Random):
        self.t = topic
        self.r = rng
        self.v = dict(
            service=topic.service, Service=_cap(topic.service), unit=topic.unit, condition=topic.condition,
            specialty=topic.specialty, test=topic.test, Test=_cap(topic.test),
            specialty_title=" ".join(w.capitalize() if w not in {"and", "of"} else w for w in topic.specialty.split()),
        )
        self.headings = {k: rng.choice(v) for k, v in T.HEADINGS.items()}
        self.cov_tpl = rng.choice(T.COVERAGE_TEMPLATES)
        self.cov_items: dict[str, str] = {}
        self.unused: dict[str, list[str]] = {}
        self.touched: set[str] = set()

    # ------------------------------------------------------------ helpers
    def f(self, s: str, **kw) -> str:
        return s.format(**{**self.v, **kw})

    def cov(self, ind: str) -> str:
        item = self.f(self.cov_tpl, ind=ind)
        self.cov_items[item] = ind
        return item

    def mk(self, key: str, items: list[str], style: Optional[str] = None) -> Sec:
        intro_opts = T.INTROS.get(key, (None,))
        intro = self.r.choice(intro_opts)
        styles = ["numbered", "bullet"] if key == "coding" else ["numbered", "bullet", "prose"]
        return Sec(key, self.headings[key], self.f(intro) if intro else None, items, style or self.r.choice(styles))

    @staticmethod
    def get(state: list[Sec], key: str) -> Optional[Sec]:
        return next((s for s in state if s.key == key), None)

    def insert(self, sec: Sec, item: str) -> None:
        sec.items.insert(self.r.randint(0, len(sec.items)), item)

    def ensure(self, key: str) -> Sec:
        """Ensure a section exists in both old and new state (created identically)."""
        if self.get(self.new, key) is None:
            sec = self.build_section(key)
            for state in (self.old, self.new):
                state.append(copy.deepcopy(sec))
                state.sort(key=lambda s: SECTION_ORDER.index(s.key))
        return self.get(self.new, key)

    def pick(self, items: list[str]) -> str:
        return self.r.choice(items)

    # --------------------------------------------------------- base policy
    def build_section(self, key: str) -> Sec:
        r, t = self.r, self.t
        if key == "coverage_indications":
            inds = r.sample(list(t.indications), r.randint(3, 4))
            self.unused["indications"] = [i for i in t.indications if i not in inds]
            return self.mk(key, [self.cov(i) for i in inds])
        if key == "eligibility":
            self.thr_val = r.choice(t.threshold.values)
            self.thr_item = self.f(t.threshold.template, v=_fmt_num(self.thr_val))
            pool = list(t.eligibility) + [self.f(g, n=r.choice([30, 60, 90])) for g in T.GENERIC_ELIGIBILITY]
            r.shuffle(pool)
            k = r.randint(2, 3)
            items = pool[:k]
            self.unused["eligibility"] = pool[k:]
            items.insert(r.randint(0, len(items)), self.thr_item)
            return self.mk(key, items)
        if key == "limitations":
            pool = list(t.limitations)
            r.shuffle(pool)
            k = r.randint(2, 3)
            self.unused["limitations"] = pool[k:]
            return self.mk(key, pool[:k])
        if key == "utilization_guidelines":
            self.dur_n, self.dur_m = r.choice(t.duration.n_values), r.choice(t.duration.m_values)
            self.dur_item = self.f(t.duration.template, n=self.dur_n, m=self.dur_m)
            return self.mk(key, [self.dur_item])
        if key == "evidence_requirements":
            self.evr_n = r.choice(T.EVIDENCE_REQ_VALUES)
            self.evr_item = self.f(T.EVIDENCE_REQ_TEMPLATE, n=self.evr_n)
            extra = [self.f(x) for x in T.EVIDENCE_REQ_EXTRA]
            r.shuffle(extra)
            items = [self.evr_item] + (extra[:1] if r.random() < 0.4 else [])
            self.unused["evidence_extra"] = [x for x in extra if x not in items]
            return self.mk(key, items)
        if key == "documentation_requirements":
            pool = [self.f(x) for x in T.DOCUMENTATION]
            r.shuffle(pool)
            k = r.randint(3, 5)
            self.unused["documentation"] = pool[k:]
            return self.mk(key, pool[:k])
        if key == "provider_qualifications":
            pool = [self.f(x) for x in T.PROVIDER_QUALS]
            r.shuffle(pool)
            k = r.randint(2, 3)
            self.unused["quals"] = pool[k:]
            items = pool[:k]
            if r.random() < 0.6:
                self.sup_level = r.choice(T.SUPERVISION_LEVELS)
                items.append(self.f(T.SUPERVISION_TEMPLATE, level=self.sup_level))
            return self.mk(key, items)
        if key == "coding":
            codes = list(t.icd)
            r.shuffle(codes)
            k = r.randint(3, min(4, len(codes)))
            self.unused["codes"] = [f"{c} - {d}" for c, d in codes[k:]]
            return self.mk(key, [f"{c} - {d}" for c, d in sorted(codes[:k])])
        if key == "billing":
            pool = [self.f(x) for x in T.BILLING]
            r.shuffle(pool)
            k = r.randint(2, 4)
            self.unused["billing"] = pool[k:]
            return self.mk(key, pool[:k])
        if key == "summary_of_evidence":
            pool = [self.f(x, outcome=t.outcome) for x in T.EVIDENCE_SUMMARY]
            r.shuffle(pool)
            return self.mk(key, pool[: r.randint(2, 3)], style="prose")
        if key == "effective_dates":
            self.app_date = dt.date(r.randint(2019, 2025), r.randint(1, 12), 1)
            self.app_tpl = r.choice(T.APPLICABILITY_TEMPLATES)
            return self.mk(key, [self.f(self.app_tpl, date=_fmt_date(self.app_date))], style="prose")
        if key == "sources":
            tpls = list(T.REFERENCE_TEMPLATES)
            r.shuffle(tpls)
            self.unused["refs"] = tpls[2:]
            items = [self.f(x, year=r.randint(2012, 2022)) for x in tpls[: r.randint(1, 2)]]
            return self.mk(key, items, style="bullet")
        if key == "contacts":
            self.contact_tpl = r.choice(T.CONTACT_TEMPLATES)
            return self.mk(key, [self.contact_item()], style="prose")
        raise KeyError(key)

    def contact_item(self) -> str:
        return self.f(self.contact_tpl, email=f"policy-unit-{self.r.randint(10, 99)}@example.org",
                      phone=f"(555) 01{self.r.randint(10, 99)}-{self.r.randint(1000, 9999)}")

    def build_base(self, required: set[str]) -> list[Sec]:
        optional = {"utilization_guidelines": 0.75, "evidence_requirements": 0.5, "summary_of_evidence": 0.5,
                    "effective_dates": 0.45, "contacts": 0.4}
        always = ["coverage_indications", "eligibility", "limitations", "documentation_requirements",
                  "provider_qualifications", "coding", "billing", "sources"]
        keys = [k for k in SECTION_ORDER if k in always or k in required or self.r.random() < optional.get(k, 0)]
        keys = [k for k in keys if k != "retirement_notice"]
        return [self.build_section(k) for k in keys]

    # ---------------------------------------------------------- scenarios
    def s_new_coverage_indication(self) -> Outcome:
        sec = self.get(self.new, "coverage_indications")
        ind = self.pick(self.unused["indications"])
        item = self.cov(ind)
        self.insert(sec, item)
        return Outcome([GT(
            "coverage_indications", ChangeType.ADDED, ChangeClass.COVERAGE,
            f"The new version adds {ind} as a covered indication for {self.t.service}.", None, item, [PE, BILL],
            self.pick([f"Beneficiaries in this group may now meet coverage requirements, which could increase eligible referrals and claims for {self.t.service}.",
                       f"Scheduling, intake and claims teams may begin seeing {self.t.service} requests for this newly listed indication."]),
            self.pick(["Update eligibility screening tools and order templates to include the new indication, and confirm supporting diagnosis codes in the related Billing and Coding Article.",
                       "Notify ordering practitioners of the added indication and verify the coding requirements before submitting claims."]),
            "high")])

    def s_removed_coverage_indication(self) -> Outcome:
        sec = self.get(self.new, "coverage_indications")
        item = self.pick(sec.items)
        sec.items.remove(item)
        ind = self.cov_items[item]
        return Outcome([GT(
            "coverage_indications", ChangeType.REMOVED, ChangeClass.COVERAGE,
            f"The new version no longer lists {ind} as a covered indication.", item, None, [PE, BILL, COMP],
            "Services for this group may no longer be supported under this policy, which could lead to denials if scheduling continues unchanged.",
            self.pick(["Review scheduled and pending services for this indication and confirm with the contractor whether coverage moved to another policy before changing practice.",
                       "Hold changes to scheduling until the removal is confirmed in the official policy, then review pending orders for this indication."]),
            "high")], limitations=["Removal from this section does not show whether the indication is addressed in another policy or article."])

    def s_expanded_eligibility(self) -> Outcome:
        sec = self.get(self.new, "eligibility")
        others = [i for i in sec.items if i != self.thr_item]
        if len(sec.items) >= 3 and self.r.random() < 0.5:
            item = self.pick(others)
            sec.items.remove(item)
            return Outcome([GT(
                "eligibility", ChangeType.REMOVED, ChangeClass.CLINICAL_CRITERIA,
                f"An eligibility criterion was removed, which broadens who may qualify for {self.t.service}.", item, None, [PE],
                "More beneficiaries may meet the criteria, so intake teams may see additional qualifying patients.",
                "Update intake checklists to remove the criterion only after confirming the change in the official policy.",
                "high")])
        alt = self.t.alt_pathway
        sec.items.insert(sec.items.index(self.thr_item) + 1, alt)
        return Outcome([GT(
            "eligibility", ChangeType.ADDED, ChangeClass.CLINICAL_CRITERIA,
            f"The new version adds an alternative qualifying pathway, which broadens eligibility for {self.t.service}.", None, alt, [PE],
            "Beneficiaries who previously did not qualify may now meet coverage criteria through the alternative pathway.",
            "Add the alternative pathway to eligibility screening and review recently declined referrals.",
            "high")])

    def s_narrowed_eligibility(self) -> Outcome:
        sec = self.get(self.new, "eligibility")
        item = self.pick(self.unused["eligibility"])
        self.insert(sec, item)
        return Outcome([GT(
            "eligibility", ChangeType.ADDED, ChangeClass.CLINICAL_CRITERIA,
            f"A new eligibility criterion was added, which narrows who qualifies for {self.t.service}.", None, item, [PE, WF],
            "Some beneficiaries who previously qualified may no longer meet the criteria, and staff will need to confirm the new criterion before services begin.",
            self.pick(["Add the new criterion to eligibility screening and order templates and review active patients for continued eligibility.",
                       "Update intake checklists with the new criterion and notify ordering practitioners."]),
            "high")])

    def s_treatment_duration(self) -> Outcome:
        if self.get(self.new, "utilization_guidelines") is None:
            self.dur_n, self.dur_m = self.r.choice(self.t.duration.n_values), self.r.choice(self.t.duration.m_values)
            self.dur_item = self.f(self.t.duration.template, n=self.dur_n, m=self.dur_m)
            self.new.append(self.mk("utilization_guidelines", [self.dur_item]))
            self.new.sort(key=lambda s: SECTION_ORDER.index(s.key))
            return Outcome([GT(
                "utilization_guidelines", ChangeType.ADDED, ChangeClass.COVERAGE,
                f"A new utilization limit was added for {self.t.service}.", None, self.dur_item, [PE, WF, BILL],
                "Treatment plans and scheduling will need to fit within the new limit, and services beyond it may be denied.",
                "Add the limit to scheduling rules and claim edits, and review beneficiaries currently in a course of treatment.",
                "high")])
        sec = self.get(self.new, "utilization_guidelines")
        d = self.t.duration
        if len(d.n_values) > 1:
            a = self.dur_n
            b = self.r.choice([x for x in d.n_values if x != a])
            new_item = self.f(d.template, n=b, m=self.dur_m)
        else:
            a = self.dur_m
            b = self.r.choice([x for x in d.m_values if x != a])
            new_item = self.f(d.template, n=self.dur_n, m=b)
        sec.items[sec.items.index(self.dur_item)] = new_item
        return Outcome([GT(
            "utilization_guidelines", ChangeType.MODIFIED, ChangeClass.COVERAGE,
            f"The utilization limit for {self.t.service} was revised: the stated value of {a} was changed to {b}.",
            self.dur_item, new_item, [PE, WF, BILL],
            "Treatment plans and scheduling may need to fit the revised limit, and services beyond it may be denied.",
            "Update scheduling rules and claim edits to reflect the revised limit, and review beneficiaries currently in a course of treatment.",
            "high")])

    def s_threshold_change(self) -> Outcome:
        th = self.t.threshold
        sec = self.get(self.new, "eligibility")
        b = self.r.choice([x for x in th.values if x != self.thr_val])
        new_item = self.f(th.template, v=_fmt_num(b))
        sec.items[sec.items.index(self.thr_item)] = new_item
        stricter = (b > self.thr_val) == (th.stricter == "higher")
        direction = "more restrictive" if stricter else "less restrictive"
        return Outcome([GT(
            "eligibility", ChangeType.MODIFIED, ChangeClass.CLINICAL_CRITERIA,
            f"The {th.measure} threshold changed from {_fmt_num(self.thr_val)} to {_fmt_num(b)}, making the criterion {direction}.",
            self.thr_item, new_item, [PE],
            ("Fewer beneficiaries may qualify, and patients near the previous threshold may no longer meet coverage criteria."
             if stricter else "Additional beneficiaries may now meet the threshold and qualify for coverage."),
            "Update eligibility screening to the new threshold and review borderline cases against the official policy.",
            "high")])

    def s_added_documentation(self) -> Outcome:
        sec = self.get(self.new, "documentation_requirements")
        item = self.pick(self.unused["documentation"])
        self.insert(sec, item)
        return Outcome([GT(
            "documentation_requirements", ChangeType.ADDED, ChangeClass.DOCUMENTATION,
            "A new documentation requirement was added.", None, item, [DOC, COMP, WF],
            "Clinical staff may need to capture additional information, and claims lacking it could be denied on medical review.",
            self.pick(["Update documentation templates and staff guidance to capture the new requirement before the policy takes effect.",
                       "Add the requirement to EHR templates and pre-billing documentation audits."]),
            "high")])

    def s_removed_documentation(self) -> Outcome:
        sec = self.get(self.new, "documentation_requirements")
        item = self.pick(sec.items)
        sec.items.remove(item)
        return Outcome([GT(
            "documentation_requirements", ChangeType.REMOVED, ChangeClass.DOCUMENTATION,
            "A documentation requirement was removed from the policy text.", item, None, [DOC, COMP],
            "Documentation workflows may not need to change, because the requirement may still apply through other rules.",
            "Keep the current documentation practice until the change is confirmed with the contractor or the related Billing and Coding Article.",
            "low")], limitations=["Removal of wording does not confirm that the documentation requirement no longer applies; it may still be required by general Medicare documentation rules or a related article."])

    def s_diagnosis_codes(self) -> Outcome:
        sec = self.get(self.new, "coding")
        if self.unused.get("codes") and (len(sec.items) < 3 or self.r.random() < 0.55):
            item = self.pick(self.unused["codes"])
            sec.items.append(item)
            sec.items.sort()
            code, desc = item.split(" - ", 1)
            return Outcome([GT(
                "coding", ChangeType.ADDED, ChangeClass.CODING_BILLING,
                f"Diagnosis code {code} ({desc}) was added to the codes that support medical necessity.", None, item, [BILL, PE],
                "Claims reporting this diagnosis may now be supported under the policy.",
                "Update claim edits and coding references, and confirm the code list against the related Billing and Coding Article.",
                "high")])
        item = self.pick(sec.items)
        sec.items.remove(item)
        code, desc = item.split(" - ", 1)
        return Outcome([GT(
            "coding", ChangeType.REMOVED, ChangeClass.CODING_BILLING,
            f"Diagnosis code {code} ({desc}) was removed from the codes that support medical necessity.", item, None, [BILL, PE],
            "Claims reporting this diagnosis may be denied if billed under this policy.",
            "Review coding edits and pending claims that use this code, and confirm whether another code applies.",
            "high")])

    def s_billing_instructions(self) -> Outcome:
        sec = self.get(self.new, "billing")
        mods = {self.f(k): self.f(v) for k, v in T.BILLING_MODIFICATIONS.items()}
        candidates = [i for i in sec.items if i in mods]
        if candidates and self.r.random() < 0.6:
            item = self.pick(candidates)
            sec.items[sec.items.index(item)] = mods[item]
            return Outcome([GT(
                "billing", ChangeType.MODIFIED, ChangeClass.CODING_BILLING,
                "A billing instruction was revised.", item, mods[item], [BILL],
                f"Billing staff may need to change how claims for {self.t.service} are reported.",
                "Update billing guidance and claim scrubber rules, and verify the instruction against the related Billing and Coding Article.",
                "medium")])
        item = self.pick(self.unused["billing"])
        self.insert(sec, item)
        return Outcome([GT(
            "billing", ChangeType.ADDED, ChangeClass.CODING_BILLING,
            "A new billing instruction was added.", None, item, [BILL],
            f"Billing staff may need to change how claims for {self.t.service} are reported.",
            "Update billing guidance and claim scrubber rules before submitting affected claims.",
            "high")])

    def s_provider_qualifications(self) -> Outcome:
        sec = self.get(self.new, "provider_qualifications")
        mods = {self.f(k): self.f(v) for k, v in T.QUAL_MODIFICATIONS.items()}
        candidates = [i for i in sec.items if i in mods]
        if candidates and self.r.random() < 0.5:
            item = self.pick(candidates)
            sec.items[sec.items.index(item)] = mods[item]
            return Outcome([GT(
                "provider_qualifications", ChangeType.MODIFIED, ChangeClass.PROVIDER_REQUIREMENTS,
                "A provider qualification requirement was revised.", item, mods[item], [WF, COMP],
                f"Some practitioners or staff may no longer meet the requirements to order or furnish {self.t.service}.",
                f"Review credentialing records for practitioners who furnish {self.t.service} and update credentialing checklists.",
                "high")])
        item = self.pick(self.unused["quals"])
        self.insert(sec, item)
        return Outcome([GT(
            "provider_qualifications", ChangeType.ADDED, ChangeClass.PROVIDER_REQUIREMENTS,
            "A new provider qualification requirement was added.", None, item, [WF, COMP],
            f"Some practitioners or staff may no longer meet the requirements to order or furnish {self.t.service}.",
            f"Review credentialing records for practitioners who furnish {self.t.service} and update credentialing checklists.",
            "high")])

    def s_supervision(self) -> Outcome:
        sec = self.get(self.new, "provider_qualifications")
        if not hasattr(self, "sup_level"):
            self.sup_level = self.r.choice(T.SUPERVISION_LEVELS)
            item = self.f(T.SUPERVISION_TEMPLATE, level=self.sup_level)
            self.get(self.old, "provider_qualifications").items.append(item)
            sec.items.append(item)
        old_item = self.f(T.SUPERVISION_TEMPLATE, level=self.sup_level)
        b = self.r.choice([x for x in T.SUPERVISION_LEVELS if x != self.sup_level])
        new_item = self.f(T.SUPERVISION_TEMPLATE, level=b)
        sec.items[sec.items.index(old_item)] = new_item
        stricter = T.SUPERVISION_LEVELS.index(b) > T.SUPERVISION_LEVELS.index(self.sup_level)
        return Outcome([GT(
            "provider_qualifications", ChangeType.MODIFIED, ChangeClass.PROVIDER_REQUIREMENTS,
            f"The required supervision level changed from {self.sup_level} to {b} supervision.", old_item, new_item, [WF, COMP],
            ("A physician may need to be immediately available or present more often, which affects staffing and scheduling."
             if stricter else "Supervision requirements are less stringent, which may allow more flexible staffing."),
            "Confirm the supervision level with the contractor and update staffing schedules and compliance checklists.",
            "high")])

    def s_new_limitation(self) -> Outcome:
        sec = self.get(self.new, "limitations")
        item = self.pick(self.unused["limitations"])
        self.insert(sec, item)
        return Outcome([GT(
            "limitations", ChangeType.ADDED, ChangeClass.COVERAGE,
            "A new limitation or contraindication was added.", None, item, [PE, BILL],
            "Services matching the new limitation may be denied, so ordering practitioners may need to screen for it.",
            "Add the limitation to pre-service screening and notify ordering practitioners.",
            "high")])

    def s_effective_date(self) -> Outcome:
        self.ensure("effective_dates")
        sec = self.get(self.new, "effective_dates")
        old_item = sec.items[0]
        delta = self.r.choice([-1, 1]) * self.r.choice([1, 2, 3, 6])
        m = self.app_date.month - 1 + delta
        b = dt.date(self.app_date.year + m // 12, m % 12 + 1, 1)
        new_item = self.f(self.app_tpl, date=_fmt_date(b))
        sec.items[0] = new_item
        return Outcome([GT(
            "effective_dates", ChangeType.MODIFIED, ChangeClass.ADMINISTRATIVE,
            f"The date from which the policy requirements apply changed from {_fmt_date(self.app_date)} to {_fmt_date(b)}.",
            old_item, new_item, [TIME],
            "Implementation timelines and claim edits tied to this policy may need to shift.",
            "Align implementation plans and claim system effective dates with the revised date.",
            "high")])

    def s_retirement(self) -> Outcome:
        date = _fmt_date(dt.date(self.r.randint(2024, 2026), self.r.randint(1, 12), 1))
        if self.r.random() < 0.7:
            rep = f"SYN-L9{self.r.randint(1000, 9999)}"
            items = [f"This policy is retired effective {date}.",
                     f"Services previously addressed by this policy are addressed by {rep}, {self.t.title} (Consolidated)."]
            what = f"The policy is retired and replaced by {rep}."
            impact = f"Coverage rules for {self.t.service} will come from the replacement policy, which may have different requirements."
            action = "Obtain and compare the replacement policy before the retirement date, and update references in workflows and claim edits."
            lim = "The replacement policy text was not supplied, so its requirements could not be compared."
        else:
            items = [f"This policy is retired effective {date} and is not being replaced."]
            what = "The policy is retired without a replacement policy."
            impact = f"There may no longer be a local policy describing coverage criteria for {self.t.service}."
            action = "Determine how claims will be reviewed after retirement, such as under national policy or individual review, before the retirement date."
            lim = "The supplied text does not state how coverage will be determined after retirement."
        self.new.insert(0, Sec("retirement_notice", self.headings["retirement_notice"], None, items, "prose"))
        self.new_status, self.retire_date = "Retired", date
        return Outcome([GT(
            "retirement_notice", ChangeType.ADDED, ChangeClass.RETIREMENT_REPLACEMENT,
            what, None, items[0], [COMP, BILL, WF], impact, action, "high")], limitations=[lim])

    def _admin(self, kind: str) -> Optional[GT]:
        """Administrative edit; mutates old/new. Returns None if not applicable."""
        if kind == "typo":
            cands = []
            for s_old, s_new in ((self.get(self.old, k), self.get(self.new, k)) for k in
                                 ["eligibility", "limitations", "documentation_requirements", "provider_qualifications", "billing"]):
                if s_old is None or s_new is None or s_old.key in self.touched:
                    continue
                for idx, it in enumerate(s_new.items):
                    if s_old.items == s_new.items:
                        for w in T.TYPO_WORDS:
                            if re.search(rf"\b{w}\b", it):
                                cands.append((s_old, s_new, idx, w))
            if not cands:
                return None
            s_old, s_new, idx, w = self.pick(cands)
            good = s_new.items[idx]
            bad = re.sub(rf"\b{w}\b", T.TYPO_WORDS[w], good, count=1)
            s_old.items[idx] = bad
            self.touched.add(s_old.key)
            return GT(s_old.key, ChangeType.MODIFIED, ChangeClass.ADMINISTRATIVE,
                      "A spelling error was corrected without changing the meaning.", bad, good, [],
                      "No operational change is expected.", "No workflow change is needed; confirm that no other text in the section changed.",
                      "high", verify=False)
        if kind == "reference":
            sec = self.get(self.new, "sources")
            if sec is None or "sources" in self.touched or not self.unused.get("refs"):
                return None
            item = self.f(self.pick(self.unused["refs"]), year=self.r.randint(2023, 2026))
            sec.items.append(item)
            self.touched.add("sources")
            return GT("sources", ChangeType.ADDED, ChangeClass.ADMINISTRATIVE,
                      "A reference was added to the sources list.", None, item, [],
                      "No change to coverage criteria is indicated by this update.", "No workflow change is needed.",
                      "high", verify=False)
        if kind == "contact":
            if "contacts" in self.touched:
                return None
            sec = self.ensure("contacts")
            old_item = sec.items[0]
            new_item = self.contact_item()
            if new_item == old_item:
                return None
            sec.items[0] = new_item
            self.touched.add("contacts")
            return GT("contacts", ChangeType.MODIFIED, ChangeClass.ADMINISTRATIVE,
                      "Contractor contact information was updated.", old_item, new_item, [],
                      "Staff who contact the contractor about this policy should use the updated contact details.",
                      "Update internal contact references for this policy.", "high", verify=False)
        raise KeyError(kind)

    def s_administrative(self) -> Outcome:
        for kind in self.r.sample(["typo", "reference", "contact"], 3):
            gt = self._admin(kind)
            if gt:
                return Outcome([gt])
        raise RuntimeError("no administrative edit applicable")

    def s_no_substantive_change(self) -> Outcome:
        keys = [s.key for s in self.new if s.key in {"eligibility", "limitations", "documentation_requirements",
                                                      "provider_qualifications", "coverage_indications", "billing"}]
        for key in self.r.sample(keys, self.r.randint(1, min(3, len(keys)))):
            sec = self.get(self.new, key)
            ops = self.r.sample(["reword", "style", "heading", "reorder"], self.r.randint(1, 2))
            if "reword" in ops:
                rules = [(p, s) for p, s in T.REWORDINGS if any(re.search(p, it) for it in sec.items)]
                for p, s in self.r.sample(rules, min(len(rules), self.r.randint(1, 3))):
                    sec.items = [re.sub(p, s, it) for it in sec.items]
            if "style" in ops:
                sec.style = self.r.choice([x for x in ("numbered", "bullet", "prose") if x != sec.style])
            if "heading" in ops:
                opts = [h for h in T.HEADINGS[key] if h != sec.heading]
                if opts:
                    sec.heading = self.r.choice(opts)
            if "reorder" in ops and len(sec.items) > 2 and key != "coverage_indications":
                self.r.shuffle(sec.items)
        if [_section_lines(s) + [s.heading] for s in self.old] == [_section_lines(s) + [s.heading] for s in self.new]:
            multi = [self.get(self.new, k) for k in keys if len(self.get(self.new, k).items) > 1]
            if multi:
                multi[0].style = "numbered" if multi[0].style != "numbered" else "bullet"
            else:
                sec = self.get(self.new, keys[0])
                sec.heading = next(h for h in T.HEADINGS[sec.key] if h != sec.heading)
        return Outcome([], reworded_ok=True,
                       summary="The versions differ only in wording or formatting; no substantive change was identified in the supplied text.",
                       limitations=["Equivalence of the reworded text was assessed from the supplied text only."])

    def s_ambiguous(self) -> Outcome:
        variant = self.r.choice(["truncated", "conflict", "vague"])
        unit_tpl_ok = hasattr(self, "dur_item") and f"{{n}} {self.t.unit}" in self.t.duration.template
        if variant == "conflict" and not unit_tpl_ok:
            variant = self.r.choice(["truncated", "vague"])
        if variant == "truncated":
            key = self.pick([s.key for s in self.new if s.key in {"eligibility", "limitations", "documentation_requirements", "coverage_indications"}])
            sec = self.get(self.new, key)
            keep = sec.items[: max(0, len(sec.items) // 2 - 1)]
            sec.items = keep + ["Text for the remainder of this section could not be retrieved from the source document."]
            return Outcome([], abstain=True, unexplained_ok={key},
                           summary=f"The new version of the {sec.heading} section is incomplete, so changes to it cannot be determined.",
                           limitations=[f"The {sec.heading} section text in the new version is incomplete; compare against the official source before drawing conclusions."])
        if variant == "conflict":
            n3 = self.r.choice([x for x in range(max(2, self.dur_n - 12), self.dur_n + 13) if x != self.dur_n])
            item = f"{self.v['Service']} is limited to {n3} {self.t.unit}s under this policy."
            self.insert(self.get(self.new, "limitations"), item)
            return Outcome([GT(
                "limitations", ChangeType.ADDED, ChangeClass.COVERAGE,
                "A new limit was added that conflicts with the utilization limit stated elsewhere in the policy.", None, item, [PE, BILL],
                "The applicable limit cannot be determined from the supplied text.",
                "Do not change scheduling limits until the contractor clarifies which limit applies.",
                "low")], abstain=True,
                summary="The new version adds a limit that conflicts with another limit in the same policy, so the applicable requirement is unclear.",
                limitations=[f"The new version states conflicting limits ({self.dur_n} versus {n3}), so the applicable requirement cannot be determined."])
        sec = self.get(self.new, "documentation_requirements")
        item = self.pick(sec.items)
        vague = "Additional documentation may be requested as determined by the contractor."
        sec.items[sec.items.index(item)] = vague
        return Outcome([GT(
            "documentation_requirements", ChangeType.MODIFIED, ChangeClass.DOCUMENTATION,
            "A specific documentation requirement was replaced with general language.", item, vague, [DOC, COMP],
            "It is unclear whether the specific requirement still applies.",
            "Continue the existing documentation practice and request clarification from the contractor.",
            "low")], limitations=["The new wording is vague, so the actual documentation requirement cannot be determined from the supplied text."])

    def s_evidence_requirements(self) -> Outcome:
        if self.r.random() < 0.65:
            sec = self.ensure("evidence_requirements")
            if self.unused.get("evidence_extra") and self.r.random() < 0.4:
                item = self.pick(self.unused["evidence_extra"])
                sec.items.append(item)
                return Outcome([GT(
                    "evidence_requirements", ChangeType.ADDED, ChangeClass.CLINICAL_CRITERIA,
                    "A new clinical evidence requirement was added before coverage.", None, item, [PE, WF, DOC],
                    "Beneficiaries may need additional testing or assessment before services can begin.",
                    "Add the required evidence to pre-service checklists and confirm it is captured in the record.",
                    "high")])
            b = self.r.choice([x for x in T.EVIDENCE_REQ_VALUES if x != self.evr_n])
            new_item = self.f(T.EVIDENCE_REQ_TEMPLATE, n=b)
            sec.items[sec.items.index(self.evr_item)] = new_item
            return Outcome([GT(
                "evidence_requirements", ChangeType.MODIFIED, ChangeClass.CLINICAL_CRITERIA,
                f"The time window for required {self.t.test} results changed from {self.evr_n} to {b} months.",
                self.evr_item, new_item, [PE, WF],
                ("Beneficiaries may need repeat testing if prior results fall outside the shorter window."
                 if b < self.evr_n else "Older test results may now be accepted, which may reduce repeat testing."),
                "Update pre-service checklists to confirm test results fall within the revised window.",
                "high")])
        sec = self.ensure("summary_of_evidence")
        item = self.f(self.pick(list(T.EVIDENCE_SUMMARY_UPDATES)), year=self.r.randint(2023, 2026))
        sec.items.append(item)
        return Outcome([GT(
            "summary_of_evidence", ChangeType.ADDED, ChangeClass.ADMINISTRATIVE,
            "The summary of evidence was updated with additional literature.", None, item, [],
            "The evidence update does not by itself change coverage criteria.",
            "No workflow change is indicated; monitor for future revisions to coverage criteria.",
            "medium", verify=False)],
            limitations=["An updated evidence summary may precede future criteria changes, but none are stated in the supplied text."])

    def s_billing_article_reference(self) -> Outcome:
        sec = self.get(self.new, "coding")
        aid = f"SYN-A5{self.r.randint(1000, 9999)}"
        old_item = sec.items[0]
        new_item = f"Refer to the related Billing and Coding Article {aid} for the ICD-10-CM codes that support medical necessity."
        sec.items, sec.intro = [new_item], None
        return Outcome([GT(
            "coding", ChangeType.MODIFIED, ChangeClass.CODING_BILLING,
            "The diagnosis code list was removed from the policy and replaced with a reference to a related Billing and Coding Article.",
            old_item, new_item, [BILL],
            "Coding staff will need the referenced article to determine which diagnosis codes are supported.",
            f"Obtain Billing and Coding Article {aid} and compare its code list with the previous list before updating claim edits.",
            "medium")], limitations=["The related Billing and Coding Article was not supplied, so it is unknown whether the supported codes changed."])


@dataclass(frozen=True)
class ScenarioSpec:
    method: str
    requires: frozenset = frozenset()
    allow_noise: bool = True


SCENARIOS: dict[str, ScenarioSpec] = {
    "new_coverage_indication": ScenarioSpec("s_new_coverage_indication"),
    "removed_coverage_indication": ScenarioSpec("s_removed_coverage_indication"),
    "expanded_eligibility": ScenarioSpec("s_expanded_eligibility"),
    "narrowed_eligibility": ScenarioSpec("s_narrowed_eligibility"),
    "treatment_duration": ScenarioSpec("s_treatment_duration"),
    "threshold_change": ScenarioSpec("s_threshold_change"),
    "added_documentation": ScenarioSpec("s_added_documentation"),
    "removed_documentation": ScenarioSpec("s_removed_documentation"),
    "diagnosis_codes": ScenarioSpec("s_diagnosis_codes"),
    "billing_instructions": ScenarioSpec("s_billing_instructions"),
    "provider_qualifications": ScenarioSpec("s_provider_qualifications"),
    "supervision_requirements": ScenarioSpec("s_supervision"),
    "new_limitation": ScenarioSpec("s_new_limitation"),
    "effective_date_revision": ScenarioSpec("s_effective_date", frozenset({"effective_dates"}), allow_noise=False),
    "retirement_replacement": ScenarioSpec("s_retirement"),
    "administrative_edit": ScenarioSpec("s_administrative", allow_noise=False),
    "no_substantive_change": ScenarioSpec("s_no_substantive_change", allow_noise=False),
    "ambiguous_evidence": ScenarioSpec("s_ambiguous", allow_noise=False),
    "evidence_requirements": ScenarioSpec("s_evidence_requirements"),
    "billing_article_reference": ScenarioSpec("s_billing_article_reference"),
}

COMPOUNDABLE = ["new_coverage_indication", "narrowed_eligibility", "added_documentation", "diagnosis_codes",
                "billing_instructions", "provider_qualifications", "new_limitation", "threshold_change"]


def _finding(gt: GT, heading: str) -> Finding:
    return Finding(section=heading, change_type=gt.change_type, category=gt.category, what_changed=gt.what,
                   evidence_old=gt.old, evidence_new=gt.new, impact_areas=gt.impacts,
                   potential_operational_impact=gt.impact, recommended_action=gt.action,
                   confidence=gt.confidence, requires_verification=gt.verify)


def generate_example(scenario: str, family: str, seed: int, example_id: str,
                     noise_p: float = 0.25, compound_p: float = 0.12, unverified_p: float = 0.12) -> ExampleRecord:
    rng = random.Random(seed)
    topic = T.TOPICS_BY_KEY[family]
    spec = SCENARIOS[scenario]
    g = PolicyPairGenerator(topic, rng)
    g.old = g.build_base(set(spec.requires))
    g.new = copy.deepcopy(g.old)
    g.new_status, g.retire_date = "Currently in Effect", None
    layout = "consolidated" if rng.random() < CONSOLIDATED_LAYOUT_P else "separate"

    out: Outcome = getattr(g, spec.method)()
    g.touched |= {f.key for f in out.findings}
    if spec.allow_noise and rng.random() < compound_p:
        other = rng.choice([s for s in COMPOUNDABLE if s != scenario])
        probe = copy.deepcopy((g.old, g.new, g.unused, g.cov_items))
        extra = getattr(g, SCENARIOS[other].method)()
        if any(f.key in g.touched for f in extra.findings):
            g.old, g.new, g.unused, g.cov_items = probe  # roll back overlapping compound edit
        else:
            out.findings += extra.findings
            out.limitations += extra.limitations
            g.touched |= {f.key for f in extra.findings}
    if spec.allow_noise and rng.random() < noise_p:
        gt = g._admin(rng.choice(["typo", "reference", "contact"]))
        if gt:
            out.findings.append(gt)

    # -------------------------------------------------------------- render
    pid = f"SYN-L9{rng.randint(1000, 9999)}"
    v_old = rng.randint(1, 12)
    d_old = dt.date(rng.randint(2019, 2024), rng.randint(1, 12), 1)
    d_new = d_old + dt.timedelta(days=rng.randint(60, 540))
    verification = "unverified" if rng.random() < unverified_p else "verified"
    common = dict(doc_id=pid, title=f"{topic.title} (Synthetic)", contractor=f"Fictional MAC Jurisdiction {rng.choice('ABCDEFGHJK')}",
                  data_origin="synthetic", verification_status=verification,
                  verification_notes=["Synthetic fictional policy generated from a scenario specification; not CMS guidance."])
    old_sections, headings_old = render_state(g.old, layout)
    new_sections, headings_new = render_state(g.new, layout)
    old_doc = PolicyDocument(version=str(v_old), status="Superseded", revision_effective_date=_fmt_date(d_old),
                             sections=old_sections, **common)
    new_doc = PolicyDocument(version=str(v_old + 1), status=g.new_status, revision_effective_date=_fmt_date(d_new),
                             retirement_date=g.retire_date, sections=new_sections, **common)
    diff = diff_documents(old_doc, new_doc)

    findings = [_finding(gt, headings_new.get(gt.key) or headings_old[gt.key]) for gt in out.findings]
    findings.sort(key=lambda f: CLASS_PRIORITY.index(f.category))

    if out.abstain:
        cls = ChangeClass.INSUFFICIENT_EVIDENCE
    elif findings:
        cls = findings[0].category
    else:
        cls = ChangeClass.NO_SUBSTANTIVE_CHANGE
    summary = out.summary or " ".join(f.what_changed for f in findings[:3])
    limitations = [BASE_LIMITATION] + out.limitations
    if verification == "unverified":
        limitations.append(UNVERIFIED_LIMITATION)
    target = PolicyAnalysis(
        document_id=pid, change_classification=cls, is_substantive=cls in SUBSTANTIVE_CLASSES,
        change_summary=summary, changed_sections=[s.heading for s in diff.changed], findings=findings,
        abstain=out.abstain, limitations=limitations,
    )

    # ------------------------------------------------------ consistency checks
    errors: list[str] = []
    old_text = "\n".join(s.text for s in old_doc.sections)
    new_text = "\n".join(s.text for s in new_doc.sections)
    for f in findings:
        if f.evidence_old and not quote_in_source(f.evidence_old, old_text):
            errors.append(f"evidence_old not in old text: {f.evidence_old[:60]}")
        if f.evidence_new and not quote_in_source(f.evidence_new, new_text):
            errors.append(f"evidence_new not in new text: {f.evidence_new[:60]}")
        if f.evidence_old is None and f.evidence_new is None:
            errors.append("finding without evidence")
    changed_headings = set(target.changed_sections)
    for f in findings:
        if f.section not in changed_headings:
            errors.append(f"finding section '{f.section}' not detected as changed by the diff")
    explained = {f.section for f in findings} | {headings_new.get(k, k) for k in out.unexplained_ok}
    if not out.reworded_ok:
        for h in changed_headings - explained:
            errors.append(f"changed section '{h}' has no ground-truth finding")
    if not diff.changed:
        errors.append("no textual difference between versions")

    return ExampleRecord(
        example_id=example_id, data_origin="synthetic", family=family, scenario=scenario,
        old=old_doc, new=new_doc, target=target, target_source="scenario_spec",
        generator_version=GENERATOR_VERSION, seed=seed,
        validation_status="failed_checks" if errors else "auto_validated",
        validation_errors=errors, training_eligible=not errors,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


def prompt_for(rec: ExampleRecord) -> str:
    return build_user_message(rec.old, rec.new, diff_documents(rec.old, rec.new))


ScenarioFn = Callable[[PolicyPairGenerator], Outcome]
