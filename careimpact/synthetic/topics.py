"""Fictional policy families used by the synthetic generator.

Every policy generated from these families is FICTIONAL. Clinical topics and
ICD-10-CM codes are used for realism only; thresholds, durations and rules are
invented and must never be read as actual CMS or MAC requirements.

Each family is a unit for leakage-safe splitting: families listed in
SPLIT_FAMILIES["test"] never appear in training data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Threshold:
    template: str  # contains {v}
    values: tuple
    measure: str
    stricter: str  # "higher" or "lower": which direction narrows eligibility


@dataclass(frozen=True)
class Duration:
    template: str  # contains {n} and {m}
    n_values: tuple
    m_values: tuple


@dataclass(frozen=True)
class Topic:
    key: str
    service: str
    title: str
    unit: str
    condition: str
    specialty: str
    test: str
    indications: tuple
    eligibility: tuple
    threshold: Threshold
    limitations: tuple
    duration: Duration
    icd: tuple  # (code, description)
    alt_pathway: str
    outcome: str
    extra: dict = field(default_factory=dict)


TOPICS: list[Topic] = [
    Topic(
        key="pulmonary_rehab", service="pulmonary rehabilitation", title="Outpatient Pulmonary Rehabilitation",
        unit="session", condition="chronic obstructive pulmonary disease (COPD)", specialty="pulmonary disease",
        test="pulmonary function testing",
        indications=(
            "beneficiaries with moderate to very severe COPD",
            "beneficiaries recovering from lung volume reduction surgery",
            "beneficiaries being evaluated for lung transplantation",
            "beneficiaries with interstitial lung disease and exertional dyspnea",
            "beneficiaries with bronchiectasis and persistent exercise limitation",
            "beneficiaries with clinically stable pulmonary hypertension",
        ),
        eligibility=(
            "The beneficiary has persistent dyspnea despite optimal medical therapy.",
            "The beneficiary does not currently smoke or is enrolled in a smoking cessation program.",
            "The beneficiary has not completed a pulmonary rehabilitation program within the past 12 months.",
        ),
        threshold=Threshold("The beneficiary has a post-bronchodilator FEV1/FVC ratio of less than {v}.",
                            (0.60, 0.65, 0.70), "post-bronchodilator FEV1/FVC ratio", "lower"),
        limitations=(
            "Maintenance programs furnished after the beneficiary has reached a plateau are not covered.",
            "Sessions consisting only of education without exercise are not covered.",
            "Pulmonary rehabilitation furnished in the home is not covered under this policy.",
            "Pulmonary rehabilitation is not covered for beneficiaries with unstable angina.",
            "Pulmonary rehabilitation is not covered during an acute exacerbation requiring hospitalization.",
        ),
        duration=Duration("Pulmonary rehabilitation is limited to {n} sessions within a {m}-week period.", (24, 30, 36, 48), (12, 18, 24, 36)),
        icd=(("J44.0", "Chronic obstructive pulmonary disease with acute lower respiratory infection"),
             ("J44.1", "Chronic obstructive pulmonary disease with (acute) exacerbation"),
             ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
             ("J43.9", "Emphysema, unspecified"),
             ("J84.10", "Pulmonary fibrosis, unspecified"),
             ("J47.9", "Bronchiectasis, uncomplicated")),
        alt_pathway="Beneficiaries who do not meet the spirometry criterion may also qualify if exertional oxygen desaturation is documented.",
        outcome="exercise capacity and dyspnea scores",
    ),
    Topic(
        key="cardiac_rehab", service="cardiac rehabilitation", title="Cardiac Rehabilitation Programs",
        unit="session", condition="cardiovascular disease", specialty="cardiovascular disease",
        test="exercise stress testing",
        indications=(
            "beneficiaries who had an acute myocardial infarction within the preceding 12 months",
            "beneficiaries who had coronary artery bypass surgery",
            "beneficiaries with current stable angina pectoris",
            "beneficiaries who had heart valve repair or replacement",
            "beneficiaries who had percutaneous coronary intervention",
            "beneficiaries with stable chronic heart failure",
        ),
        eligibility=(
            "The beneficiary has been clinically stable for at least six weeks.",
            "The beneficiary has had no hospitalization for heart failure in the past six weeks.",
            "The beneficiary has a written exercise prescription from the supervising physician.",
        ),
        threshold=Threshold("The beneficiary has a left ventricular ejection fraction of {v} percent or less.",
                            (35, 40, 45), "left ventricular ejection fraction", "lower"),
        limitations=(
            "Programs that do not include individualized exercise are not covered.",
            "Cardiac rehabilitation is not covered for beneficiaries with uncontrolled arrhythmias.",
            "Repeat programs for the same qualifying event are not covered.",
            "Cardiac rehabilitation is not covered for beneficiaries with severe symptomatic aortic stenosis.",
            "Sessions furnished by telephone only are not covered.",
        ),
        duration=Duration("Cardiac rehabilitation is limited to {n} sessions within a {m}-week period.", (24, 36, 48, 72), (18, 24, 36)),
        icd=(("I21.4", "Non-ST elevation (NSTEMI) myocardial infarction"),
             ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
             ("I20.9", "Angina pectoris, unspecified"),
             ("I50.22", "Chronic systolic (congestive) heart failure"),
             ("Z95.1", "Presence of aortocoronary bypass graft"),
             ("Z95.5", "Presence of coronary angioplasty implant and graft")),
        alt_pathway="Beneficiaries who do not meet the ejection fraction criterion may also qualify after a qualifying cardiac procedure within the past 12 months.",
        outcome="functional capacity and readmission rates",
    ),
    Topic(
        key="sleep_apnea_pap", service="positive airway pressure therapy", title="Positive Airway Pressure Devices for Obstructive Sleep Apnea",
        unit="month of rental", condition="obstructive sleep apnea", specialty="sleep medicine",
        test="a sleep study",
        indications=(
            "beneficiaries with obstructive sleep apnea confirmed by a facility-based sleep study",
            "beneficiaries with obstructive sleep apnea confirmed by a home sleep apnea test",
            "beneficiaries with central sleep apnea related to heart failure",
            "beneficiaries with obesity hypoventilation syndrome",
            "beneficiaries with complex sleep apnea identified during titration",
        ),
        eligibility=(
            "The beneficiary has had a face-to-face clinical evaluation before the sleep test.",
            "The beneficiary and caregiver have received instruction in the proper use of the device.",
            "The beneficiary reports excessive daytime sleepiness or impaired cognition.",
        ),
        threshold=Threshold("The beneficiary has an apnea-hypopnea index of at least {v} events per hour.",
                            (5, 10, 15, 20), "apnea-hypopnea index", "higher"),
        limitations=(
            "Devices used solely for snoring without documented sleep apnea are not covered.",
            "Replacement of a device within its reasonable useful lifetime is not covered unless the device is lost or irreparably damaged.",
            "Positive airway pressure therapy is not covered for beneficiaries with untreated severe bullous lung disease.",
            "Backup devices are not covered.",
            "Positive airway pressure therapy is not covered when the sleep test was interpreted by the supplier.",
        ),
        duration=Duration("Continued coverage after the first {n} days requires documented adherence of at least four hours per night on {m} percent of nights.", (60, 90, 120), (70, 75, 80)),
        icd=(("G47.33", "Obstructive sleep apnea (adult) (pediatric)"),
             ("G47.30", "Sleep apnea, unspecified"),
             ("G47.31", "Primary central sleep apnea"),
             ("E66.2", "Morbid (severe) obesity with alveolar hypoventilation"),
             ("G47.10", "Hypersomnia, unspecified")),
        alt_pathway="Beneficiaries who do not meet the index criterion may also qualify if they have documented hypertension and an index of at least 5 events per hour.",
        outcome="daytime sleepiness and blood pressure",
    ),
    Topic(
        key="home_oxygen", service="home oxygen therapy", title="Home Oxygen Therapy",
        unit="month of rental", condition="chronic hypoxemia", specialty="pulmonary disease",
        test="arterial blood gas or oximetry testing",
        indications=(
            "beneficiaries with chronic hypoxemia at rest",
            "beneficiaries with hypoxemia during exercise that improves with oxygen",
            "beneficiaries with nocturnal hypoxemia",
            "beneficiaries with pulmonary hypertension and hypoxemia",
            "beneficiaries with erythrocytosis related to chronic hypoxemia",
        ),
        eligibility=(
            "Qualifying testing was performed while the beneficiary was in a chronic stable state.",
            "Alternative treatment measures have been tried or considered and deemed clinically ineffective.",
            "The beneficiary has been seen by the treating practitioner within 30 days before the initial order.",
        ),
        threshold=Threshold("The beneficiary has a resting oxygen saturation of {v} percent or less on room air.",
                            (86, 88, 89), "resting oxygen saturation", "lower"),
        limitations=(
            "Oxygen used only as needed for dyspnea without documented hypoxemia is not covered.",
            "Portable oxygen is not covered when used only during sleep.",
            "Oxygen therapy is not covered for beneficiaries with angina pectoris in the absence of hypoxemia.",
            "Oxygen is not covered for terminal illnesses that do not affect the respiratory system.",
            "Spare tanks for emergency use only are not covered.",
        ),
        duration=Duration("Initial coverage is limited to {n} days, after which retesting within the last {m} days of the initial period is required.", (60, 90, 120), (30, 45)),
        icd=(("J96.11", "Chronic respiratory failure with hypoxia"),
             ("J96.12", "Chronic respiratory failure with hypercapnia"),
             ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
             ("I27.20", "Pulmonary hypertension, unspecified"),
             ("R09.02", "Hypoxemia")),
        alt_pathway="Beneficiaries who do not meet the resting criterion may also qualify if oxygen saturation falls to the qualifying level during exercise.",
        outcome="survival and exercise tolerance",
    ),
    Topic(
        key="epidural_steroid", service="epidural steroid injections", title="Epidural Steroid Injections for Pain Management",
        unit="injection", condition="radicular pain", specialty="pain medicine",
        test="spinal imaging",
        indications=(
            "beneficiaries with lumbar radicular pain due to disc herniation",
            "beneficiaries with cervical radicular pain due to disc herniation",
            "beneficiaries with lumbar spinal stenosis and neurogenic claudication",
            "beneficiaries with radicular pain following spinal surgery",
            "beneficiaries with thoracic radicular pain due to herpes zoster",
        ),
        eligibility=(
            "The pain has persisted for at least four weeks despite conservative care.",
            "The pain causes functional limitation documented with a validated scale.",
            "Imaging findings are consistent with the clinical presentation.",
        ),
        threshold=Threshold("The beneficiary has radicular pain rated at least {v} on a 10-point scale.",
                            (4, 5, 6), "pain severity rating", "higher"),
        limitations=(
            "Injections performed without imaging guidance are not covered.",
            "Epidural steroid injections for axial back pain without radicular symptoms are not covered.",
            "Injections at more than two spinal levels during one session are not covered.",
            "Epidural steroid injections are not covered for beneficiaries with an active systemic infection.",
            "Injections performed under moderate or deep sedation without a documented indication are not covered.",
        ),
        duration=Duration("No more than {n} injection sessions are covered in a rolling {m}-month period.", (3, 4, 6), (6, 12)),
        icd=(("M54.16", "Radiculopathy, lumbar region"),
             ("M54.17", "Radiculopathy, lumbosacral region"),
             ("M54.12", "Radiculopathy, cervical region"),
             ("M51.16", "Intervertebral disc disorders with radiculopathy, lumbar region"),
             ("M48.062", "Spinal stenosis, lumbar region with neurogenic claudication")),
        alt_pathway="Beneficiaries who do not meet the pain rating criterion may also qualify if the radicular pain prevents participation in physical therapy.",
        outcome="pain scores and functional status",
    ),
    Topic(
        key="vitamin_d_testing", service="vitamin D testing", title="Vitamin D Assay Testing",
        unit="test", condition="vitamin D deficiency", specialty="endocrinology",
        test="serum calcium testing",
        indications=(
            "beneficiaries with chronic kidney disease stage 3 or higher",
            "beneficiaries with osteoporosis",
            "beneficiaries with malabsorption syndromes",
            "beneficiaries with hyperparathyroidism",
            "beneficiaries receiving treatment for documented vitamin D deficiency",
            "beneficiaries with hypercalcemia",
        ),
        eligibility=(
            "The test result will be used in the management of the beneficiary's condition.",
            "The ordering practitioner has documented signs or symptoms supporting the test.",
            "The beneficiary is not being screened in the absence of a qualifying condition.",
        ),
        threshold=Threshold("The beneficiary has a prior 25-hydroxyvitamin D level below {v} ng/mL.",
                            (12, 20, 30), "prior 25-hydroxyvitamin D level", "lower"),
        limitations=(
            "Routine screening for vitamin D deficiency is not covered.",
            "Testing of 1,25-dihydroxyvitamin D is not covered unless hypercalcemia or kidney disease is documented.",
            "Testing performed solely because the beneficiary is taking supplements is not covered.",
            "Repeat testing within 12 weeks of starting replacement therapy is not covered.",
            "Testing ordered as part of a general health panel is not covered.",
        ),
        duration=Duration("Repeat testing is limited to {n} tests within a {m}-month period.", (1, 2, 3, 4), (12, 24)),
        icd=(("E55.9", "Vitamin D deficiency, unspecified"),
             ("N18.4", "Chronic kidney disease, stage 4 (severe)"),
             ("K90.0", "Celiac disease"),
             ("M81.0", "Age-related osteoporosis without current pathological fracture"),
             ("E21.0", "Primary hyperparathyroidism"),
             ("E83.52", "Hypercalcemia")),
        alt_pathway="Beneficiaries who do not meet the prior level criterion may also qualify if they have a documented malabsorption condition.",
        outcome="bone health markers",
    ),
    Topic(
        key="bone_density", service="bone mass measurement", title="Bone Mass Measurement (DXA)",
        unit="scan", condition="osteoporosis", specialty="radiology",
        test="dual-energy x-ray absorptiometry",
        indications=(
            "estrogen-deficient women at clinical risk for osteoporosis",
            "beneficiaries with vertebral abnormalities on imaging",
            "beneficiaries receiving long-term glucocorticoid therapy",
            "beneficiaries with primary hyperparathyroidism",
            "beneficiaries being monitored on an FDA-approved osteoporosis medication",
            "beneficiaries with a history of fragility fracture",
        ),
        eligibility=(
            "The ordering practitioner is treating the beneficiary for a condition that affects bone density.",
            "The results will be used to guide treatment decisions.",
            "The beneficiary has at least one documented risk factor for osteoporosis.",
        ),
        threshold=Threshold("The beneficiary is at least {v} years of age.", (50, 60, 65), "minimum age", "higher"),
        limitations=(
            "Peripheral bone density testing used only for screening is not covered.",
            "Testing more often than the frequency limit is not covered without documented medical necessity.",
            "Bone mass measurement is not covered when performed only to confirm a prior result from a different modality.",
            "Bone mass measurement is not covered for beneficiaries without a qualifying risk factor.",
            "Vertebral fracture assessment billed alone without a qualifying indication is not covered.",
        ),
        duration=Duration("Bone mass measurement is limited to {n} scan every {m} months.", (1,), (12, 18, 23, 24)),
        icd=(("M81.0", "Age-related osteoporosis without current pathological fracture"),
             ("Z79.52", "Long term (current) use of systemic steroids"),
             ("E21.3", "Hyperparathyroidism, unspecified"),
             ("E28.310", "Symptomatic premature menopause"),
             ("Z87.310", "Personal history of (healed) osteoporosis fracture")),
        alt_pathway="Beneficiaries who do not meet the age criterion may also qualify if they are receiving long-term glucocorticoid therapy.",
        outcome="fracture risk assessment",
    ),
    Topic(
        key="skin_substitutes", service="skin substitute grafts", title="Skin Substitute Grafts for Chronic Wounds",
        unit="application", condition="chronic non-healing wounds", specialty="wound care",
        test="vascular assessment",
        indications=(
            "beneficiaries with diabetic foot ulcers",
            "beneficiaries with venous leg ulcers",
            "beneficiaries with full-thickness pressure ulcers",
            "beneficiaries with chronic surgical wounds that have failed to heal",
            "beneficiaries with ulcers related to arterial insufficiency after revascularization",
        ),
        eligibility=(
            "The wound is free of active infection and necrotic tissue.",
            "Adequate circulation to the wound has been documented.",
            "Glycemic control has been addressed for beneficiaries with diabetes.",
        ),
        threshold=Threshold("The ulcer has been present for at least {v} weeks despite standard wound care.",
                            (4, 6, 8), "minimum ulcer duration", "higher"),
        limitations=(
            "Skin substitutes are not covered for wounds with exposed bone or tendon without a surgical plan.",
            "Application to wounds that are improving with standard care is not covered.",
            "Skin substitutes are not covered for beneficiaries who continue to smoke without a cessation plan.",
            "Applications to wounds with untreated osteomyelitis are not covered.",
            "Use of more than one skin substitute product on the same wound during an episode is not covered.",
        ),
        duration=Duration("Skin substitute grafts are limited to {n} applications within a {m}-week episode of care.", (4, 8, 10), (12, 16)),
        icd=(("E11.621", "Type 2 diabetes mellitus with foot ulcer"),
             ("E10.621", "Type 1 diabetes mellitus with foot ulcer"),
             ("L97.411", "Non-pressure chronic ulcer of right heel and midfoot limited to breakdown of skin"),
             ("I87.311", "Chronic venous hypertension (idiopathic) with ulcer of right lower extremity"),
             ("L89.154", "Pressure ulcer of sacral region, stage 4")),
        alt_pathway="Beneficiaries who do not meet the ulcer duration criterion may also qualify if the wound area increased during standard care.",
        outcome="complete wound closure",
    ),
    Topic(
        key="nerve_conduction", service="nerve conduction studies", title="Nerve Conduction Studies and Electromyography",
        unit="study", condition="peripheral nerve disorders", specialty="neurology",
        test="a neurological examination",
        indications=(
            "beneficiaries with suspected carpal tunnel syndrome",
            "beneficiaries with suspected peripheral polyneuropathy",
            "beneficiaries with suspected radiculopathy",
            "beneficiaries with suspected ulnar neuropathy",
            "beneficiaries with suspected motor neuron disease",
            "beneficiaries with suspected myasthenia gravis",
        ),
        eligibility=(
            "A clinical history and examination support the need for testing.",
            "The results are expected to change diagnosis or management.",
            "The study is directed to the specific nerves implicated by the examination.",
        ),
        threshold=Threshold("Symptoms have persisted for at least {v} weeks.", (3, 4, 6), "minimum symptom duration", "higher"),
        limitations=(
            "Studies performed with devices that do not produce waveforms are not covered.",
            "Screening studies in asymptomatic beneficiaries are not covered.",
            "Studies interpreted by a practitioner without electrodiagnostic training are not covered.",
            "Routine repeat studies without a change in clinical status are not covered.",
            "Testing performed solely to establish a baseline before surgery is not covered.",
        ),
        duration=Duration("No more than {n} studies of the same nerve are covered within a {m}-month period.", (1, 2), (6, 12)),
        icd=(("G56.01", "Carpal tunnel syndrome, right upper limb"),
             ("G56.02", "Carpal tunnel syndrome, left upper limb"),
             ("G62.9", "Polyneuropathy, unspecified"),
             ("E11.42", "Type 2 diabetes mellitus with diabetic polyneuropathy"),
             ("M54.12", "Radiculopathy, cervical region")),
        alt_pathway="Beneficiaries who do not meet the symptom duration criterion may also qualify if there is objective muscle weakness.",
        outcome="diagnostic accuracy",
    ),
    Topic(
        key="tms", service="transcranial magnetic stimulation", title="Transcranial Magnetic Stimulation for Depression",
        unit="treatment session", condition="major depressive disorder", specialty="psychiatry",
        test="a standardized depression rating scale",
        indications=(
            "beneficiaries with major depressive disorder without psychotic features",
            "beneficiaries with treatment-resistant depression who cannot tolerate medication",
            "beneficiaries with depressive episodes of bipolar disorder",
            "beneficiaries with recurrent depression who previously responded to transcranial magnetic stimulation",
            "beneficiaries with obsessive-compulsive disorder",
        ),
        eligibility=(
            "A psychiatric evaluation has confirmed the diagnosis.",
            "The beneficiary has no implanted metallic device near the treatment coil.",
            "The beneficiary is not at acute risk of suicide requiring inpatient care.",
        ),
        threshold=Threshold("The beneficiary has not responded to at least {v} adequate trials of antidepressant medication.",
                            (1, 2, 3, 4), "minimum number of failed medication trials", "higher"),
        limitations=(
            "Maintenance treatment after remission is not covered.",
            "Transcranial magnetic stimulation is not covered for beneficiaries with a seizure disorder.",
            "Treatment for depression with psychotic features is not covered.",
            "Treatment delivered with devices not cleared for depression is not covered.",
            "Simultaneous treatment with electroconvulsive therapy is not covered.",
        ),
        duration=Duration("A course of treatment is limited to {n} sessions over {m} weeks.", (30, 36, 42), (6, 7, 9)),
        icd=(("F32.2", "Major depressive disorder, single episode, severe without psychotic features"),
             ("F33.2", "Major depressive disorder, recurrent severe without psychotic features"),
             ("F32.1", "Major depressive disorder, single episode, moderate"),
             ("F33.1", "Major depressive disorder, recurrent, moderate"),
             ("F33.9", "Major depressive disorder, recurrent, unspecified")),
        alt_pathway="Beneficiaries who do not meet the medication trial criterion may also qualify if they had intolerable side effects from two medications.",
        outcome="depression rating scale scores",
    ),
    Topic(
        key="outpatient_pt_lbp", service="outpatient physical therapy", title="Outpatient Physical Therapy for Low Back Pain",
        unit="visit", condition="low back pain", specialty="physical medicine and rehabilitation",
        test="a standardized functional assessment",
        indications=(
            "beneficiaries with acute low back pain and functional limitation",
            "beneficiaries with chronic low back pain",
            "beneficiaries recovering from lumbar spine surgery",
            "beneficiaries with lumbar radiculopathy",
            "beneficiaries with lumbar spinal stenosis",
            "beneficiaries with low back pain following a fall",
        ),
        eligibility=(
            "The beneficiary is under the care of a physician or non-physician practitioner.",
            "A plan of care has been established and certified.",
            "The services require the skills of a qualified therapist.",
        ),
        threshold=Threshold("The beneficiary has a functional disability score of at least {v} percent on a validated low back pain questionnaire.",
                            (20, 30, 40), "functional disability score", "higher"),
        limitations=(
            "Services that can be safely performed by the beneficiary without a therapist are not covered.",
            "Maintenance therapy without a skilled need is not covered.",
            "Passive modalities furnished as the only treatment are not covered.",
            "Physical therapy is not covered for beneficiaries with suspected cauda equina syndrome before surgical evaluation.",
            "Group therapy sessions without individualized interventions are not covered.",
        ),
        duration=Duration("Physical therapy is limited to {n} visits within a {m}-week period before re-certification.", (8, 12, 16), (6, 8, 12)),
        icd=(("M54.50", "Low back pain, unspecified"),
             ("M54.59", "Other low back pain"),
             ("M51.26", "Other intervertebral disc displacement, lumbar region"),
             ("M47.816", "Spondylosis without myelopathy or radiculopathy, lumbar region"),
             ("M54.16", "Radiculopathy, lumbar region")),
        alt_pathway="Beneficiaries who do not meet the disability score criterion may also qualify after lumbar spine surgery.",
        outcome="pain and functional outcomes",
    ),
    Topic(
        key="hyperbaric_oxygen", service="hyperbaric oxygen therapy", title="Hyperbaric Oxygen Therapy",
        unit="treatment", condition="hypoxic wounds and related conditions", specialty="undersea and hyperbaric medicine",
        test="transcutaneous oxygen measurement",
        indications=(
            "beneficiaries with diabetic lower extremity wounds",
            "beneficiaries with acute carbon monoxide poisoning",
            "beneficiaries with decompression illness",
            "beneficiaries with osteoradionecrosis",
            "beneficiaries with chronic refractory osteomyelitis",
            "beneficiaries with compromised skin grafts or flaps",
        ),
        eligibility=(
            "The wound has failed to respond to standard wound therapy for at least 30 days.",
            "The beneficiary has been evaluated for hyperbaric treatment by the supervising physician.",
            "There is no untreated pneumothorax.",
        ),
        threshold=Threshold("The wound is classified as Wagner grade {v} or higher.", (2, 3, 4), "minimum Wagner grade", "higher"),
        limitations=(
            "Topical oxygen applied to the wound is not covered as hyperbaric oxygen therapy.",
            "Treatment continued without measurable signs of healing after 30 days is not covered.",
            "Hyperbaric oxygen therapy is not covered for beneficiaries with untreated pneumothorax.",
            "Treatment for chronic peripheral vascular insufficiency alone is not covered.",
            "Hyperbaric oxygen therapy is not covered for acute cerebral edema.",
        ),
        duration=Duration("Hyperbaric oxygen therapy is limited to {n} treatments within a {m}-day period.", (20, 30, 40), (30, 45, 60)),
        icd=(("E11.621", "Type 2 diabetes mellitus with foot ulcer"),
             ("E10.621", "Type 1 diabetes mellitus with foot ulcer"),
             ("T70.3XXA", "Caisson disease [decompression sickness], initial encounter"),
             ("T58.01XA", "Toxic effect of carbon monoxide from motor vehicle exhaust, accidental (unintentional), initial encounter"),
             ("M27.8", "Other specified diseases of jaws")),
        alt_pathway="Beneficiaries who do not meet the wound grade criterion may also qualify if osteomyelitis has been confirmed.",
        outcome="wound healing and amputation rates",
    ),
]

TOPICS_BY_KEY = {t.key: t for t in TOPICS}

# Leakage-safe family splits. Test families never appear in training data.
SPLIT_FAMILIES = {
    "train": ["pulmonary_rehab", "cardiac_rehab", "sleep_apnea_pap", "home_oxygen",
              "epidural_steroid", "vitamin_d_testing", "skin_substitutes", "outpatient_pt_lbp"],
    "val": ["nerve_conduction"],
    "test": ["tms", "bone_density", "hyperbaric_oxygen"],
}

# ------------------------------------------------------------ shared pools
GENERIC_ELIGIBILITY = (
    "The beneficiary has been evaluated in person by the treating practitioner within the past {n} days.",
    "The beneficiary has a documented diagnosis of {condition}.",
    "The beneficiary is able to understand and participate in the plan of care.",
    "Conservative treatment has been tried and has not produced adequate improvement.",
    "The treating practitioner expects measurable improvement within a predictable period.",
)

DOCUMENTATION = (
    "The medical record must contain a signed and dated order for {service} from the treating practitioner.",
    "Progress notes must document the beneficiary's response to each {unit}.",
    "The medical record must include the diagnosis that supports medical necessity.",
    "A plan of care that specifies goals, frequency and expected duration must be on file before {service} begins.",
    "Documentation must identify the practitioner who performed or supervised each {unit}.",
    "Results of {test} relied upon to establish medical necessity must be available in the medical record.",
    "The treating practitioner must document a face-to-face evaluation within 30 days before the initial {unit}.",
    "Documentation must describe prior treatments attempted and the outcome of each.",
    "When {service} continues beyond the initial period, the record must document measurable improvement.",
    "The record must include a signed attestation that the beneficiary received education about {service}.",
    "Documentation must state the clinical reason any {unit} was missed or discontinued.",
)

PROVIDER_QUALS = (
    "{Service} must be ordered by a physician or non-physician practitioner who is treating the beneficiary for {condition}.",
    "Services must be performed by personnel who are licensed or certified to perform {service} under state law.",
    "The performing practitioner must have documented training and experience in {service}.",
    "The interpreting practitioner must be board certified in {specialty}.",
    "Facilities furnishing {service} must maintain written protocols for emergency response.",
    "The ordering practitioner must be enrolled in Medicare.",
    "Staff furnishing {service} must hold current basic life support certification.",
)

SUPERVISION_LEVELS = ("general", "direct", "personal")  # increasing stringency
SUPERVISION_TEMPLATE = "{Service} must be furnished under {level} supervision of a physician."

BILLING = (
    "Report one unit of service for each {unit} furnished.",
    "The ICD-10-CM code that supports medical necessity must be reported on the claim.",
    "Services that do not meet the criteria in this policy should be reported with the modifier indicating that denial is expected.",
    "Claims for {service} furnished on the same date of service by the same provider should be combined on a single claim line.",
    "Report the date of the initial {unit} on the claim.",
    "Services must be billed with the place of service code that reflects where the {unit} was furnished.",
    "Telehealth delivery of {service} is not separately billable under this policy.",
    "Claims submitted without the ordering practitioner's National Provider Identifier will be rejected.",
)

BILLING_MODIFICATIONS = {
    "Report one unit of service for each {unit} furnished.":
        "Report one unit of service per date of service regardless of the number of {unit}s furnished.",
    "Report the date of the initial {unit} on the claim.":
        "Report the date of the initial {unit} and the total number of {unit}s to date on the claim.",
    "Telehealth delivery of {service} is not separately billable under this policy.":
        "Telehealth delivery of {service} may be billed when the telehealth modifier is appended.",
    "Claims for {service} furnished on the same date of service by the same provider should be combined on a single claim line.":
        "Claims for {service} furnished on the same date of service by the same provider should be reported on separate claim lines.",
}

QUAL_MODIFICATIONS = {
    "The performing practitioner must have documented training and experience in {service}.":
        "The performing practitioner must have completed a recognized training program in {service} and performed at least 25 supervised procedures.",
    "Services must be performed by personnel who are licensed or certified to perform {service} under state law.":
        "Services must be performed by a licensed physician, physician assistant or nurse practitioner.",
    "The ordering practitioner must be enrolled in Medicare.":
        "The ordering practitioner must be enrolled in Medicare and must not be the same practitioner who furnishes {service}.",
}

EVIDENCE_REQ_TEMPLATE = "Results of {test} obtained within {n} months before the initial {unit} must be available."
EVIDENCE_REQ_VALUES = (3, 6, 12, 18)
EVIDENCE_REQ_EXTRA = (
    "A baseline result from {test} must be recorded before the initial {unit}.",
    "Objective findings that confirm {condition} must be documented by the treating practitioner.",
    "Results of a repeat assessment must be documented before continued coverage is approved.",
)

EVIDENCE_SUMMARY = (
    "Published studies report improvements in {outcome} among beneficiaries receiving {service}.",
    "The evidence reviewed consists primarily of randomized controlled trials and prospective cohort studies.",
    "Evidence for benefit is strongest when {service} is combined with ongoing medical management.",
    "Studies with longer follow-up show that benefits decline without continued adherence.",
)
EVIDENCE_SUMMARY_UPDATES = (
    "A {year} systematic review reported findings consistent with the studies summarized above.",
    "Additional observational data published in {year} were reviewed and did not change the conclusions of this policy.",
    "A {year} guideline update from a professional society was reviewed and is consistent with this policy.",
)

APPLICABILITY_TEMPLATES = (
    "The requirements of this policy apply to services performed on or after {date}.",
    "This policy applies to claims with dates of service on or after {date}.",
)

REFERENCE_TEMPLATES = (
    "Fictional Society of {specialty_title}. Clinical practice guideline for {condition}. {year} edition.",
    "Synthetic Evidence Review Group. Systematic review of {service} outcomes. {year}.",
    "Example Journal of {specialty_title}. Outcomes of {service} in older adults. {year}.",
)

CONTACT_TEMPLATES = (
    "Questions about this policy may be directed to the fictional contractor policy unit at {email}.",
    "Inquiries may be submitted to the fictional contractor medical review department at {phone}.",
)

TYPO_WORDS = {
    "beneficiary": "beneficary",
    "documented": "documeted",
    "treatment": "treatmnet",
    "practitioner": "practioner",
    "medical": "medcal",
    "furnished": "furnshed",
    "covered": "coverd",
}

# Rewording transforms that preserve meaning (used for no-substantive-change cases).
REWORDINGS = (
    (r"\bmust\b", "shall"),
    (r"\bbefore\b", "prior to"),
    (r"\bbeneficiary's\b", "patient's"),
    (r"\bbeneficiary\b", "patient"),
    (r"\bbeneficiaries\b", "patients"),
    (r"is considered reasonable and necessary", "is considered medically reasonable and necessary"),
    (r"\bwithin the past\b", "within the preceding"),
    (r"has not produced adequate improvement", "has failed to produce adequate improvement"),
    (r"\bare not covered\b", "are non-covered"),
)

COVERAGE_TEMPLATES = (
    "{Service} is considered reasonable and necessary for {ind}.",
    "{Service} may be covered for {ind}.",
    "Coverage of {service} is provided for {ind}.",
)

HEADINGS = {
    "coverage_indications": ("Coverage Indications", "Indications for Coverage", "Covered Indications"),
    "eligibility": ("Patient Selection Criteria", "Eligibility Criteria", "Medical Necessity Criteria"),
    "limitations": ("Limitations", "Limitations and Exclusions", "Non-Covered Services"),
    "utilization_guidelines": ("Utilization Guidelines", "Frequency and Duration"),
    "evidence_requirements": ("Required Testing", "Clinical Evidence Required for Coverage"),
    "documentation_requirements": ("Documentation Requirements", "Documentation Requirements for the Medical Record"),
    "provider_qualifications": ("Provider Qualifications", "Practitioner Qualifications and Supervision"),
    "coding": ("ICD-10-CM Codes that Support Medical Necessity", "Covered Diagnosis Codes (ICD-10-CM)"),
    "billing": ("Billing and Coding Guidelines", "Billing Instructions"),
    "summary_of_evidence": ("Summary of Evidence", "Summary of Evidence and Rationale"),
    "effective_dates": ("Applicability", "Implementation"),
    "sources": ("Sources of Information", "References"),
    "contacts": ("Contractor Contact Information", "Policy Contact"),
    "retirement_notice": ("Retirement Notice", "Retirement and Replacement"),
}

INTROS = {
    "coverage_indications": ("{Service} is covered when the requirements of this policy are met.",
                             "This section describes the indications for which {service} may be covered."),
    "eligibility": ("Coverage requires that all of the following criteria are met:",
                    "The beneficiary must meet each of the following criteria:"),
    "limitations": ("The following are not considered reasonable and necessary:",
                    "The following limitations apply:"),
    "utilization_guidelines": (None, "The following frequency and duration limits apply:"),
    "evidence_requirements": ("The following clinical evidence is required:", None),
    "documentation_requirements": ("The medical record must support the medical necessity of each service.",
                                   "The following documentation must be available upon request:"),
    "provider_qualifications": (None, "The following qualifications apply:"),
    "coding": ("The following ICD-10-CM codes support medical necessity for {service}:",
               "Claims must include one of the following ICD-10-CM codes:"),
    "billing": (None, "The following billing instructions apply:"),
    "summary_of_evidence": (None,),
    "effective_dates": (None,),
    "sources": (None,),
    "contacts": (None,),
}
