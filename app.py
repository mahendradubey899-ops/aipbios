"""
AIPBIOS Live Prototype — OpenAI Connected
Flask + SQLite + OpenAI GPT-4o
Any disease, any formulation — real AI responses.
"""
import sqlite3, uuid, hashlib, json, datetime, os, io, threading
from flask import Flask, request, jsonify, g, send_from_directory, make_response
import jwt
from functools import wraps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use /tmp on Render (writable), local path for development
DB_PATH  = os.environ.get('DB_PATH', os.path.join('/tmp', 'aipbios.db'))
SECRET   = 'aipbios-live-prototype-secret-2024'
app      = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))

# ── OpenAI setup ──────────────────────────────────────────────────────────────
# OPENAI_API_KEY read dynamically in call_openai to pick up env changes

def call_openai(system_prompt, user_prompt, model='gpt-4o'):
    """Call OpenAI and return parsed JSON output."""
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return None, "OPENAI_API_KEY not set"
    try:
        import urllib.request as ur
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }).encode()
        req = ur.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST"
        )
        with ur.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        content = data["choices"][0]["message"]["content"]
        tokens  = data.get("usage", {}).get("total_tokens", 0)
        return json.loads(content), tokens
    except Exception as e:
        return None, str(e)

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def q(sql, args=(), one=False):
    cur = get_db().execute(sql, args)
    get_db().commit()
    rv = cur.fetchall()
    return (dict(rv[0]) if rv else None) if one else [dict(r) for r in rv]

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, first_name TEXT DEFAULT '',
        last_name TEXT DEFAULT '', role TEXT DEFAULT 'researcher',
        organisation TEXT DEFAULT 'AIPBIOS Demo',
        is_active INTEGER DEFAULT 1, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY, title TEXT NOT NULL,
        description TEXT DEFAULT '', status TEXT DEFAULT 'active',
        disease_area TEXT DEFAULT '', product_type TEXT DEFAULT '',
        created_by TEXT, job_count INTEGER DEFAULT 0,
        created_at TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS intelligence_jobs (
        id TEXT PRIMARY KEY, project_id TEXT, created_by TEXT,
        module_type TEXT, status TEXT DEFAULT 'completed',
        input_payload TEXT DEFAULT '{}',
        output_payload TEXT DEFAULT '{}',
        tokens_used INTEGER DEFAULT 0,
        created_at TEXT, completed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, project_id TEXT, job_id TEXT,
        title TEXT, doc_type TEXT, storage_path TEXT DEFAULT '',
        created_at TEXT
    );
    """)
    db.commit(); db.close()

def h(pw): return hashlib.sha256(pw.encode()).hexdigest()
def uid(): return str(uuid.uuid4())
def now(): return datetime.datetime.utcnow().isoformat()

def make_token(user):
    exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    return jwt.encode(
        {'user_id': user['id'], 'email': user['email'],
         'role': user['role'], 'exp': exp},
        SECRET, algorithm='HS256'
    )

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization','').replace('Bearer ','').strip()
        if not token:
            return jsonify({'error': True, 'message': 'Authentication required'}), 401
        try:
            g.user = jwt.decode(token, SECRET, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': True, 'message': 'Token expired'}), 401
        except Exception:
            return jsonify({'error': True, 'message': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

# ── AI PROMPTS ────────────────────────────────────────────────────────────────
# ── EXPERT PHARMACEUTICAL SYSTEM PROMPTS ─────────────────────────────────────

DISEASE_SYSTEM = """You are a senior pharmaceutical intelligence analyst with 20+ years experience at IQVIA, Clarivate, and WHO. You have deep expertise in Indian pharmaceutical markets, CDSCO regulations, AYUSH policy, and global disease epidemiology.

Generate a COMPREHENSIVE, EXPERT-LEVEL disease intelligence report. Include:
- India-specific epidemiology with actual numbers from ICMR, MOHFW, WHO
- Exact burden data: incidence, prevalence, mortality, DALYs
- Current standard of care with specific drug names, doses, costs in INR
- Unmet medical needs with clinical and economic evidence
- Regulatory opportunities including CDSCO fast track, AYUSH pathways
- Market sizing in INR crores AND USD millions
- Specific innovation opportunities with mechanism of action
- Named competitor products and their limitations
- Published clinical evidence (cite author, journal, year)
- State-wise burden for India
- Seasonal patterns if relevant

Return ONLY valid JSON with this structure:
{
  "executive_summary": "3-4 sentence expert summary with key numbers",
  "disease_overview": {
    "name": "",
    "icd_11_code": "",
    "icd_10_code": "",
    "category": "",
    "pathophysiology": "detailed mechanism",
    "disease_subtypes": [],
    "severity_classification": ""
  },
  "epidemiology": {
    "global_prevalence": "",
    "global_incidence": "",
    "india_prevalence": "",
    "india_incidence": "",
    "india_mortality": "",
    "dalys_lost": "",
    "state_wise_burden": {"state": "burden"},
    "high_risk_groups": [],
    "geographic_hotspots": [],
    "seasonal_pattern": "",
    "data_sources": []
  },
  "clinical_profile": {
    "symptoms": [],
    "diagnostic_criteria": "",
    "diagnostic_tests": [],
    "complications": [],
    "prognosis": ""
  },
  "current_treatment_landscape": {
    "standard_of_care": [{"drug": "", "dose": "", "duration": "", "cost_inr": "", "limitations": ""}],
    "treatment_guidelines": [],
    "treatment_gaps": [],
    "patient_compliance_issues": ""
  },
  "unmet_medical_needs": {
    "clinical_gaps": [],
    "patient_burden_qualitative": "",
    "economic_burden_inr": "",
    "economic_burden_usd": "",
    "caregiver_burden": ""
  },
  "market_analysis": {
    "india_market_size_inr_crores": "",
    "india_market_size_usd_million": "",
    "global_market_size_usd_billion": "",
    "market_growth_cagr": "",
    "market_drivers": [],
    "market_barriers": [],
    "key_players_india": [],
    "key_players_global": []
  },
  "opportunity_analysis": {
    "opportunity_score": 0,
    "opportunity_rationale": "",
    "first_mover_advantage": "",
    "regulatory_pathway_advantage": "",
    "key_value_drivers": [],
    "risk_factors": []
  },
  "innovation_suggestions": [
    {
      "title": "",
      "mechanism_of_action": "",
      "active_ingredients": [],
      "clinical_rationale": "",
      "supporting_evidence": [{"author": "", "journal": "", "year": "", "finding": ""}],
      "novelty_level": "",
      "feasibility": "",
      "estimated_development_cost_inr": "",
      "time_to_market_years": 0,
      "target_claim": ""
    }
  ],
  "regulatory_strategy": {
    "india_pathway": "",
    "cdsco_designation": "",
    "ayush_pathway": "",
    "estimated_timeline_months": 0,
    "estimated_cost_inr_lakhs": 0,
    "key_requirements": []
  },
  "research_priorities": [
    {"priority": "", "rationale": "", "study_type": "", "estimated_duration": ""}
  ],
  "competitive_intelligence": {
    "pipeline_products": [],
    "recent_approvals": [],
    "patent_landscape": ""
  },
  "confidence_level": "",
  "key_references": [{"author": "", "title": "", "journal": "", "year": "", "doi": ""}],
  "analyst_recommendation": ""
}"""

FORMULATION_SYSTEM = """You are a Principal Formulation Scientist with PhD in Pharmaceutical Sciences and 15+ years experience in drug product development at leading Indian pharma companies. Expert in Ayurvedic, Allopathic, and Nutraceutical formulations.

Generate a COMPLETE, SCIENTIFICALLY RIGOROUS formulation design with:
- Evidence-based active ingredient selection with published dose justification
- Complete excipient list with pharmacopoeial grade, supplier, and function
- FULL batch formula for specified batch size with exact quantities
- Step-by-step manufacturing method with critical process parameters
- Complete evaluation parameters (physical, chemical, microbiological)
- In-process quality control checkpoints
- Stability considerations and packaging
- Cost-of-goods estimate in INR
- Equipment list with specifications
- Alternative ingredients if primary unavailable

Return ONLY valid JSON:
{
  "product_overview": {
    "proposed_name": "",
    "category": "",
    "dosage_form": "",
    "route_of_administration": "",
    "strength": "",
    "target_indication": "",
    "target_patient_profile": "",
    "shelf_life_proposed": ""
  },
  "formulation_rationale": "",
  "active_ingredients": [
    {
      "name": "",
      "iupac_name": "",
      "cas_number": "",
      "pharmacopoeial_standard": "",
      "pharmacological_class": "",
      "mechanism_of_action": "",
      "dose_per_unit": "",
      "dose_justification": "",
      "published_reference": {"author": "", "journal": "", "year": ""},
      "solubility": "",
      "stability_concerns": "",
      "role_in_formulation": "",
      "alternative_if_unavailable": ""
    }
  ],
  "herbal_ingredients": [
    {
      "common_name": "",
      "botanical_name": "",
      "family": "",
      "plant_part_used": "",
      "extract_type": "",
      "standardisation_marker": "",
      "quantity_per_unit": "",
      "therapeutic_action": "",
      "traditional_use": "",
      "clinical_evidence": "",
      "pharmacopoeial_reference": "",
      "alternative_if_unavailable": ""
    }
  ],
  "excipients": [
    {
      "name": "",
      "iupac_or_common_name": "",
      "category": "",
      "quantity_per_unit": "",
      "quantity_per_batch": "",
      "function": "",
      "grade": "",
      "pharmacopoeial_standard": "",
      "typical_supplier": "",
      "alternatives": []
    }
  ],
  "batch_formula": {
    "batch_size_units": 0,
    "batch_size_kg": "",
    "theoretical_yield_percent": "",
    "overages": {},
    "components": [
      {
        "sr_no": 0,
        "ingredient": "",
        "role": "",
        "quantity_per_unit_mg": 0,
        "quantity_per_batch_kg": 0,
        "overage_percent": 0,
        "quantity_with_overage_kg": 0
      }
    ],
    "total_weight_per_unit_mg": 0,
    "total_batch_weight_kg": 0
  },
  "manufacturing_method": {
    "process_name": "",
    "process_flow": [],
    "detailed_steps": [
      {
        "step_number": 0,
        "operation": "",
        "equipment": "",
        "parameters": {},
        "duration": "",
        "critical_control_point": false,
        "acceptance_criteria": "",
        "in_process_check": ""
      }
    ],
    "critical_process_parameters": [],
    "process_validation_requirements": []
  },
  "evaluation_parameters": {
    "physical_tests": [
      {
        "test": "",
        "method": "",
        "specification": "",
        "frequency": "",
        "reference_standard": ""
      }
    ],
    "chemical_tests": [
      {
        "test": "",
        "method": "",
        "specification": "",
        "frequency": "",
        "reference_standard": ""
      }
    ],
    "microbiological_tests": [
      {
        "test": "",
        "specification": "",
        "reference_standard": ""
      }
    ],
    "dissolution_test": {
      "apparatus": "",
      "medium": "",
      "rpm": 0,
      "time_points": [],
      "specification": ""
    },
    "stability_tests": [
      {
        "condition": "",
        "duration": "",
        "parameters_to_test": [],
        "acceptance_criteria": ""
      }
    ]
  },
  "equipment_list": [
    {
      "equipment": "",
      "capacity": "",
      "specification": "",
      "purpose": ""
    }
  ],
  "packaging": {
    "primary_packaging": "",
    "secondary_packaging": "",
    "storage_conditions": "",
    "shelf_life": "",
    "justification": ""
  },
  "cost_estimate": {
    "raw_material_cost_per_unit_inr": 0,
    "manufacturing_cost_per_unit_inr": 0,
    "packaging_cost_per_unit_inr": 0,
    "total_cogs_per_unit_inr": 0,
    "suggested_mrp_inr": 0,
    "gross_margin_percent": 0
  },
  "regulatory_classification": {
    "drug_category": "",
    "applicable_schedule": "",
    "license_required": "",
    "gmp_standard": ""
  },
  "formulation_notes": "",
  "scale_up_considerations": ""
}"""

LITERATURE_SYSTEM = """You are a senior medical research analyst and systematic review expert with extensive PubMed database knowledge. Expert in evidence-based medicine, clinical pharmacology, and pharmaceutical research methodology.

Generate a COMPREHENSIVE literature review with:
- Actual published studies with real authors, journals, years, and PMIDs where known
- Evidence hierarchy (Level I meta-analyses to Level V expert opinion)
- Statistical findings with actual numbers (p-values, confidence intervals, effect sizes)
- Research gaps with specific study design recommendations
- PubMed search strategy with MeSH terms
- Indian research landscape specific analysis
- Future research directions with fundable grant proposals

Return ONLY valid JSON:
{
  "executive_summary": "",
  "topic_overview": {
    "standardised_topic": "",
    "mesh_terms": [],
    "related_terms": [],
    "field_maturity": "",
    "total_publications_estimated": "",
    "growth_trend": ""
  },
  "evidence_pyramid": {
    "level_1_meta_analyses": [
      {
        "title": "",
        "authors": "",
        "journal": "",
        "year": "",
        "pmid": "",
        "doi": "",
        "n_studies_included": 0,
        "n_patients": 0,
        "key_finding": "",
        "effect_size": "",
        "confidence_interval": "",
        "p_value": "",
        "evidence_level": "I"
      }
    ],
    "level_2_rcts": [
      {
        "title": "",
        "authors": "",
        "journal": "",
        "year": "",
        "pmid": "",
        "study_design": "",
        "n_patients": 0,
        "duration": "",
        "intervention": "",
        "comparator": "",
        "primary_endpoint": "",
        "primary_result": "",
        "p_value": "",
        "evidence_level": "II"
      }
    ],
    "level_3_observational": [],
    "level_4_case_series": [],
    "level_5_expert_opinion": []
  },
  "key_findings_summary": {
    "efficacy_evidence": [],
    "safety_evidence": [],
    "pharmacology_evidence": [],
    "clinical_practice_evidence": []
  },
  "indian_research_landscape": {
    "key_indian_studies": [],
    "icmr_publications": [],
    "ayush_research": [],
    "indian_research_gaps": []
  },
  "research_gaps": [
    {
      "gap": "",
      "clinical_significance": "",
      "priority": "",
      "suggested_study_design": "",
      "estimated_sample_size": "",
      "estimated_duration": "",
      "estimated_budget_inr_lakhs": 0,
      "potential_funding_sources": []
    }
  ],
  "consensus_and_controversy": {
    "consensus_areas": [],
    "contested_areas": [],
    "ongoing_debates": []
  },
  "research_summary": {
    "state_of_evidence": "",
    "strength_of_evidence": "",
    "clinical_translation_status": "",
    "regulatory_acceptability": "",
    "future_directions": []
  },
  "pubmed_search_strategy": {
    "primary_query": "",
    "secondary_queries": [],
    "filters": [],
    "databases": [],
    "estimated_results": ""
  },
  "grant_opportunities": [
    {
      "funding_body": "",
      "scheme": "",
      "amount_inr_lakhs": 0,
      "eligibility": "",
      "application_deadline": "",
      "website": ""
    }
  ],
  "key_references": [
    {
      "citation_number": 0,
      "authors": "",
      "title": "",
      "journal": "",
      "year": "",
      "volume": "",
      "pages": "",
      "pmid": "",
      "doi": "",
      "evidence_level": ""
    }
  ],
  "analyst_notes": ""
}"""

REGULATORY_SYSTEM = """You are a senior regulatory affairs specialist with 20+ years experience at CDSCO, AYUSH Ministry, and leading CROs. Expert in Indian drug regulation, CTD dossier preparation, and global regulatory strategy.

Generate a COMPREHENSIVE regulatory intelligence report with:
- Exact CDSCO/AYUSH form numbers and fee schedules
- Specific Schedule references from Drugs & Cosmetics Act
- Current regulatory timelines based on 2023-2024 CDSCO data
- Step-by-step submission checklist with document specifications
- Common deficiency letter (CRL) points to avoid
- Specific Indian regulatory consultants and CROs
- Clinical trial requirements with CTRI registration details
- Post-approval requirements including pharmacovigilance

Return ONLY valid JSON:
{
  "executive_summary": "",
  "product_classification": {
    "category": "",
    "drug_type": "",
    "schedule_india": "",
    "atc_code": "",
    "new_drug_status": "",
    "applicable_rules": []
  },
  "regulatory_pathways": [
    {
      "authority": "",
      "pathway_name": "",
      "application_form": "",
      "application_fee_inr": 0,
      "estimated_timeline_months": 0,
      "estimated_total_cost_inr_lakhs": 0,
      "expedited_designations": [],
      "key_requirements": [],
      "common_deficiencies_to_avoid": [],
      "success_rate_percent": 0
    }
  ],
  "required_documents": {
    "module_1_administrative": [],
    "module_2_summaries": [],
    "module_3_quality_cmc": [],
    "module_4_nonclinical": [],
    "module_5_clinical": [],
    "additional_india_specific": []
  },
  "compliance_checklist": [
    {
      "item": "",
      "category": "",
      "authority": "",
      "mandatory": true,
      "guidance_reference": "",
      "cdsco_circular": "",
      "completion_status": "pending"
    }
  ],
  "clinical_trial_requirements": {
    "phase_required": "",
    "waiver_possible": false,
    "waiver_criteria": "",
    "ctri_registration": "mandatory",
    "ethics_committee_requirement": "",
    "sample_size_guidance": "",
    "gcp_guidelines": "ICMR GCP 2017"
  },
  "gmp_requirements": {
    "applicable_schedule": "",
    "audit_agency": "",
    "audit_frequency": "",
    "key_gmp_requirements": []
  },
  "timeline_milestones": [
    {
      "month": 0,
      "milestone": "",
      "responsible_party": "",
      "deliverable": "",
      "estimated_cost_inr_lakhs": 0
    }
  ],
  "post_approval_requirements": {
    "pharmacovigilance": "",
    "periodic_safety_update": "",
    "post_market_surveillance": "",
    "renewal_requirements": ""
  },
  "regulatory_strategy_recommendation": "",
  "recommended_cros_india": [
    {
      "name": "",
      "location": "",
      "speciality": "",
      "approximate_cost_inr_lakhs": 0
    }
  ],
  "regulatory_risks": [
    {
      "risk": "",
      "probability": "",
      "impact": "",
      "mitigation": ""
    }
  ],
  "consultant_notes": ""
}"""

PATENT_SYSTEM = """You are a senior pharmaceutical patent attorney and IP strategist with expertise in Indian Patents Act 1970, PCT applications, and pharmaceutical IP strategy.

Generate a COMPREHENSIVE patent intelligence report with:
- Prior art analysis with actual patent numbers
- Freedom to operate preliminary assessment
- Patentability score with detailed justification
- Claim drafting strategy
- India-specific patent considerations (Section 3d, compulsory licensing)
- Filing strategy with cost estimates in INR
- Patent landscape map

Return ONLY valid JSON:
{
  "executive_summary": "",
  "invention_analysis": {
    "title_suggested": "",
    "technology_area": "",
    "invention_type": "",
    "novelty_elements": [],
    "inventive_step_elements": [],
    "industrial_applicability": ""
  },
  "prior_art_review": {
    "search_databases": ["Indian Patent Office", "USPTO", "EPO", "WIPO"],
    "closest_prior_art": [
      {
        "patent_number": "",
        "title": "",
        "assignee": "",
        "inventors": "",
        "filing_date": "",
        "grant_date": "",
        "jurisdiction": "",
        "key_claims": [],
        "relevance_to_invention": "",
        "distinguishing_features": ""
      }
    ],
    "relevant_scientific_literature": [],
    "freedom_to_operate_opinion": "",
    "blocking_patents": []
  },
  "patentability_assessment": {
    "novelty_score": 0,
    "novelty_analysis": "",
    "inventive_step_score": 0,
    "inventive_step_analysis": "",
    "industrial_applicability_score": 0,
    "overall_patentability_score": 0,
    "patentability_opinion": "",
    "section_3d_india_risk": "",
    "compulsory_licensing_risk": ""
  },
  "claim_strategy": {
    "recommended_claim_types": [],
    "independent_claims_suggested": [],
    "dependent_claims_suggested": [],
    "method_claims": [],
    "composition_claims": []
  },
  "filing_strategy": {
    "recommended_first_filing": "",
    "provisional_application": true,
    "pct_application": true,
    "recommended_jurisdictions": [],
    "filing_timeline": [],
    "estimated_cost_india_inr_lakhs": 0,
    "estimated_cost_pct_usd": 0,
    "patent_attorney_recommendation": ""
  },
  "ip_landscape": {
    "key_patent_holders": [],
    "patent_expiry_opportunities": [],
    "white_space_areas": []
  },
  "analyst_notes": ""
}"""

STABILITY_SYSTEM = """You are a pharmaceutical stability expert with deep knowledge of ICH Q1A-Q1F, WHO stability guidelines, and Indian climatic zone requirements.

Generate a COMPREHENSIVE stability programme with:
- Complete ICH-compliant stability design
- India-specific Zone IVb (30°C/65%RH) requirements
- Specific test parameters with acceptance criteria
- Degradation pathway analysis
- Packaging selection with moisture vapour transmission rate data
- Statistical analysis approach for shelf-life estimation
- Arrhenius equation application
- Stability-indicating method requirements

Return ONLY valid JSON:
{
  "executive_summary": "",
  "stability_classification": {
    "product_type": "",
    "ich_zone": "IVb (India)",
    "storage_condition_proposed": "",
    "shelf_life_target": ""
  },
  "stability_programme": {
    "real_time_studies": [
      {
        "study_type": "Long-term",
        "condition": "30°C ± 2°C / 65% RH ± 5%",
        "duration_months": 24,
        "test_intervals_months": [0, 3, 6, 9, 12, 18, 24],
        "n_batches": 3,
        "container_orientations": ["upright", "inverted"]
      }
    ],
    "accelerated_studies": [
      {
        "study_type": "Accelerated",
        "condition": "40°C ± 2°C / 75% RH ± 5%",
        "duration_months": 6,
        "test_intervals_months": [0, 1, 2, 3, 6],
        "n_batches": 3
      }
    ],
    "stress_studies": [
      {
        "study_type": "",
        "condition": "",
        "purpose": "",
        "duration": ""
      }
    ]
  },
  "test_parameters": [
    {
      "parameter": "",
      "test_method": "",
      "pharmacopoeial_reference": "",
      "specification": "",
      "stability_indicating": true,
      "frequency": ""
    }
  ],
  "degradation_pathways": [
    {
      "pathway": "",
      "conditions": [],
      "degradation_products": [],
      "risk_level": "",
      "mitigation_strategy": ""
    }
  ],
  "analytical_methods": [
    {
      "method_name": "",
      "technique": "",
      "parameters_tested": [],
      "validation_required": true,
      "ich_validation_parameters": []
    }
  ],
  "packaging_evaluation": {
    "primary_packaging_options": [
      {
        "packaging_type": "",
        "wvtr_g_m2_day": 0,
        "oxygen_transmission_rate": "",
        "light_protection": "",
        "cost_per_unit_inr": 0,
        "recommendation": ""
      }
    ],
    "recommended_packaging": "",
    "justification": ""
  },
  "shelf_life_prediction": {
    "method": "Arrhenius equation / ICH Q1E",
    "accelerated_data_extrapolation": "",
    "predicted_shelf_life": "",
    "confidence_level": "",
    "label_claim_recommended": ""
  },
  "regulatory_requirements": {
    "ich_guidelines_applicable": [],
    "indian_regulatory_requirements": "",
    "submission_requirements": "",
    "post_approval_commitments": []
  },
  "stability_budget_estimate": {
    "analytical_testing_cost_inr_lakhs": 0,
    "chamber_rental_cost_inr_lakhs": 0,
    "total_estimated_cost_inr_lakhs": 0,
    "timeline_months": 0
  },
  "stability_notes": ""
}"""

ANALYTICAL_SYSTEM = """You are a senior analytical chemist with expertise in pharmaceutical analysis, chromatography, spectroscopy, and analytical method validation per ICH Q2(R1).

Interpret the analytical data/image and provide:
- Detailed interpretation of peaks/signals/bands
- Identification of compounds with confidence level
- Quality assessment against pharmacopoeial standards
- Method performance evaluation
- Impurity profile assessment
- Actionable recommendations
- Comparison with reference standards

Return ONLY valid JSON:
{
  "analysis_summary": "",
  "technique": "",
  "sample_information": {
    "sample_name": "",
    "sample_type": "",
    "analysis_date": "",
    "analyst_notes": ""
  },
  "system_suitability": {
    "parameters_evaluated": [],
    "results": {},
    "pass_fail": "",
    "comments": ""
  },
  "peaks_signals_identified": [
    {
      "peak_number": 0,
      "retention_time_or_wavelength": "",
      "area_or_absorbance": "",
      "area_percent": "",
      "identification": "",
      "confidence_level": "",
      "pharmacopoeial_reference": "",
      "specification": "",
      "result": ""
    }
  ],
  "quantitative_results": [
    {
      "analyte": "",
      "amount_found": "",
      "specification": "",
      "result": "",
      "uncertainty": ""
    }
  ],
  "impurity_profile": {
    "total_impurities": "",
    "individual_impurities": [],
    "unknown_impurities": [],
    "ich_q3_compliance": ""
  },
  "method_performance": {
    "specificity": "",
    "linearity": "",
    "precision_rsd": "",
    "accuracy": "",
    "detection_limit": "",
    "quantitation_limit": ""
  },
  "overall_assessment": "",
  "pharmacopoeial_compliance": "",
  "recommendations": [],
  "corrective_actions_required": [],
  "report_conclusion": ""
}"""

MANUFACTURING_SYSTEM = """You are a cGMP manufacturing expert and regulatory compliance specialist with 20+ years in Schedule M and WHO-GMP certified pharmaceutical manufacturing.

Generate COMPLETE manufacturing documentation with:
- Batch Manufacturing Record (BMR) with all steps
- Batch Packaging Record (BPR)
- Standard Operating Procedures (SOPs) with step-by-step instructions
- Equipment qualification requirements (IQ/OQ/PQ)
- In-process quality control with acceptance criteria
- Deviation and OOS handling procedures
- Cleaning validation requirements
- Yield calculation and reconciliation

Return ONLY valid JSON:
{
  "product_details": {
    "product_name": "",
    "batch_number_format": "",
    "batch_size": "",
    "manufacturing_date": "",
    "expiry_date_calculation": "",
    "manufacturing_site": ""
  },
  "bmr": {
    "document_number": "",
    "version": "1.0",
    "approved_by": "",
    "raw_material_dispensing": [
      {
        "sr_no": 0,
        "material_name": "",
        "ar_number": "",
        "quantity_required": "",
        "quantity_dispensed": "",
        "dispensed_by": "",
        "checked_by": ""
      }
    ],
    "manufacturing_steps": [
      {
        "step_number": 0,
        "operation": "",
        "equipment": "",
        "equipment_id": "",
        "parameters": {
          "parameter_name": {"target": "", "range": "", "actual": ""}
        },
        "in_process_checks": [
          {"test": "", "specification": "", "result": "", "done_by": ""}
        ],
        "critical_control_point": false,
        "ccp_rationale": "",
        "time_started": "",
        "time_completed": "",
        "performed_by": "",
        "verified_by": ""
      }
    ],
    "yield_calculation": {
      "theoretical_yield": "",
      "actual_yield": "",
      "yield_percent": "",
      "acceptable_range": ""
    },
    "reconciliation": {
      "materials_used": "",
      "finished_goods": "",
      "rejected_material": "",
      "reconciliation_percent": "",
      "acceptable_range": ""
    }
  },
  "in_process_qc_plan": [
    {
      "stage": "",
      "test": "",
      "specification": "",
      "frequency": "",
      "action_if_ooc": ""
    }
  ],
  "sops": [
    {
      "sop_number": "",
      "title": "",
      "version": "1.0",
      "scope": "",
      "responsible_department": "",
      "procedure_summary": [],
      "critical_steps": [],
      "safety_precautions": []
    }
  ],
  "equipment_list": [
    {
      "equipment_name": "",
      "equipment_id": "",
      "capacity": "",
      "manufacturer": "",
      "qualification_status": "",
      "last_calibration": "",
      "next_calibration": "",
      "cleaning_procedure": ""
    }
  ],
  "cleaning_validation": {
    "cleaning_agent": "",
    "cleaning_procedure_sop": "",
    "swab_locations": [],
    "acceptance_criteria": "",
    "analytical_method": ""
  },
  "deviation_handling": {
    "planned_deviations": [],
    "unplanned_deviation_procedure": "",
    "oos_handling_procedure": ""
  },
  "quality_control_release": {
    "tests_required": [],
    "certificate_of_analysis_format": "",
    "release_criteria": ""
  },
  "manufacturing_notes": ""
}"""

COST_SYSTEM = """You are a pharmaceutical health economics expert with expertise in Indian pharmaceutical market pricing, cost-of-goods analysis, and financial modelling.

Generate a COMPREHENSIVE cost analysis with:
- Detailed raw material costs with Indian market prices (INR/kg)
- Manufacturing costs including labour, utilities, overhead
- Regulatory and compliance costs
- Three scenarios: optimistic, realistic, pessimistic
- Break-even analysis
- ROI calculation
- Pricing strategy recommendations
- Comparison with competitor pricing
- Export market pricing opportunities

Return ONLY valid JSON:
{
  "executive_summary": "",
  "cost_assumptions": {
    "batch_size": "",
    "location": "India",
    "labour_rate_inr_per_day": 0,
    "utility_cost_assumptions": "",
    "overhead_percent": 0
  },
  "raw_material_costs": {
    "items": [
      {
        "ingredient": "",
        "quantity_per_batch_kg": 0,
        "market_price_inr_per_kg": 0,
        "supplier_options": [],
        "cost_per_batch_inr": 0,
        "cost_per_unit_inr": 0,
        "price_volatility": "",
        "alternative_to_reduce_cost": ""
      }
    ],
    "total_rm_cost_per_batch_inr": 0,
    "total_rm_cost_per_unit_inr": 0
  },
  "manufacturing_costs": {
    "direct_labour": {"person_hours": 0, "cost_per_hour_inr": 0, "total_inr": 0},
    "utilities": {"electricity_kwh": 0, "water_kl": 0, "total_inr": 0},
    "consumables_inr": 0,
    "quality_control_inr": 0,
    "equipment_depreciation_inr": 0,
    "overhead_inr": 0,
    "total_manufacturing_cost_per_unit_inr": 0
  },
  "packaging_costs": {
    "primary_packaging_per_unit_inr": 0,
    "secondary_packaging_per_unit_inr": 0,
    "labelling_per_unit_inr": 0,
    "total_packaging_per_unit_inr": 0
  },
  "regulatory_compliance_costs": {
    "license_fee_inr": 0,
    "testing_per_batch_inr": 0,
    "stability_study_total_inr": 0,
    "amortised_per_unit_inr": 0
  },
  "cost_summary": {
    "scenarios": {
      "optimistic": {
        "total_cogs_per_unit_inr": 0,
        "assumptions": ""
      },
      "realistic": {
        "total_cogs_per_unit_inr": 0,
        "assumptions": ""
      },
      "pessimistic": {
        "total_cogs_per_unit_inr": 0,
        "assumptions": ""
      }
    }
  },
  "pricing_strategy": {
    "cost_plus_pricing": {
      "target_gross_margin_percent": 0,
      "recommended_trade_price_inr": 0,
      "recommended_mrp_inr": 0
    },
    "competitive_pricing": {
      "competitor_products": [
        {"product": "", "company": "", "mrp_inr": 0, "market_share_percent": 0}
      ],
      "recommended_positioning": ""
    },
    "value_based_pricing": {
      "patient_willingness_to_pay_inr": 0,
      "cost_effectiveness_analysis": ""
    }
  },
  "financial_projections": {
    "break_even_units": 0,
    "break_even_months": 0,
    "roi_percent_year_1": 0,
    "roi_percent_year_3": 0,
    "npv_5_year_inr_crores": 0
  },
  "export_opportunities": {
    "target_markets": [],
    "export_pricing_usd": 0,
    "regulatory_requirements_for_export": []
  },
  "cost_reduction_opportunities": [
    {
      "opportunity": "",
      "potential_saving_percent": 0,
      "implementation_timeline": "",
      "investment_required_inr": 0
    }
  ],
  "analyst_notes": ""
}"""

DOSSIER_SYSTEM = """You are a senior regulatory documentation specialist with 20+ years experience preparing CTD dossiers for CDSCO, AYUSH, FDA, and EMA submissions.

Generate a COMPLETE, SUBMISSION-READY dossier framework with:
- Full CTD structure (Modules 1-5) with India-specific requirements
- Detailed content for each section
- Specific data requirements with ICH guideline references
- Executive summary suitable for senior management review
- Gap analysis identifying what data is missing
- Submission timeline with critical path
- Estimated cost for dossier preparation

Return ONLY valid JSON:
{
  "dossier_overview": {
    "product_name": "",
    "inn_or_common_name": "",
    "dosage_form": "",
    "strength": "",
    "route": "",
    "applicant": "",
    "reference_member_state": "India (CDSCO/AYUSH)",
    "dossier_type": "",
    "application_form": "",
    "submission_fee_inr": 0
  },
  "executive_summary": "",
  "module_1_administrative": {
    "1_1_application_form": "",
    "1_2_product_information": "",
    "1_3_labelling": "",
    "1_4_information_about_experts": "",
    "completeness": ""
  },
  "module_2_summaries": {
    "2_3_quality_overall_summary": "",
    "2_4_nonclinical_overview": "",
    "2_5_clinical_overview": "",
    "2_6_nonclinical_written_summaries": "",
    "2_7_clinical_summary": ""
  },
  "module_3_quality": {
    "3_2_s_drug_substance": {
      "nomenclature": "",
      "structure": "",
      "general_properties": "",
      "manufacturer": "",
      "control_of_drug_substance": "",
      "reference_standards": "",
      "container_closure": "",
      "stability_summary": "",
      "data_available": [],
      "data_required": []
    },
    "3_2_p_drug_product": {
      "description_composition": "",
      "pharmaceutical_development": "",
      "manufacture": "",
      "control_of_excipients": "",
      "control_of_finished_product": "",
      "reference_standards": "",
      "container_closure_system": "",
      "stability_results_summary": "",
      "data_available": [],
      "data_required": []
    }
  },
  "module_4_nonclinical": {
    "4_2_1_pharmacology": "",
    "4_2_2_pharmacokinetics": "",
    "4_2_3_toxicology": "",
    "studies_completed": [],
    "studies_waived": [],
    "studies_required": [],
    "ich_guidelines_applicable": []
  },
  "module_5_clinical": {
    "5_2_pharmacology_studies": "",
    "5_3_clinical_study_reports": "",
    "studies_available": [
      {
        "study_id": "",
        "phase": "",
        "design": "",
        "n_patients": 0,
        "status": "",
        "key_results": ""
      }
    ],
    "literature_references": [],
    "benefit_risk_assessment": ""
  },
  "gap_analysis": {
    "completed_sections": [],
    "sections_in_progress": [],
    "sections_required": [],
    "critical_missing_data": [],
    "timeline_to_complete": ""
  },
  "submission_readiness": {
    "overall_readiness_percent": 0,
    "estimated_completion_timeline": "",
    "critical_path_items": [],
    "estimated_preparation_cost_inr_lakhs": 0
  },
  "india_specific_requirements": {
    "cdsco_specific_documents": [],
    "ayush_specific_requirements": [],
    "local_clinical_data_required": "",
    "bridging_study_requirement": ""
  },
  "consultant_recommendation": ""
}"""

RESEARCH_SYSTEM = """You are a senior academic researcher, biostatistician, and scientific writer with expertise in clinical trial design, preclinical research, and publication writing. Expert in ICMR guidelines, GCP, and journal publication standards.

Generate a COMPREHENSIVE research document with:
- Complete research proposal/protocol/publication as requested
- Proper statistical methodology with sample size justification
- CONSORT/PRISMA/STROBE compliance as applicable
- Ethical considerations per ICMR guidelines
- Complete reference list with actual published papers
- Budget breakdown in INR
- Timeline with Gantt chart description
- For publications: structured abstract, methods, results template, discussion points

Return ONLY valid JSON:
{
  "document_type": "",
  "title": "",
  "running_title": "",
  "keywords": [],
  "abstract": {
    "background": "",
    "objective": "",
    "methods": "",
    "expected_results": "",
    "conclusion": "",
    "word_count": 0
  },
  "background": {
    "introduction": "",
    "problem_statement": "",
    "current_knowledge": "",
    "gaps_in_knowledge": "",
    "rationale": "",
    "hypothesis": ""
  },
  "objectives": {
    "primary_objective": "",
    "secondary_objectives": [],
    "exploratory_objectives": []
  },
  "methodology": {
    "study_design": "",
    "study_design_justification": "",
    "study_setting": "",
    "study_duration": "",
    "study_population": {
      "inclusion_criteria": [],
      "exclusion_criteria": [],
      "withdrawal_criteria": []
    },
    "sample_size_calculation": {
      "primary_endpoint": "",
      "effect_size": "",
      "alpha": 0.05,
      "power": 0.80,
      "calculated_sample_size": 0,
      "attrition_allowance_percent": 0,
      "final_sample_size": 0,
      "software_used": "G*Power 3.1 / nQuery",
      "formula_used": "",
      "reference": ""
    },
    "randomisation": {
      "method": "",
      "allocation_ratio": "",
      "stratification_factors": [],
      "sequence_generation": "",
      "allocation_concealment": ""
    },
    "blinding": {
      "blinding_level": "",
      "who_is_blinded": [],
      "blinding_method": ""
    },
    "intervention": {
      "experimental_arm": "",
      "control_arm": "",
      "dose_justification": "",
      "duration": "",
      "compliance_monitoring": ""
    },
    "outcome_measures": {
      "primary_endpoint": {
        "outcome": "",
        "measurement_tool": "",
        "time_point": "",
        "clinically_meaningful_difference": ""
      },
      "secondary_endpoints": [],
      "safety_endpoints": [],
      "exploratory_endpoints": []
    },
    "data_collection": {
      "case_report_form": "",
      "data_collection_schedule": [],
      "source_data_verification": ""
    }
  },
  "statistical_analysis_plan": {
    "analysis_populations": {
      "itt_population": "",
      "pp_population": "",
      "safety_population": ""
    },
    "primary_analysis": {
      "statistical_test": "",
      "software": "SAS 9.4 / SPSS 26 / R 4.3",
      "covariates": [],
      "missing_data_handling": "",
      "multiplicity_correction": ""
    },
    "secondary_analyses": [],
    "subgroup_analyses": [],
    "interim_analysis": {
      "planned": false,
      "timing": "",
      "stopping_rules": ""
    },
    "sensitivity_analyses": []
  },
  "preclinical_validation": {
    "in_vitro_studies": [
      {
        "study_name": "",
        "model": "",
        "endpoint": "",
        "expected_result": "",
        "rationale": ""
      }
    ],
    "in_vivo_studies": [
      {
        "study_name": "",
        "animal_model": "",
        "strain": "",
        "n_per_group": 0,
        "dose_groups": [],
        "duration": "",
        "endpoints": [],
        "statistical_analysis": ""
      }
    ],
    "pk_pd_studies": []
  },
  "ethical_considerations": {
    "ethics_committee": "Institutional Ethics Committee (IEC) per ICMR Guidelines 2017",
    "ctri_registration": "Mandatory — Clinical Trials Registry India",
    "informed_consent_process": "",
    "patient_safety_monitoring": "",
    "data_safety_monitoring_board": "",
    "insurance_compensation": "",
    "vulnerable_population_considerations": ""
  },
  "timeline": [
    {
      "phase": "",
      "activities": [],
      "duration_months": 0,
      "start_month": 0,
      "end_month": 0,
      "milestones": []
    }
  ],
  "budget": {
    "personnel_inr": [
      {"role": "", "number": 0, "months": 0, "cost_inr": 0}
    ],
    "investigational_product_inr": 0,
    "investigations_lab_inr": 0,
    "regulatory_ethics_inr": 0,
    "equipment_inr": 0,
    "data_management_inr": 0,
    "publication_inr": 0,
    "overhead_percent": 0,
    "total_budget_inr_lakhs": 0,
    "funding_sources": []
  },
  "publication_plan": {
    "target_journal_primary": "",
    "target_journal_secondary": "",
    "impact_factor": 0,
    "journal_scope": "",
    "open_access_cost_usd": 0,
    "expected_publication_timeline": ""
  },
  "references": [
    {
      "number": 0,
      "authors": "",
      "title": "",
      "journal": "",
      "year": 0,
      "volume": "",
      "issue": "",
      "pages": "",
      "pmid": "",
      "doi": ""
    }
  ],
  "acknowledgements": "",
  "conflicts_of_interest": "None declared",
  "data_sharing_statement": ""
}"""

MODULE_PROMPTS = {
    'disease_intel': (DISEASE_SYSTEM, lambda d: f"""Conduct expert pharmaceutical disease intelligence analysis for:
Disease/Condition: {d.get('disease', '')}
Healthcare Problem: {d.get('healthcare_problem', '')}
Target Population: {d.get('target_population', '')}
Additional Context: {d.get('additional_context', '')}

Provide India-specific data, cite actual studies, include exact market figures in INR and USD, name specific drugs and competitors."""),

    'formulation_intel': (FORMULATION_SYSTEM, lambda d: f"""Design a complete pharmaceutical formulation for:
Disease/Indication: {d.get('disease', '')}
Product Type: {d.get('product_type', 'Pharmaceutical')}
Dosage Form: {d.get('dosage_form', 'Tablet')}
Batch Size: {d.get('batch_size', '100,000 units')}
Special Requirements: {d.get('additional_context', '')}
Preferred Ingredients: {d.get('active_ingredients', '')}

Provide complete batch formula with quantities, evaluation parameters with specifications, step-by-step manufacturing method, and cost estimate in INR."""),

    'literature_intel': (LITERATURE_SYSTEM, lambda d: f"""Conduct a comprehensive pharmaceutical literature review on:
Topic: {d.get('topic', '')}
Year Range: {d.get('year_range', '2019-2024')}
Context: {d.get('context', '')}

Include actual published studies with authors, journals, years, PMIDs. Provide evidence hierarchy, statistical findings, Indian research landscape, and research gaps with fundable study designs."""),

    'regulatory_intel': (REGULATORY_SYSTEM, lambda d: f"""Provide comprehensive regulatory intelligence for India submission:
Disease/Indication: {d.get('disease', '')}
Product Type: {d.get('product_type', '')}
Dosage Form: {d.get('dosage_form', '')}
Target Regulatory Authorities: {d.get('target_markets', ['CDSCO'])}
Additional Context: {d.get('additional_context', '')}

Include exact CDSCO form numbers, fees, specific Schedule references, current timelines, common deficiencies to avoid, and recommended CROs in India."""),

    'patent_intel': (PATENT_SYSTEM, lambda d: f"""Conduct comprehensive patent intelligence analysis for:
Invention Description: {d.get('invention_description', '')}
Disease/Therapeutic Area: {d.get('disease', '')}
Product Type: {d.get('product_type', '')}
Additional Context: {d.get('additional_context', '')}

Search Indian Patent Office, USPTO, EPO databases. Assess Section 3(d) risk for India. Provide filing strategy with cost estimates in INR."""),

    'stability_intel': (STABILITY_SYSTEM, lambda d: f"""Design a complete pharmaceutical stability programme for:
Product Name: {d.get('product_name', '')}
Dosage Form: {d.get('dosage_form', '')}
Active Ingredients: {d.get('active_ingredients', '')}
Storage Condition Proposed: {d.get('storage_condition', '25°C/60% RH')}
Packaging: {d.get('packaging', '')}

India is Zone IVb (30°C/65%RH). Provide complete ICH Q1A stability design, test parameters with specifications, degradation pathway analysis, and budget estimate in INR."""),

    'analytical_intel': (ANALYTICAL_SYSTEM, lambda d: f"""Interpret this pharmaceutical analytical data:
Analysis Type: {d.get('image_type', 'HPLC')}
Sample: {d.get('product_name', 'pharmaceutical sample')}
Analytical Context: {d.get('context', '')}
Expected Analytes: {d.get('expected_analytes', '')}

Provide detailed interpretation, identification with confidence levels, pharmacopoeial compliance assessment, method performance evaluation, and actionable recommendations."""),

    'manufacturing_intel': (MANUFACTURING_SYSTEM, lambda d: f"""Generate complete cGMP manufacturing documentation for:
Product Name: {d.get('product_name', '')}
Dosage Form: {d.get('dosage_form', 'Tablet')}
Batch Size: {d.get('batch_size', '100,000 units')}
Product Type: {d.get('product_type', 'Pharmaceutical')}
Active Ingredients: {d.get('active_ingredients', '')}

Provide complete BMR with all manufacturing steps, in-process QC plan, SOPs with procedure steps, equipment list with qualification status, and cleaning validation requirements. Comply with Schedule M and WHO-GMP."""),

    'cost_intel': (COST_SYSTEM, lambda d: f"""Generate comprehensive pharmaceutical cost analysis for Indian market:
Product Name: {d.get('product_name', '')}
Dosage Form: {d.get('dosage_form', 'Tablet')}
Batch Size: {d.get('batch_size', '100,000 units')}
Target Market: {d.get('target_market', 'India')}
Active Ingredients: {d.get('active_ingredients', '')}

Use current Indian market prices for raw materials. Include three scenarios, break-even analysis, competitor pricing comparison, export opportunity assessment, and cost reduction strategies."""),

    'dossier': (DOSSIER_SYSTEM, lambda d: f"""Prepare a complete CTD regulatory dossier framework for India (CDSCO/AYUSH):
Product Name: {d.get('product_name', '')}
Disease/Indication: {d.get('disease', '')}
Dosage Form: {d.get('dosage_form', '')}
Product Type: {d.get('product_type', '')}
Document Type: {d.get('doc_type', 'full_ctd')}

Include complete Module 1-5 content, India-specific requirements, gap analysis with critical missing data, submission timeline, and preparation cost estimate in INR lakhs."""),

    'research_asst': (RESEARCH_SYSTEM, lambda d: f"""Generate a comprehensive {d.get('doc_type', 'research_proposal')} for:
Title: {d.get('title', '')}
Disease/Research Area: {d.get('disease', '')}
Primary Objective: {d.get('objective', '')}
Background Context: {d.get('context', '')}

{'STATISTICAL VALIDATION PLAN: Include detailed pre-execution validation framework with: (1) Statistical models recommended for this study type, (2) Complete sample size calculation with formula, assumptions, and G*Power inputs, (3) Dosing rationale for preclinical and clinical phases with mg/kg to human dose conversion, (4) Sampling strategy (random/stratified/cluster) with justification, (5) Primary and secondary statistical tests with assumptions, (6) Interim analysis design if applicable, (7) Missing data handling strategy, (8) Sensitivity analysis plan, (9) Multiplicity correction if multiple endpoints, (10) Software recommendations.' if d.get('doc_type') == 'statistical_validation' else ''}

Include complete statistical analysis plan with sample size calculation using G*Power, preclinical validation studies design, ethical considerations per ICMR 2017 guidelines, CTRI registration requirements, detailed budget in INR with personnel costs, publication plan with target journals and impact factors, and complete reference list with actual published papers (authors, journal, year, PMID)."""),
}

# ── Fallback reports (used when no API key) ───────────────────────────────────
def fallback_report(module, data):
    disease = data.get('disease', data.get('topic', data.get('product_name', 'the specified condition')))
    return {
        "_demo_mode": True,
        "_message": f"This is a demo report for '{disease}'. Add your OPENAI_API_KEY to get real AI-generated analysis.",
        "disease_overview": {"name": disease, "icd_code": "See ICD-11", "description": f"AI analysis for {disease} will appear here once OpenAI API key is configured."},
        "opportunity_analysis": {"market_size_estimate": "AI will calculate", "opportunity_score": 0, "opportunity_rationale": "Add API key to enable"},
        "innovation_suggestions": [{"title": "Add API key to generate innovations", "approach": "Set OPENAI_API_KEY environment variable", "feasibility": "immediate"}],
        "note": "Set OPENAI_API_KEY in your environment to get real AI reports for any disease."
    }

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.route('/api/v1/auth/register/', methods=['POST'])
def register():
    d = request.get_json() or {}
    if not d.get('email') or not d.get('password'):
        return jsonify({'error': True, 'message': 'Email and password required'}), 400
    if q('SELECT id FROM users WHERE email=?', [d['email']], one=True):
        return jsonify({'error': True, 'message': 'Email already registered'}), 400
    if len(d['password']) < 8:
        return jsonify({'error': True, 'message': 'Password must be at least 8 characters'}), 400
    user = {'id': uid(), 'email': d['email'],
            'first_name': d.get('first_name',''), 'last_name': d.get('last_name',''),
            'role': d.get('role','researcher'), 'organisation': 'AIPBIOS Demo'}
    q('INSERT INTO users(id,email,password_hash,first_name,last_name,role,created_at) VALUES(?,?,?,?,?,?,?)',
      [user['id'], user['email'], h(d['password']),
       user['first_name'], user['last_name'], user['role'], now()])
    user['full_name'] = f"{user['first_name']} {user['last_name']}".strip()
    token = make_token(user)
    return jsonify({'message': 'Registration successful', 'user': user,
                    'tokens': {'access': token, 'refresh': token}}), 201

@app.route('/api/v1/auth/login/', methods=['POST'])
def login():
    d = request.get_json() or {}
    user = q('SELECT * FROM users WHERE email=? AND password_hash=?',
             [d.get('email',''), h(d.get('password',''))], one=True)
    if not user:
        return jsonify({'error': True, 'message': 'Invalid email or password'}), 401
    user['full_name'] = f"{user['first_name']} {user['last_name']}".strip()
    token = make_token(user)
    return jsonify({'access': token, 'refresh': token, 'user': user})

@app.route('/api/v1/auth/logout/', methods=['POST'])
@auth_required
def logout():
    return jsonify({'message': 'Logged out'})

@app.route('/api/v1/auth/refresh/', methods=['POST'])
def refresh():
    d = request.get_json() or {}
    try:
        data = jwt.decode(d.get('refresh',''), SECRET, algorithms=['HS256'])
        user = q('SELECT * FROM users WHERE id=?', [data['user_id']], one=True)
        if not user: return jsonify({'error': 'User not found'}), 401
        return jsonify({'access': make_token(user)})
    except Exception:
        return jsonify({'error': 'Invalid token'}), 401

@app.route('/api/v1/users/me/', methods=['GET','PATCH'])
@auth_required
def me():
    user = q('SELECT * FROM users WHERE id=?', [g.user['user_id']], one=True)
    if not user: return jsonify({'error': 'Not found'}), 404
    if request.method == 'PATCH':
        d = request.get_json() or {}
        for k in ['first_name','last_name']:
            if k in d: q(f'UPDATE users SET {k}=? WHERE id=?', [d[k], user['id']])
        user = q('SELECT * FROM users WHERE id=?', [g.user['user_id']], one=True)
    user['full_name'] = f"{user['first_name']} {user['last_name']}".strip()
    user.pop('password_hash', None)
    return jsonify(user)

@app.route('/api/v1/users/', methods=['GET'])
@auth_required
def list_users():
    search = request.args.get('search','')
    rows = q('SELECT * FROM users WHERE email LIKE ? OR first_name LIKE ?',
             [f'%{search}%', f'%{search}%']) if search else q('SELECT * FROM users')
    for u in rows:
        u.pop('password_hash', None)
        u['full_name'] = f"{u['first_name']} {u['last_name']}".strip()
        u['organisation_name'] = u.get('organisation','')
    return jsonify({'count': len(rows), 'results': rows})

@app.route('/api/v1/users/<uid_>/activate/',   methods=['POST'])
@auth_required
def activate_user(uid_):
    q('UPDATE users SET is_active=1 WHERE id=?', [uid_])
    return jsonify({'message': 'User activated'})

@app.route('/api/v1/users/<uid_>/deactivate/', methods=['POST'])
@auth_required
def deactivate_user(uid_):
    q('UPDATE users SET is_active=0 WHERE id=?', [uid_])
    return jsonify({'message': 'User deactivated'})

# ── PROJECT ROUTES ────────────────────────────────────────────────────────────
@app.route('/api/v1/projects/', methods=['GET','POST'])
@auth_required
def projects():
    uid_ = g.user['user_id']
    if request.method == 'POST':
        d = request.get_json() or {}
        if not d.get('title'):
            return jsonify({'error': True, 'message': 'Title required'}), 400
        pid = uid(); t = now()
        q('INSERT INTO projects(id,title,description,status,disease_area,product_type,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
          [pid, d['title'], d.get('description',''), 'active',
           d.get('disease_area',''), d.get('product_type',''), uid_, t, t])
        return jsonify({'id': pid, 'title': d['title'], 'status': 'active',
                        'disease_area': d.get('disease_area',''),
                        'product_type': d.get('product_type',''),
                        'job_count': 0, 'created_at': t, 'updated_at': t}), 201
    status_f = request.args.get('status','')
    rows = q('SELECT * FROM projects WHERE created_by=? AND status=? ORDER BY updated_at DESC', [uid_, status_f]) \
           if status_f and status_f != 'all' else \
           q('SELECT * FROM projects WHERE created_by=? ORDER BY updated_at DESC', [uid_])
    return jsonify({'count': len(rows), 'results': rows})

@app.route('/api/v1/projects/<pid>/', methods=['GET','PATCH'])
@auth_required
def project_detail(pid):
    p = q('SELECT * FROM projects WHERE id=?', [pid], one=True)
    if not p: return jsonify({'error': 'Not found'}), 404
    if request.method == 'PATCH':
        d = request.get_json() or {}
        for k in ['title','description','status','disease_area','product_type']:
            if k in d: q(f'UPDATE projects SET {k}=? WHERE id=?', [d[k], pid])
        q('UPDATE projects SET updated_at=? WHERE id=?', [now(), pid])
        p = q('SELECT * FROM projects WHERE id=?', [pid], one=True)
    return jsonify(p)

@app.route('/api/v1/projects/<pid>/archive/',  methods=['POST'])
@auth_required
def archive_project(pid):
    q('UPDATE projects SET status=?,updated_at=? WHERE id=?', ['archived', now(), pid])
    return jsonify({'message': 'Archived', 'status': 'archived'})

@app.route('/api/v1/projects/<pid>/activate/', methods=['POST'])
@auth_required
def activate_project(pid):
    q('UPDATE projects SET status=?,updated_at=? WHERE id=?', ['active', now(), pid])
    return jsonify({'message': 'Activated', 'status': 'active'})

@app.route('/api/v1/projects/<pid>/dashboard/', methods=['GET'])
@auth_required
def project_dashboard(pid):
    p = q('SELECT * FROM projects WHERE id=?', [pid], one=True)
    if not p: return jsonify({'error': 'Not found'}), 404
    jobs = q('SELECT module_type,status,tokens_used FROM intelligence_jobs WHERE project_id=?', [pid])
    by_mod = {}; by_stat = {}; total_tok = 0
    for j in jobs:
        by_mod[j['module_type']]  = by_mod.get(j['module_type'], 0) + 1
        by_stat[j['status']]      = by_stat.get(j['status'], 0) + 1
        total_tok += j.get('tokens_used', 0)
    recent = q('SELECT * FROM intelligence_jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 10', [pid])
    return jsonify({**p, 'job_by_module': by_mod, 'job_by_status': by_stat,
                    'total_tokens': total_tok, 'recent_jobs': recent, 'member_count': 1})

@app.route('/api/v1/projects/<pid>/jobs/',      methods=['GET'])
@auth_required
def project_jobs(pid):
    return jsonify(q('SELECT * FROM intelligence_jobs WHERE project_id=? ORDER BY created_at DESC', [pid]))

@app.route('/api/v1/projects/<pid>/documents/', methods=['GET'])
@auth_required
def project_documents(pid):
    return jsonify(q('SELECT * FROM documents WHERE project_id=? ORDER BY created_at DESC', [pid]))

@app.route('/api/v1/projects/jobs/<jid>/status/', methods=['GET'])
@auth_required
def job_status(jid):
    j = q('SELECT id,status,module_type,created_at,completed_at FROM intelligence_jobs WHERE id=?', [jid], one=True)
    return jsonify(j) if j else (jsonify({'error': 'Not found'}), 404)

@app.route('/api/v1/projects/jobs/<jid>/result/', methods=['GET'])
@auth_required
def job_result(jid):
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [jid], one=True)
    if not j: return jsonify({'error': 'Not found'}), 404
    j['output_payload'] = json.loads(j['output_payload'] or '{}')
    j['input_payload']  = json.loads(j['input_payload']  or '{}')
    return jsonify(j)

# ── INTELLIGENCE ROUTES ───────────────────────────────────────────────────────
def make_intel_route(module_name, url_name, action):
    @app.route(f'/api/v1/intelligence/{url_name}/{action}/',
               methods=['POST'], endpoint=f'intel_{module_name}')
    @auth_required
    def handler():
        d = request.get_json() or {}
        if request.content_type and 'multipart' in request.content_type:
            d = {k: request.form.get(k,'') for k in request.form}

        pid = d.get('project_id')
        if not pid:
            return jsonify({'error': True, 'message': 'project_id required'}), 400
        p = q('SELECT * FROM projects WHERE id=?', [pid], one=True)
        if not p:
            return jsonify({'error': True, 'message': 'Project not found'}), 404

        t = now()

        # Call OpenAI if key is set, else use fallback
        api_key = os.environ.get('OPENAI_API_KEY','')
        print(f"DEBUG: api_key present={bool(api_key)}, module={module_name}, disease={d.get('disease','')}")
        if api_key and module_name in MODULE_PROMPTS:
            try:
                system_prompt, user_prompt_fn = MODULE_PROMPTS[module_name]
                user_prompt = user_prompt_fn(d)
                print(f"DEBUG: Calling OpenAI for {module_name}")
                output, tokens = call_openai(system_prompt, user_prompt)
                print(f"DEBUG: OpenAI response: output_type={type(output)}, tokens={tokens}")
                if output is None:
                    print(f"DEBUG: OpenAI returned None, error: {tokens}")
                    output = fallback_report(module_name, d)
                    tokens = 0
            except Exception as e:
                import traceback
                print(f"DEBUG: Exception: {traceback.format_exc()}")
                output = fallback_report(module_name, d)
                tokens = 0
        else:
            print(f"DEBUG: Using fallback - api_key={bool(api_key)}, in_prompts={module_name in MODULE_PROMPTS}")
            output = fallback_report(module_name, d)
            tokens = 0

        jid = uid()
        q('INSERT INTO intelligence_jobs(id,project_id,created_by,module_type,status,input_payload,output_payload,tokens_used,created_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
          [jid, pid, g.user['user_id'], module_name, 'completed',
           json.dumps({k:v for k,v in d.items() if k != 'project_id'}),
           json.dumps(output), tokens, t, t])
        q('UPDATE projects SET job_count=job_count+1, updated_at=? WHERE id=?', [t, pid])

        return jsonify({
            'job_id':   jid,
            'status':   'completed',
            'message':  f'{module_name.replace("_"," ").title()} analysis complete.',
            'ai_used':  bool(os.environ.get('OPENAI_API_KEY','')),
            'poll_url': f'/api/v1/projects/jobs/{jid}/status/'
        }), 202

ROUTES = [
    ('disease_intel',     'disease',       'analyse'),
    ('formulation_intel', 'formulation',   'analyse'),
    ('literature_intel',  'literature',    'analyse'),
    ('regulatory_intel',  'regulatory',    'analyse'),
    ('patent_intel',      'patent',        'analyse'),
    ('stability_intel',   'stability',     'analyse'),
    ('analytical_intel',  'analytical',    'analyse'),
    ('manufacturing_intel','manufacturing','analyse'),
    ('cost_intel',        'cost',          'analyse'),
    ('dossier',           'dossier',       'build'),
    ('research_asst',     'research',      'generate'),
]
for mod, url, act in ROUTES:
    make_intel_route(mod, url, act)

@app.route('/api/v1/intelligence/<module>/reports/', methods=['GET'])
@auth_required
def intel_reports(module):
    mt = module.replace('-','_')
    if not mt.endswith('_intel') and mt not in ('dossier','research_asst'):
        mt += '_intel'
    rows = q('''SELECT ij.*, p.title as project_title
                FROM intelligence_jobs ij
                JOIN projects p ON ij.project_id=p.id
                WHERE ij.module_type=? AND ij.created_by=?
                ORDER BY ij.created_at DESC''',
             [mt, g.user['user_id']])
    for r in rows:
        r['output_payload'] = json.loads(r['output_payload'] or '{}')
        r['input_payload']  = json.loads(r['input_payload']  or '{}')
    return jsonify({'count': len(rows), 'results': rows})

@app.route('/api/v1/intelligence/<module>/reports/<rid>/', methods=['GET'])
@auth_required
def intel_report_detail(module, rid):
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [rid], one=True)
    if not j: return jsonify({'error': 'Not found'}), 404
    j['output_payload'] = json.loads(j['output_payload'] or '{}')
    j['input_payload']  = json.loads(j['input_payload']  or '{}')
    return jsonify(j)

# ── PDF GENERATION ────────────────────────────────────────────────────────────
def generate_pdf(output, title):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18,
                        textColor=colors.HexColor('#1a3a5c'), spaceAfter=8)
    H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13,
                        textColor=colors.HexColor('#2e6da4'), spaceAfter=6, spaceBefore=14)
    H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11,
                        textColor=colors.HexColor('#444'), spaceAfter=4, spaceBefore=8)
    BD = ParagraphStyle('BD', parent=styles['Normal'], fontSize=10, leading=15, spaceAfter=5)
    SM = ParagraphStyle('SM', parent=styles['Normal'], fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph('AIPBIOS Intelligence Platform', SM))
    story.append(Paragraph(title, H1))
    story.append(Paragraph(f"Generated: {datetime.datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}", SM))
    story.append(HRFlowable(width='100%', thickness=1.5,
                            color=colors.HexColor('#2e6da4'), spaceAfter=10))
    story.append(Spacer(1, 0.3*cm))

    def add_section(key, heading):
        val = output.get(key)
        if not val: return
        story.append(Paragraph(heading, H2))
        if isinstance(val, str):
            story.append(Paragraph(val, BD))
        elif isinstance(val, dict):
            for k, v in val.items():
                label = k.replace('_',' ').title()
                if isinstance(v, str) and v:
                    story.append(Paragraph(f"<b>{label}:</b> {v}", BD))
                elif isinstance(v, list) and v:
                    story.append(Paragraph(f"<b>{label}:</b>", BD))
                    for item in v:
                        story.append(Paragraph(f"• {item}" if isinstance(item, str) else f"• {json.dumps(item)}", BD))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    story.append(Paragraph(f"• {item}", BD))
                elif isinstance(item, dict):
                    for k, v in item.items():
                        story.append(Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", BD))
                    story.append(Spacer(1, 0.2*cm))
        story.append(Spacer(1, 0.2*cm))

    # Add all sections
    for key in output:
        if key.startswith('_'): continue
        heading = key.replace('_',' ').replace('-',' ').title()
        add_section(key, heading)

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        'Generated by AIPBIOS Intelligence Platform. '
        'For review purposes only. Verify with qualified professionals.',
        SM))

    doc.build(story)
    buf.seek(0)
    return buf.read()

@app.route('/api/v1/intelligence/dossier/reports/<rid>/download/<fmt>/', methods=['GET'])
@auth_required
def download_dossier(rid, fmt):
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [rid], one=True)
    if not j: return jsonify({'error': 'Not found'}), 404
    output = json.loads(j['output_payload'] or '{}')
    inp    = json.loads(j['input_payload']   or '{}')
    name   = output.get('dossier_overview', {}).get('product_name',
             inp.get('product_name', 'Dossier'))
    if fmt == 'pdf':
        pdf = generate_pdf(output, f"Regulatory Dossier — {name}")
        resp = make_response(pdf)
        resp.headers['Content-Type'] = 'application/pdf'
        resp.headers['Content-Disposition'] = f'attachment; filename="dossier_{name.replace(" ","_")}.pdf"'
        return resp
    elif fmt == 'json':
        resp = make_response(json.dumps(output, indent=2))
        resp.headers['Content-Type'] = 'application/json'
        resp.headers['Content-Disposition'] = f'attachment; filename="dossier_{rid}.json"'
        return resp
    return jsonify({'error': 'Use pdf or json'}), 400

@app.route('/api/v1/intelligence/research/documents/<rid>/download/', methods=['GET'])
@auth_required
def download_research(rid):
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [rid], one=True)
    if not j: return jsonify({'error': 'Not found'}), 404
    output = json.loads(j['output_payload'] or '{}')
    title  = output.get('title', 'Research Document')
    pdf    = generate_pdf(output, title)
    resp   = make_response(pdf)
    resp.headers['Content-Type'] = 'application/pdf'
    resp.headers['Content-Disposition'] = 'attachment; filename="research_document.pdf"'
    return resp

# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status':  'healthy',
        'service': 'AIPBIOS',
        'version': '2.0.0',
        'db':      'SQLite',
        'ai_mode': 'live' if os.environ.get('OPENAI_API_KEY','') else 'demo'
    })

@app.route('/api/status')
def api_status():
    key = os.environ.get('OPENAI_API_KEY','')
    return jsonify({
        'openai_connected': bool(key),
        'mode': 'Live AI — any disease supported' if key else 'Demo mode — add OPENAI_API_KEY for live AI'
    })

# ── STATIC / FRONTEND ─────────────────────────────────────────────────────────
EMBEDDED_HTML = None  # Loaded from index.html at startup

def load_html():
    global EMBEDDED_HTML
    for path in [
        os.path.join(BASE_DIR, 'static', 'index.html'),
        os.path.join(BASE_DIR, 'index.html'),
        'static/index.html',
        'index.html'
    ]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                EMBEDDED_HTML = f.read()
            print(f"HTML loaded from {path}: {len(EMBEDDED_HTML)} chars")
            return
    print("WARNING: index.html not found in any location")

load_html()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api/') or path.startswith('health'):
        return jsonify({'error': 'Not found'}), 404
    static_dir = os.path.join(BASE_DIR, 'static')
    if path and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    if os.path.exists(os.path.join(static_dir, 'index.html')):
        return send_from_directory(static_dir, 'index.html')
    for p in [os.path.join(BASE_DIR,'index.html'),'index.html']:
        if os.path.exists(p):
            return open(p,'r',encoding='utf-8').read(), 200, {'Content-Type':'text/html; charset=utf-8'}
    return EMBEDDED_HTML, 200, {'Content-Type':'text/html; charset=utf-8'}

# ── SEED DATA ─────────────────────────────────────────────────────────────────
def seed():
    db = sqlite3.connect(DB_PATH)
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        db.close(); return
    t = now()
    USERS = [
        (uid(),'admin@aipbios.com',      h('Admin@12345'),    'Platform','Admin',    'superadmin'),
        (uid(),'researcher@aipbios.com', h('Research@12345'), 'Dr. Priya','Sharma',  'researcher'),
        (uid(),'student@aipbios.com',    h('Student@12345'),  'Rahul',   'Verma',    'student'),
        (uid(),'industry@aipbios.com',   h('Industry@12345'), 'Meera',   'Patel',    'industry_user'),
    ]
    for u in USERS:
        db.execute('INSERT INTO users(id,email,password_hash,first_name,last_name,role,created_at) VALUES(?,?,?,?,?,?,?)', (*u, t))
    res_id = USERS[1][0]
    PROJS = [
        (uid(),'GlycoHerb-DS: Herbal Anti-Diabetic Tablet','Disease=Diabetes, Product=Herbal Ayurvedic — demo project with pre-loaded reports','Type 2 Diabetes Mellitus','Ayurvedic'),
        (uid(),'Diclofenac 1% Topical Gel — CDSCO Filing','Allopathic NSAID gel for musculoskeletal pain','Musculoskeletal Pain','Allopathic'),
        (uid(),'AshwaRelax Sleep Support Capsule','Nutraceutical: Ashwagandha + L-Theanine','Sleep Disorders','Nutraceutical'),
    ]
    for p in PROJS:
        db.execute('INSERT INTO projects(id,title,description,status,disease_area,product_type,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                   (p[0],p[1],p[2],'active',p[3],p[4],res_id,t,t))
    main_pid = PROJS[0][0]

    # Pre-load demo reports for the first project
    try:
        from server_demo_data import DEMO_REPORTS
    except Exception as e:
        print(f'Demo data import failed: {e}'); db.commit(); db.close(); return
    for mod, inp, out, tok in DEMO_REPORTS(main_pid, res_id, t):
        db.execute('INSERT INTO intelligence_jobs(id,project_id,created_by,module_type,status,input_payload,output_payload,tokens_used,created_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (uid(),main_pid,res_id,mod,'completed',json.dumps(inp),json.dumps(out),tok,t,t))
        db.execute('UPDATE projects SET job_count=job_count+1 WHERE id=?',[main_pid])
    db.commit(); db.close()
    print("✓ Demo data seeded")

# Initialize on startup - wrapped in try/except to prevent hanging
try:
    init_db()
    print("Database initialized")
except Exception as e:
    print(f"DB init error: {e}")

try:
    with app.app_context():
        seed()
    print("Seed complete")
except Exception as e:
    print(f"Seed error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    mode = 'LIVE AI MODE' if os.environ.get('OPENAI_API_KEY','') else 'DEMO MODE'
    print(f"AIPBIOS starting — {mode} on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
