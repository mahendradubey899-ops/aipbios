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
DISEASE_SYSTEM = """You are a senior pharmaceutical and healthcare intelligence analyst.
Analyse the disease/condition and return ONLY a valid JSON object with this exact structure:
{
  "disease_overview": {"name": "", "icd_code": "", "category": "", "description": ""},
  "epidemiology": {"global_prevalence": "", "incidence_rate": "", "mortality_rate": "", "geographic_hotspots": [], "high_risk_groups": []},
  "unmet_needs": {"clinical_gaps": [], "patient_burden": "", "economic_burden": ""},
  "opportunity_analysis": {"market_size_estimate": "", "growth_rate_cagr": "", "opportunity_score": 0, "opportunity_rationale": "", "key_value_drivers": []},
  "innovation_suggestions": [{"title": "", "approach": "", "rationale": "", "novelty_level": "", "feasibility": "", "time_to_market_years": 0}],
  "regulatory_considerations": {"designation_opportunities": [], "key_regulatory_agencies": [], "special_considerations": ""},
  "research_priorities": [],
  "confidence_level": "",
  "analyst_notes": ""
}"""

FORMULATION_SYSTEM = """You are a Principal Formulation Scientist specialising in pharmaceutical and Ayurvedic product development.
Return ONLY a valid JSON object with this exact structure:
{
  "product_overview": {"proposed_name": "", "category": "", "route_of_administration": "", "target_patient_profile": ""},
  "active_ingredients": [{"name": "", "pharmacological_class": "", "mechanism_of_action": "", "dose_per_unit": "", "role_in_formulation": ""}],
  "herbal_ingredients": [{"common_name": "", "botanical_name": "", "plant_part_used": "", "quantity_per_unit": "", "therapeutic_action": "", "traditional_use": ""}],
  "excipients": [{"name": "", "category": "", "quantity_per_unit": "", "function": "", "grade": ""}],
  "batch_formula": {"batch_size": "", "theoretical_yield": "", "components": [{"ingredient": "", "quantity_per_unit": "", "quantity_per_batch": ""}], "total_weight_per_unit": ""},
  "manufacturing_method": {"primary_process": "", "critical_steps": [], "equipment_required": [], "in_process_controls": []},
  "packaging_recommendation": {"primary_pack": "", "storage_conditions": "", "shelf_life_estimate": ""},
  "formulation_notes": ""
}"""

LITERATURE_SYSTEM = """You are a senior medical research analyst and systematic review expert.
Return ONLY a valid JSON object with this exact structure:
{
  "topic_overview": {"standardised_topic": "", "mesh_terms": [], "field_maturity": ""},
  "literature_review": {"summary": "", "total_studies_estimated": "", "key_findings": [], "consensus_areas": [], "contested_areas": []},
  "key_studies": [{"title": "", "authors": "", "year": "", "journal": "", "study_type": "", "key_finding": "", "evidence_level": ""}],
  "research_gaps": [{"gap": "", "rationale": "", "priority": "", "suggested_study_design": ""}],
  "research_summary": {"state_of_evidence": "", "strength_of_evidence": "", "future_directions": []},
  "pubmed_search_strategy": {"recommended_query": "", "databases": []},
  "analyst_notes": ""
}"""

REGULATORY_SYSTEM = """You are a senior regulatory affairs specialist with expertise in CDSCO, AYUSH, FSSAI, FDA, and EMA.
Return ONLY a valid JSON object with this exact structure:
{
  "product_classification": {"category": "", "drug_type": "", "schedule_india": ""},
  "regulatory_pathways": [{"authority": "", "pathway": "", "estimated_timeline_months": 0, "estimated_cost_usd": "", "key_requirements": []}],
  "required_documents": {"administrative": [], "quality_cmc": [], "nonclinical": [], "clinical": [], "labelling": []},
  "compliance_checklist": [{"item": "", "category": "", "authority": "", "mandatory": true, "guidance_reference": ""}],
  "timeline_overview": {"total_estimated_months": 0, "key_milestones": []},
  "regulatory_risks": [{"risk": "", "mitigation": "", "probability": ""}],
  "consultant_notes": ""
}"""

DOSSIER_SYSTEM = """You are a senior regulatory documentation specialist and CTD expert.
Return ONLY a valid JSON object with this exact structure:
{
  "dossier_overview": {"product_name": "", "inn_or_common_name": "", "dosage_form": "", "route": "", "dossier_type": ""},
  "executive_summary": "",
  "module_2_summaries": {"quality_overall_summary": "", "nonclinical_overview": "", "clinical_overview": ""},
  "module_3_quality": {"drug_substance_summary": "", "drug_product_summary": ""},
  "module_4_nonclinical": {"pharmacology_summary": "", "toxicology_summary": "", "studies_required": []},
  "module_5_clinical": {"efficacy_summary": "", "safety_summary": "", "benefit_risk_assessment": ""},
  "submission_readiness": {"completed_sections": [], "sections_requiring_data": [], "estimated_completion_timeline": "", "critical_path_items": []}
}"""

PATENT_SYSTEM = """You are a senior patent analyst and IP strategist specialising in pharmaceuticals.
Return ONLY a valid JSON object with this exact structure:
{
  "invention_analysis": {"title_suggested": "", "technology_area": "", "novelty_elements": []},
  "prior_art_review": {"closest_prior_art": [{"patent_number": "", "title": "", "assignee": "", "year": "", "relevance": ""}], "freedom_to_operate_preliminary": ""},
  "patentability_assessment": {"novelty_score": 0, "inventive_step_score": 0, "overall_patentability_score": 0, "patentability_opinion": "", "rationale": ""},
  "filing_strategy": {"recommended_jurisdictions": [], "pct_recommended": true, "estimated_cost_range_usd": ""},
  "analyst_notes": ""
}"""

STABILITY_SYSTEM = """You are a pharmaceutical stability expert with deep knowledge of ICH Q1A-Q1F guidelines.
Return ONLY a valid JSON object with this exact structure:
{
  "stability_plan": {"study_types": [{"type": "", "conditions": "", "duration_months": 0, "test_intervals": []}], "test_parameters": [], "acceptance_criteria": [{"parameter": "", "limit": ""}]},
  "degradation_pathways": [{"pathway": "", "risk_level": "", "mitigation": ""}],
  "packaging_recommendation": {"primary_packaging": "", "secondary_packaging": "", "desiccant_required": true, "rationale": ""},
  "shelf_life_assessment": {"predicted_shelf_life": "", "basis_for_prediction": "", "label_claim_recommended": ""},
  "regulatory_requirements": {"ich_guidelines": [], "submission_requirements": ""},
  "stability_notes": ""
}"""

MANUFACTURING_SYSTEM = """You are a cGMP manufacturing expert specialising in pharmaceutical documentation.
Return ONLY a valid JSON object with this exact structure:
{
  "product_details": {"name": "", "dosage_form": "", "batch_size": ""},
  "bmr": {"document_number": "", "manufacturing_steps": [{"step_number": 0, "operation": "", "equipment": "", "parameters": {}, "in_process_checks": [], "critical_control_point": true}], "yield_calculation": ""},
  "bpr": {"document_number": "", "sections": [{"section": "", "data_to_record": [], "acceptance_criteria": ""}]},
  "sops": [{"sop_number": "", "title": "", "scope": "", "responsible_department": ""}],
  "equipment_list": [{"equipment": "", "capacity": "", "qualification_status": "", "calibration_frequency": ""}],
  "quality_control_plan": {"raw_material_testing": [], "in_process_testing": [], "finished_product_testing": []},
  "manufacturing_notes": ""
}"""

COST_SYSTEM = """You are a pharmaceutical health economics and cost analysis expert.
Return ONLY a valid JSON object with this exact structure:
{
  "cost_summary": {"total_cost_per_unit_usd": {"low": 0, "mid": 0, "high": 0}, "gross_margin_percent": {"low": 0, "mid": 0, "high": 0}, "break_even_units": 0},
  "raw_material_cost": {"total_per_batch_usd": {"low": 0, "mid": 0, "high": 0}, "items": [{"ingredient": "", "quantity_per_batch": "", "unit_cost_usd": {"low": 0, "mid": 0, "high": 0}}]},
  "manufacturing_cost": {"total_per_batch_usd": {"low": 0, "mid": 0, "high": 0}, "direct_labour": {"cost_usd": 0, "person_hours": 0}, "overhead_usd": 0},
  "packaging_cost": {"total_per_batch_usd": {"low": 0, "mid": 0, "high": 0}},
  "cost_reduction_opportunities": [{"opportunity": "", "potential_saving_percent": 0, "implementation": ""}],
  "analyst_notes": ""
}"""

RESEARCH_SYSTEM = """You are a senior academic researcher and scientific writer.
Return ONLY a valid JSON object with this exact structure:
{
  "document_type": "",
  "title": "",
  "abstract": "",
  "background": "",
  "problem_statement": "",
  "objectives": {"primary_objective": "", "secondary_objectives": []},
  "methodology": {"study_design": "", "study_population": "", "sample_size_justification": "", "intervention_or_exposure": "", "outcome_measures": [], "statistical_plan": ""},
  "timeline": [{"phase": "", "duration": "", "activities": []}],
  "budget_outline": [{"item": "", "estimated_cost_inr": 0}],
  "ethical_considerations": "",
  "expected_outcomes": [],
  "significance": "",
  "references": [],
  "keywords": []
}"""

MODULE_PROMPTS = {
    'disease_intel':      (DISEASE_SYSTEM,       lambda d: f"Analyse this disease intelligence request:\nDisease: {d.get('disease','')}\nHealthcare Problem: {d.get('healthcare_problem','')}\nTarget Population: {d.get('target_population','')}"),
    'formulation_intel':  (FORMULATION_SYSTEM,   lambda d: f"Design a formulation for:\nDisease: {d.get('disease','')}\nProduct Type: {d.get('product_type','')}\nDosage Form: {d.get('dosage_form','')}\nContext: {d.get('additional_context','')}"),
    'literature_intel':   (LITERATURE_SYSTEM,    lambda d: f"Conduct a literature review on:\nTopic: {d.get('topic','')}\nYear Range: {d.get('year_range','2019-2024')}\nContext: {d.get('context','')}"),
    'regulatory_intel':   (REGULATORY_SYSTEM,    lambda d: f"Provide regulatory guidance for:\nDisease: {d.get('disease','')}\nProduct Type: {d.get('product_type','')}\nDosage Form: {d.get('dosage_form','')}\nTarget Markets: {d.get('target_markets',[])}"),
    'patent_intel':       (PATENT_SYSTEM,        lambda d: f"Assess patentability:\nInvention: {d.get('invention_description','')}\nDisease: {d.get('disease','')}\nProduct Type: {d.get('product_type','')}"),
    'stability_intel':    (STABILITY_SYSTEM,     lambda d: f"Design stability programme for:\nProduct: {d.get('product_name','')}\nDosage Form: {d.get('dosage_form','')}\nActive Ingredients: {d.get('active_ingredients','')}\nStorage: {d.get('storage_condition','')}"),
    'analytical_intel':   (MANUFACTURING_SYSTEM, lambda d: f"Provide analytical interpretation for {d.get('image_type','HPLC')} analysis.\nContext: {d.get('context','')}"),
    'manufacturing_intel':(MANUFACTURING_SYSTEM, lambda d: f"Generate manufacturing documentation for:\nProduct: {d.get('product_name','')}\nDosage Form: {d.get('dosage_form','')}\nBatch Size: {d.get('batch_size','')}\nProduct Type: {d.get('product_type','')}"),
    'cost_intel':         (COST_SYSTEM,          lambda d: f"Generate cost analysis for:\nProduct: {d.get('product_name','')}\nDosage Form: {d.get('dosage_form','')}\nBatch Size: {d.get('batch_size','')}\nMarket: {d.get('target_market','India')}"),
    'dossier':            (DOSSIER_SYSTEM,       lambda d: f"Build regulatory dossier for:\nProduct: {d.get('product_name','')}\nDisease: {d.get('disease','')}\nDosage Form: {d.get('dosage_form','')}\nProduct Type: {d.get('product_type','')}"),
    'research_asst':      (RESEARCH_SYSTEM,      lambda d: f"Generate {d.get('doc_type','research_proposal')} for:\nTitle: {d.get('title','')}\nDisease: {d.get('disease','')}\nObjective: {d.get('objective','')}\nContext: {d.get('context','')}"),
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
        if api_key and module_name in MODULE_PROMPTS:
            try:
                system_prompt, user_prompt_fn = MODULE_PROMPTS[module_name]
                user_prompt = user_prompt_fn(d)
                output, tokens = call_openai(system_prompt, user_prompt)
                if output is None:
                    print(f"OpenAI returned None: {tokens}")
                    output = fallback_report(module_name, d)
                    tokens = 0
            except Exception as e:
                print(f"OpenAI call exception: {e}")
                output = fallback_report(module_name, d)
                tokens = 0
        else:
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
EMBEDDED_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>AIPBIOS — AI Intelligence Platform</title>\n<style>\n*{box-sizing:border-box;margin:0;padding:0}\n:root{\n  --brand:#2563eb;--brand-dark:#1d4ed8;--brand-light:#eff6ff;\n  --teal:#0d9488;--amber:#d97706;--rose:#e11d48;--purple:#7c3aed;\n  --green:#059669;--slate:#475569;--pink:#db2777;--orange:#ea580c;\n  --bg:#f8fafc;--card:#fff;--border:#e2e8f0;--text:#0f172a;--muted:#64748b;--faint:#94a3b8;\n  --sidebar:260px;--radius:10px;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);\n}\nbody{font-family:\'Inter\',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}\n#app{display:flex;min-height:100vh}\n\n/* Auth */\n#auth-screen{display:none;align-items:center;justify-content:center;min-height:100vh;\n  background:linear-gradient(135deg,#0f172a,#1e3a5f,#0d4f47);flex-direction:column;gap:20px}\n.auth-card{background:white;border-radius:16px;padding:40px;width:420px;box-shadow:0 20px 60px rgba(0,0,0,.3)}\n.auth-logo{text-align:center;margin-bottom:28px}\n.auth-logo .dna{width:56px;height:56px;background:linear-gradient(135deg,#2563eb,#0d9488);\n  border-radius:14px;display:inline-flex;align-items:center;justify-content:center;font-size:26px;margin-bottom:10px}\n.auth-logo h1{font-size:22px;font-weight:700;color:#0f172a}\n.auth-logo p{font-size:12px;color:#64748b;margin-top:2px}\n.auth-tabs{display:flex;gap:0;margin-bottom:24px;border:1px solid var(--border);border-radius:8px;overflow:hidden}\n.auth-tab{flex:1;padding:9px;text-align:center;cursor:pointer;font-size:13px;font-weight:500;\n  background:white;color:var(--muted);border:none;transition:.15s}\n.auth-tab.active{background:var(--brand);color:white}\n\n/* Sidebar */\n#sidebar{width:var(--sidebar);background:white;border-right:1px solid var(--border);\n  display:none;flex-direction:column;position:fixed;inset:0 auto 0 0;z-index:30;overflow-y:auto}\n.logo{display:flex;align-items:center;gap:10px;padding:16px 20px;border-bottom:1px solid var(--border);height:60px}\n.logo-icon{width:32px;height:32px;background:linear-gradient(135deg,#2563eb,#0d9488);border-radius:8px;\n  display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}\n.logo-text{font-size:13px;font-weight:700;color:var(--text)}\n.logo-sub{font-size:10px;color:var(--faint)}\nnav{flex:1;padding:12px 10px;overflow-y:auto}\n.nav-section{font-size:10px;font-weight:600;color:var(--faint);text-transform:uppercase;\n  letter-spacing:.06em;padding:12px 10px 6px}\n.nav-item{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;cursor:pointer;\n  font-size:12px;font-weight:500;color:var(--muted);transition:.15s;text-decoration:none}\n.nav-item:hover{background:#f1f5f9;color:var(--text)}\n.nav-item.active{background:var(--brand-light);color:var(--brand)}\n.nav-item .icon{font-size:15px;width:20px;text-align:center;flex-shrink:0}\n.user-footer{border-top:1px solid var(--border);padding:12px}\n.user-info{display:flex;align-items:center;gap:10px;padding:8px;border-radius:8px;margin-bottom:4px}\n.avatar{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#2563eb,#0d9488);\n  color:white;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;flex-shrink:0}\n\n/* Main */\n#main{margin-left:var(--sidebar);flex:1;display:none;flex-direction:column}\n.topbar{height:60px;background:white;border-bottom:1px solid var(--border);\n  display:flex;align-items:center;justify-content:space-between;padding:0 24px;position:sticky;top:0;z-index:20}\n.topbar-title{font-size:15px;font-weight:600}\n.topbar-sub{font-size:12px;color:var(--muted);margin-top:1px}\n.content{flex:1;padding:24px;max-width:1200px;width:100%}\n\n/* Cards */\n.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow)}\n.card-hover{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);\n  box-shadow:var(--shadow);cursor:pointer;transition:box-shadow .2s}\n.card-hover:hover{box-shadow:0 4px 12px rgba(0,0,0,.1)}\n\n/* Forms */\ninput,select,textarea{width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:8px;\n  font-size:13px;font-family:inherit;outline:none;background:white;color:var(--text)}\ninput:focus,select:focus,textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(37,99,235,.1)}\nlabel{display:block;font-size:12px;font-weight:500;color:var(--muted);margin-bottom:6px}\n.form-group{margin-bottom:14px}\n.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}\n\n/* Buttons */\n.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:8px;\n  font-size:13px;font-weight:500;cursor:pointer;border:none;transition:.15s;font-family:inherit}\n.btn-primary{background:var(--brand);color:white}.btn-primary:hover{background:var(--brand-dark)}\n.btn-primary:disabled{opacity:.5;cursor:not-allowed}\n.btn-secondary{background:white;color:var(--text);border:1px solid var(--border)}.btn-secondary:hover{background:#f8fafc}\n.btn-danger{background:#ef4444;color:white}.btn-danger:hover{background:#dc2626}\n.btn-ghost{background:transparent;color:var(--muted);padding:6px 10px}.btn-ghost:hover{background:#f1f5f9;color:var(--text)}\n.btn-sm{padding:5px 10px;font-size:12px}\n.btn-full{width:100%;justify-content:center;padding:10px}\n\n/* Badges */\n.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:99px;\n  font-size:11px;font-weight:500;border:1px solid transparent}\n.badge-green{background:#f0fdf4;color:#15803d;border-color:#bbf7d0}\n.badge-blue{background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe}\n.badge-amber{background:#fffbeb;color:#b45309;border-color:#fde68a}\n.badge-red{background:#fef2f2;color:#b91c1c;border-color:#fecaca}\n.badge-purple{background:#faf5ff;color:#6d28d9;border-color:#e9d5ff}\n.badge-gray{background:#f8fafc;color:#64748b;border-color:#e2e8f0}\n\n/* Status */\n.status-completed{background:#f0fdf4;color:#15803d;border-color:#bbf7d0}\n.status-pending,.status-processing{background:#faf5ff;color:#6d28d9;border-color:#e9d5ff}\n.status-failed{background:#fef2f2;color:#b91c1c;border-color:#fecaca}\n\n/* Toast */\n#toast{position:fixed;top:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px}\n.toast{padding:12px 18px;border-radius:10px;font-size:13px;font-weight:500;box-shadow:0 8px 24px rgba(0,0,0,.15);\n  animation:slideIn .2s ease;min-width:220px}\n.toast-success{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0}\n.toast-error{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca}\n.toast-info{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe}\n@keyframes slideIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}\n\n/* Module header banner */\n.module-banner{display:flex;align-items:center;gap:14px;padding:16px 20px;border-radius:12px;\n  border:1px solid;margin-bottom:24px}\n\n/* Report viewer */\n.report-viewer{background:#f8fafc;border:1px solid var(--border);border-radius:10px;overflow:hidden}\n.report-header{padding:10px 16px;background:white;border-bottom:1px solid var(--border);\n  display:flex;align-items:center;justify-content:space-between}\n.report-body{padding:16px;overflow-y:auto;max-height:500px;font-size:12px}\n.rk{color:var(--brand);font-weight:600}.rv{color:var(--text)}\n.rs{color:#059669}.rn{color:var(--amber)}\n.r-section{margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:10px}\n.r-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;\n  letter-spacing:.04em;margin-bottom:8px}\n.r-row{display:flex;gap:8px;margin-bottom:4px;align-items:flex-start}\n.r-key{color:var(--brand);font-weight:500;min-width:180px;flex-shrink:0}\n.r-val{color:var(--text)}\n.r-list-item{padding-left:16px;margin-bottom:2px;color:var(--text)}\n.r-list-item::before{content:"•";color:var(--brand);margin-right:6px}\n.score-badge{display:inline-flex;align-items:center;justify-content:center;\n  width:36px;height:36px;border-radius:50%;font-size:15px;font-weight:700;\n  background:var(--brand);color:white}\n\n/* Job status card */\n.job-card{border:1px solid;border-radius:10px;padding:16px;display:flex;align-items:flex-start;gap:14px}\n.job-card.completed{border-color:#bbf7d0;background:#f0fdf4}\n.job-card.processing,.job-card.pending{border-color:#e9d5ff;background:#faf5ff}\n.job-card.failed{border-color:#fecaca;background:#fef2f2}\n.job-icon{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:18px}\n.spin{animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}\n.progress-bar{height:4px;background:#e9d5ff;border-radius:2px;overflow:hidden;margin-top:10px}\n.progress-fill{height:100%;background:var(--purple);border-radius:2px;animation:pulse 1.5s ease infinite}\n@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}\n\n/* Dashboard stats */\n.stat-card{padding:20px;border-radius:var(--radius);background:white;border:1px solid var(--border)}\n.stat-label{font-size:12px;font-weight:500;color:var(--muted)}\n.stat-value{font-size:28px;font-weight:700;color:var(--text);margin-top:4px}\n.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}\n\n/* Project grid */\n.projects-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}\n.project-card{padding:18px}\n.project-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px;line-height:1.3}\n.project-meta{font-size:11px;color:var(--faint)}\n.project-tags{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}\n.tag{padding:2px 8px;border-radius:6px;font-size:11px;font-weight:500}\n.tag-rose{background:#fff1f2;color:#e11d48}.tag-teal{background:#f0fdfa;color:#0d9488}\n.tag-blue{background:#eff6ff;color:#1d4ed8}.tag-amber{background:#fffbeb;color:#b45309}\n\n/* Modal */\n.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:50;\n  align-items:center;justify-content:center}\n.modal-overlay.open{display:flex}\n.modal{background:white;border-radius:16px;padding:28px;width:500px;max-width:95vw;\n  box-shadow:0 20px 60px rgba(0,0,0,.2);animation:slideUp .25s ease}\n@keyframes slideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}\n.modal-title{font-size:16px;font-weight:600;margin-bottom:20px;\n  display:flex;align-items:center;justify-content:space-between}\n\n/* Table */\ntable{width:100%;border-collapse:collapse}\nth{background:#f8fafc;padding:8px 14px;text-align:left;font-size:11px;font-weight:600;\n  color:var(--muted);text-transform:uppercase;letter-spacing:.04em}\ntd{padding:10px 14px;border-top:1px solid var(--border);font-size:12px;vertical-align:middle}\ntr:hover td{background:#fafafa}\n\n/* Misc */\n.flex{display:flex}.items-center{align-items:center}.justify-between{justify-content:space-between}\n.gap-2{gap:8px}.gap-3{gap:12px}.mb-4{margin-bottom:16px}.mt-2{margin-top:8px}.mt-3{margin-top:12px}\n.text-sm{font-size:13px}.text-xs{font-size:11px}.text-muted{color:var(--muted)}\n.font-medium{font-weight:500}.font-semibold{font-weight:600}\n.truncate{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n.empty-state{text-align:center;padding:48px 20px;color:var(--muted)}\n.empty-icon{font-size:36px;margin-bottom:10px}\n.chip-group{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}\n.chip{padding:4px 10px;border-radius:99px;font-size:11px;font-weight:500;cursor:pointer;\n  border:1px solid var(--border);background:white;color:var(--muted);transition:.15s}\n.chip.active{background:var(--brand);color:white;border-color:var(--brand)}\n.section-title{font-size:14px;font-weight:600;color:var(--text);margin-bottom:14px}\nhr{border:none;border-top:1px solid var(--border);margin:16px 0}\n.intel-2col{display:grid;grid-template-columns:420px 1fr;gap:20px;align-items:start}\n@media(max-width:900px){.intel-2col{grid-template-columns:1fr}.projects-grid{grid-template-columns:1fr 1fr}.stats-grid{grid-template-columns:1fr 1fr}}\n</style>\n</head>\n<body>\n<div id="toast"></div>\n\n<!-- Auth Screen -->\n<div id="auth-screen">\n  <div class="auth-card">\n    <div class="auth-logo">\n      <div class="dna">🧬</div>\n      <h1>AIPBIOS</h1>\n      <p>AI Intelligence Platform for Healthcare & Pharma</p>\n    </div>\n    <div class="auth-tabs">\n      <button class="auth-tab active" onclick="switchAuthTab(\'login\')">Sign in</button>\n      <button class="auth-tab" onclick="switchAuthTab(\'register\')">Create account</button>\n    </div>\n    <!-- Login -->\n    <div id="login-form">\n      <div id="login-error" style="display:none;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;padding:10px;border-radius:8px;font-size:13px;margin-bottom:12px"></div>\n      <div class="form-group"><label>Email</label><input type="email" id="login-email" placeholder="researcher@aipbios.com" autocomplete="email"></div>\n      <div class="form-group"><label>Password</label><input type="password" id="login-password" placeholder="••••••••" onkeydown="if(event.key===\'Enter\')doLogin()" autocomplete="current-password"></div>\n      <button class="btn btn-primary btn-full mt-3" id="login-btn" onclick="doLogin()">Sign in</button>\n      <div style="margin-top:14px;padding:12px;background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;font-size:12px">\n        <div style="font-weight:600;color:#0369a1;margin-bottom:6px">⚡ Quick demo — click to fill:</div>\n        <button onclick="fillLogin(\'researcher@aipbios.com\',\'Research@12345\')" style="background:#e0f2fe;border:1px solid #7dd3fc;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;color:#0369a1;margin:2px">Researcher</button>\n        <button onclick="fillLogin(\'admin@aipbios.com\',\'Admin@12345\')" style="background:#e0f2fe;border:1px solid #7dd3fc;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;color:#0369a1;margin:2px">Admin</button>\n        <button onclick="fillLogin(\'student@aipbios.com\',\'Student@12345\')" style="background:#e0f2fe;border:1px solid #7dd3fc;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;color:#0369a1;margin:2px">Student</button>\n      </div>\n    </div>\n    <!-- Register -->\n    <div id="register-form" style="display:none">\n      <div id="reg-error" style="display:none;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;padding:10px;border-radius:8px;font-size:13px;margin-bottom:12px"></div>\n      <div class="grid-2">\n        <div class="form-group"><label>First name</label><input id="reg-first" placeholder="Priya"></div>\n        <div class="form-group"><label>Last name</label><input id="reg-last" placeholder="Sharma"></div>\n      </div>\n      <div class="form-group"><label>Email</label><input type="email" id="reg-email" placeholder="you@example.com"></div>\n      <div class="form-group"><label>Role</label>\n        <select id="reg-role">\n          <option value="researcher">Researcher</option>\n          <option value="student">Student</option>\n          <option value="industry_user">Industry Professional</option>\n        </select>\n      </div>\n      <div class="form-group"><label>Password <span style="color:#94a3b8;font-weight:400">(min 8 characters)</span></label><input type="password" id="reg-pass" placeholder="e.g. MyPass@123"></div>\n      <div class="form-group"><label>Confirm password</label><input type="password" id="reg-pass2" placeholder="Repeat password" onkeydown="if(event.key===\'Enter\')doRegister()"></div>\n      <button class="btn btn-primary btn-full mt-3" id="reg-btn" onclick="doRegister()">Create account</button>\n      <div style="margin-top:12px;text-align:center;font-size:12px;color:#94a3b8">\n        Already have an account? <a href="#" onclick="switchAuthTab(\'login\')" style="color:#2563eb">Sign in instead</a>\n      </div>\n    </div>\n  </div>\n</div>\n\n<!-- Dashboard -->\n<div id="sidebar">\n  <div class="logo">\n    <div class="logo-icon">🧬</div>\n    <div><div class="logo-text">AIPBIOS</div><div class="logo-sub">AI Intelligence Platform</div></div>\n  </div>\n  <nav>\n    <div class="nav-item" onclick="showPage(\'dashboard\')" id="nav-dashboard"><span class="icon">📊</span> Dashboard</div>\n    <div class="nav-item" onclick="showPage(\'projects\')"  id="nav-projects"><span class="icon">📁</span> Projects</div>\n    <div id="nav-admin-item" style="display:none">\n      <div class="nav-item" onclick="showPage(\'admin\')" id="nav-admin"><span class="icon">👥</span> Admin</div>\n    </div>\n    <div class="nav-section">Intelligence Modules</div>\n    <div class="nav-item" onclick="showPage(\'disease\')"       id="nav-disease"><span class="icon" style="color:#e11d48">🔬</span> Disease Intel</div>\n    <div class="nav-item" onclick="showPage(\'formulation\')"   id="nav-formulation"><span class="icon" style="color:#0d9488">💊</span> Formulation</div>\n    <div class="nav-item" onclick="showPage(\'literature\')"    id="nav-literature"><span class="icon" style="color:#2563eb">📚</span> Literature</div>\n    <div class="nav-item" onclick="showPage(\'regulatory\')"    id="nav-regulatory"><span class="icon" style="color:#d97706">🛡️</span> Regulatory</div>\n    <div class="nav-item" onclick="showPage(\'patent\')"        id="nav-patent"><span class="icon" style="color:#7c3aed">🔎</span> Patent</div>\n    <div class="nav-item" onclick="showPage(\'stability\')"     id="nav-stability"><span class="icon" style="color:#ea580c">🌡️</span> Stability</div>\n    <div class="nav-item" onclick="showPage(\'analytical\')"    id="nav-analytical"><span class="icon" style="color:#059669">📈</span> Analytical</div>\n    <div class="nav-item" onclick="showPage(\'manufacturing\')" id="nav-manufacturing"><span class="icon" style="color:#475569">🏭</span> Manufacturing</div>\n    <div class="nav-item" onclick="showPage(\'cost\')"          id="nav-cost"><span class="icon" style="color:#059669">💰</span> Cost Intel</div>\n    <div class="nav-item" onclick="showPage(\'dossier\')"       id="nav-dossier"><span class="icon" style="color:#2563eb">📄</span> Dossier Builder</div>\n    <div class="nav-item" onclick="showPage(\'research\')"      id="nav-research"><span class="icon" style="color:#db2777">🎓</span> Research Asst.</div>\n  </nav>\n  <div class="user-footer">\n    <div class="user-info">\n      <div class="avatar" id="user-avatar">?</div>\n      <div style="min-width:0">\n        <div style="font-size:12px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" id="user-name">User</div>\n        <div style="font-size:10px;color:var(--faint)" id="user-role">-</div>\n      </div>\n    </div>\n    <button class="btn btn-ghost btn-full btn-sm" onclick="doLogout()">🚪 Sign out</button>\n  </div>\n</div>\n\n<div id="main">\n  <div class="topbar">\n    <div><div class="topbar-title" id="page-title">Dashboard</div><div class="topbar-sub" id="page-sub"></div></div>\n    <div class="flex items-center gap-2">\n      <span id="topbar-user" style="font-size:12px;color:var(--muted)"></span>\n    </div>\n  </div>\n  <div class="content" id="page-content"></div>\n</div>\n\n<!-- Create Project Modal -->\n<div class="modal-overlay" id="create-modal">\n  <div class="modal">\n    <div class="modal-title">New Project <button class="btn btn-ghost btn-sm" onclick="closeModal()">✕</button></div>\n    <div class="form-group"><label>Project Title *</label><input id="proj-title" placeholder="e.g. GlycoHerb Diabetes Tablet — Phase 1"></div>\n    <div class="form-group"><label>Description</label><textarea id="proj-desc" rows="2" placeholder="Brief description of this project"></textarea></div>\n    <div class="grid-2">\n      <div class="form-group"><label>Disease Area</label><input id="proj-disease" placeholder="e.g. Type 2 Diabetes"></div>\n      <div class="form-group"><label>Product Type</label><input id="proj-product" placeholder="e.g. Ayurvedic, Herbal"></div>\n    </div>\n    <div class="flex gap-2 mt-3">\n      <button class="btn btn-secondary flex-1" onclick="closeModal()">Cancel</button>\n      <button class="btn btn-primary flex-1" onclick="createProject()">Create project</button>\n    </div>\n  </div>\n</div>\n\n<script>\n// ── State ──────────────────────────────────────────────────────────────────\nlet AUTH = {token:null, user:null};\nlet PROJECTS = [];\nlet currentPage = \'dashboard\';\nconst API = \'\';\n\n// ── Utilities ──────────────────────────────────────────────────────────────\nasync function api(method, path, body, isForm) {\n  const opts = {method, headers: {}};\n  if (AUTH.token) opts.headers[\'Authorization\'] = \'Bearer \' + AUTH.token;\n  if (body && !isForm) { opts.headers[\'Content-Type\'] = \'application/json\'; opts.body = JSON.stringify(body); }\n  if (body && isForm) opts.body = body;\n  const r = await fetch(API + path, opts);\n  if (r.status === 401) { doLogout(); throw new Error(\'Unauthorized\'); }\n  const ct = r.headers.get(\'content-type\')||\'\';\n  if (ct.includes(\'application/json\')) return r.json();\n  return r;\n}\n\nfunction toast(msg, type=\'info\') {\n  const el = document.createElement(\'div\');\n  el.className = `toast toast-${type}`;\n  el.textContent = msg;\n  document.getElementById(\'toast\').appendChild(el);\n  setTimeout(() => el.remove(), 3500);\n}\n\nfunction fmtDate(s) {\n  if (!s) return \'—\';\n  return new Date(s).toLocaleDateString(\'en-IN\', {day:\'numeric\',month:\'short\',year:\'numeric\'});\n}\n\nfunction statusBadge(s) {\n  const map = {completed:\'green\',active:\'green\',pending:\'purple\',processing:\'purple\',failed:\'red\',draft:\'gray\',archived:\'gray\',on_hold:\'amber\'};\n  return `<span class="badge badge-${map[s]||\'gray\'}">${s}</span>`;\n}\n\nfunction roleBadge(r) {\n  const map = {superadmin:\'purple\',org_admin:\'blue\',researcher:\'green\',student:\'amber\',industry_user:\'gray\'};\n  return `<span class="badge badge-${map[r]||\'gray\'}">${(r||\'\').replace(\'_\',\' \')}</span>`;\n}\n\n// ── Auth ───────────────────────────────────────────────────────────────────\nfunction switchAuthTab(tab) {\n  document.querySelectorAll(\'.auth-tab\').forEach((t,i)=>t.classList.toggle(\'active\',[\'login\',\'register\'][i]===tab));\n  document.getElementById(\'login-form\').style.display = tab===\'login\'?\'block\':\'none\';\n  document.getElementById(\'register-form\').style.display = tab===\'register\'?\'block\':\'none\';\n}\n\nfunction fillLogin(email, pass) {\n  document.getElementById(\'login-email\').value = email;\n  document.getElementById(\'login-password\').value = pass;\n  document.getElementById(\'login-error\').style.display = \'none\';\n}\n\nasync function doLogin() {\n  const email    = document.getElementById(\'login-email\').value.trim();\n  const password = document.getElementById(\'login-password\').value;\n  const errEl    = document.getElementById(\'login-error\');\n  if (errEl) errEl.style.display = \'none\';\n\n  if (!email || !password) {\n    if (errEl) { errEl.style.display=\'block\'; errEl.textContent=\'⚠ Please enter your email and password\'; }\n    return;\n  }\n  const btn = document.getElementById(\'login-btn\');\n  if (btn) { btn.disabled=true; btn.textContent=\'Signing in…\'; }\n  try {\n    const d = await api(\'POST\',\'/api/v1/auth/login/\',{email,password});\n    if (d.error) {\n      if (errEl) { errEl.style.display=\'block\'; errEl.textContent=\'⚠ \' + (d.message||\'Invalid email or password\'); }\n      return;\n    }\n    AUTH = {token: d.access, user: d.user};\n    localStorage.setItem(\'aipbios_token\', d.access);\n    localStorage.setItem(\'aipbios_user\', JSON.stringify(d.user));\n    showDashboard();\n    toast(`Welcome back, ${d.user.first_name||\'User\'}! 👋`,\'success\');\n  } catch(e) {\n    if (errEl) { errEl.style.display=\'block\'; errEl.textContent=\'⚠ Login failed — server may be offline. Make sure python3 server.py is running.\'; }\n  } finally {\n    if (btn) { btn.disabled=false; btn.textContent=\'Sign in\'; }\n  }\n}\n\nfunction showRegError(msg) {\n  const el = document.getElementById(\'reg-error\');\n  if (el) { el.style.display=\'block\'; el.textContent=\'⚠ \' + msg; }\n  toast(msg, \'error\');\n}\nfunction hideRegError() {\n  const el = document.getElementById(\'reg-error\');\n  if (el) el.style.display=\'none\';\n}\n\nasync function doRegister() {\n  hideRegError();\n  const first    = document.getElementById(\'reg-first\').value.trim();\n  const last     = document.getElementById(\'reg-last\').value.trim();\n  const email    = document.getElementById(\'reg-email\').value.trim();\n  const password = document.getElementById(\'reg-pass\').value;\n  const confirm  = document.getElementById(\'reg-pass2\') ? document.getElementById(\'reg-pass2\').value : password;\n  const role     = document.getElementById(\'reg-role\').value;\n\n  if (!email)           { showRegError(\'Email address is required\'); return; }\n  if (!password)        { showRegError(\'Password is required\'); return; }\n  if (password.length < 8) { showRegError(\'Password must be at least 8 characters (e.g. MyPass@123)\'); return; }\n  if (password !== confirm) { showRegError(\'Passwords do not match — please re-type\'); return; }\n\n  const btn = document.getElementById(\'reg-btn\');\n  if (btn) { btn.disabled=true; btn.textContent=\'Creating account…\'; }\n\n  try {\n    const d = await api(\'POST\',\'/api/v1/auth/register/\',{\n      first_name: first, last_name: last, email, password,\n      confirm_password: confirm, role\n    });\n    if (d.error) { showRegError(d.message || \'Registration failed\'); return; }\n    AUTH = {token: d.tokens.access, user: d.user};\n    localStorage.setItem(\'aipbios_token\', d.tokens.access);\n    localStorage.setItem(\'aipbios_user\', JSON.stringify(d.user));\n    showDashboard();\n    toast(\'Account created! Welcome to AIPBIOS 🎉\',\'success\');\n  } catch(e) {\n    showRegError(\'Registration failed — please check your details and try again\');\n  } finally {\n    if (btn) { btn.disabled=false; btn.textContent=\'Create account\'; }\n  }\n}\n\nfunction doLogout() {\n  AUTH = {token:null, user:null};\n  localStorage.removeItem(\'aipbios_token\');\n  localStorage.removeItem(\'aipbios_user\');\n  document.getElementById(\'auth-screen\').style.display=\'flex\';\n  document.getElementById(\'sidebar\').style.display=\'none\';\n  document.getElementById(\'main\').style.display=\'none\';\n}\n\nfunction showDashboard() {\n  document.getElementById(\'auth-screen\').style.display=\'none\';\n  document.getElementById(\'sidebar\').style.display=\'flex\';\n  document.getElementById(\'main\').style.display=\'flex\';\n  const u = AUTH.user;\n  document.getElementById(\'user-name\').textContent = u.full_name || u.email;\n  document.getElementById(\'user-role\').textContent = (u.role||\'\').replace(\'_\',\' \');\n  document.getElementById(\'topbar-user\').textContent = u.full_name || u.email;\n  document.getElementById(\'user-avatar\').textContent = ((u.first_name||\'?\')[0]+(u.last_name||\'?\')[0]).toUpperCase();\n  if ([\'superadmin\',\'org_admin\'].includes(u.role)) {\n    document.getElementById(\'nav-admin-item\').style.display=\'block\';\n  }\n  showPage(\'dashboard\');\n}\n\n// ── Navigation ─────────────────────────────────────────────────────────────\nfunction showPage(page) {\n  currentPage = page;\n  document.querySelectorAll(\'.nav-item\').forEach(n=>n.classList.remove(\'active\'));\n  const navEl = document.getElementById(\'nav-\'+page);\n  if (navEl) navEl.classList.add(\'active\');\n  const pages = {\n    dashboard:    [\'Dashboard\',\'Platform overview and recent activity\'],\n    projects:     [\'Projects\',\'Manage your research and development projects\'],\n    admin:        [\'Admin — User Management\',\'Manage platform users\'],\n    disease:      [\'Disease Intelligence\',\'Opportunity analysis and innovation mapping\'],\n    formulation:  [\'Formulation Intelligence\',\'AI-designed formulations with batch formula\'],\n    literature:   [\'Literature Intelligence\',\'PubMed literature review and research gap analysis\'],\n    regulatory:   [\'Regulatory Intelligence\',\'Pathways, CTD structure, compliance checklist\'],\n    patent:       [\'Patent Intelligence\',\'Prior art review and patentability assessment\'],\n    stability:    [\'Stability Intelligence\',\'ICH-compliant stability plan and shelf-life assessment\'],\n    analytical:   [\'Analytical Intelligence\',\'AI interpretation of HPLC, FTIR, UV-Vis spectra\'],\n    manufacturing:[\'Manufacturing Intelligence\',\'BMR, BPR, SOP and equipment documentation\'],\n    cost:         [\'Cost Intelligence\',\'Raw material, manufacturing and packaging cost model\'],\n    dossier:      [\'Dossier Builder\',\'CTD-ready dossier with PDF and JSON export\'],\n    research:     [\'Research Assistant\',\'Proposals, protocols and publication drafts\'],\n  };\n  const [title,sub] = pages[page]||[page,\'\'];\n  document.getElementById(\'page-title\').textContent = title;\n  document.getElementById(\'page-sub\').textContent = sub;\n  const fn = {\n    dashboard, projects, admin,\n    disease:        ()=>intelPage(\'disease\',\'🔬\',\'Disease Intelligence\',\'bg:#fff1f2;border:#fecdd3;color:#e11d48\',\'disease_intel\',\'disease\',\'analyse\',diseaseForm),\n    formulation:    ()=>intelPage(\'formulation\',\'💊\',\'Formulation Intelligence\',\'bg:#f0fdfa;border:#99f6e4;color:#0d9488\',\'formulation_intel\',\'formulation\',\'analyse\',formulationForm),\n    literature:     ()=>intelPage(\'literature\',\'📚\',\'Literature Intelligence\',\'bg:#eff6ff;border:#bfdbfe;color:#2563eb\',\'literature_intel\',\'literature\',\'analyse\',literatureForm),\n    regulatory:     ()=>intelPage(\'regulatory\',\'🛡️\',\'Regulatory Intelligence\',\'bg:#fffbeb;border:#fde68a;color:#d97706\',\'regulatory_intel\',\'regulatory\',\'analyse\',regulatoryForm),\n    patent:         ()=>intelPage(\'patent\',\'🔎\',\'Patent Intelligence\',\'bg:#faf5ff;border:#e9d5ff;color:#7c3aed\',\'patent_intel\',\'patent\',\'analyse\',patentForm),\n    stability:      ()=>intelPage(\'stability\',\'🌡️\',\'Stability Intelligence\',\'bg:#fff7ed;border:#fed7aa;color:#ea580c\',\'stability_intel\',\'stability\',\'analyse\',stabilityForm),\n    analytical:     ()=>intelPage(\'analytical\',\'📈\',\'Analytical Intelligence\',\'bg:#f0fdf4;border:#bbf7d0;color:#059669\',\'analytical_intel\',\'analytical\',\'analyse\',analyticalForm),\n    manufacturing:  ()=>intelPage(\'manufacturing\',\'🏭\',\'Manufacturing Intelligence\',\'bg:#f8fafc;border:#cbd5e1;color:#475569\',\'manufacturing_intel\',\'manufacturing\',\'analyse\',manufacturingForm),\n    cost:           ()=>intelPage(\'cost\',\'💰\',\'Cost Intelligence\',\'bg:#f0fdf4;border:#bbf7d0;color:#059669\',\'cost_intel\',\'cost\',\'analyse\',costForm),\n    dossier:        ()=>intelPage(\'dossier\',\'📄\',\'Dossier Builder\',\'bg:#eff6ff;border:#bfdbfe;color:#2563eb\',\'dossier\',\'dossier\',\'build\',dossierForm,true),\n    research:       ()=>intelPage(\'research\',\'🎓\',\'Research Assistant\',\'bg:#fdf2f8;border:#f9a8d4;color:#db2777\',\'research_asst\',\'research\',\'generate\',researchForm),\n  };\n  (fn[page]||dashboard)();\n}\n\n// ── Dashboard ──────────────────────────────────────────────────────────────\nasync function dashboard() {\n  document.getElementById(\'page-content\').innerHTML = `<div style="color:var(--muted);text-align:center;padding:30px">Loading…</div>`;\n  try {\n    const data = await api(\'GET\',\'/api/v1/projects/\');\n    PROJECTS = data.results || data || [];\n    const active = PROJECTS.filter(p=>p.status===\'active\').length;\n    const totalJobs = PROJECTS.reduce((a,p)=>a+(p.job_count||0),0);\n\n    document.getElementById(\'page-content\').innerHTML = `\n    <div class="stats-grid">\n      <div class="stat-card"><div class="stat-label">Active Projects</div><div class="stat-value" style="color:var(--brand)">${active}</div></div>\n      <div class="stat-card"><div class="stat-label">Total Projects</div><div class="stat-value">${PROJECTS.length}</div></div>\n      <div class="stat-card"><div class="stat-label">AI Jobs Run</div><div class="stat-value" style="color:var(--teal)">${totalJobs}</div></div>\n    </div>\n    <div style="display:grid;grid-template-columns:1fr 320px;gap:20px">\n      <div class="card" style="padding:20px">\n        <div class="flex justify-between items-center mb-4">\n          <div class="section-title" style="margin:0">Recent Projects</div>\n          <button class="btn btn-secondary btn-sm" onclick="showPage(\'projects\')">View all →</button>\n        </div>\n        ${PROJECTS.length===0 ? `<div class="empty-state"><div class="empty-icon">📁</div><div>No projects yet</div><button class="btn btn-primary btn-sm mt-3" onclick="showPage(\'projects\')">Create first project</button></div>` :\n          PROJECTS.slice(0,6).map(p=>`\n          <div style="display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:8px;cursor:pointer;margin-bottom:4px" onclick="viewProject(\'${p.id}\')" onmouseover="this.style.background=\'#f8fafc\'" onmouseout="this.style.background=\'\'"  >\n            <div style="min-width:0">\n              <div style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.title}</div>\n              <div style="font-size:11px;color:var(--faint)">${p.disease_area||\'General\'} · ${p.job_count||0} jobs · ${fmtDate(p.updated_at)}</div>\n            </div>\n            ${statusBadge(p.status)}\n          </div>`).join(\'\')}\n      </div>\n      <div>\n        <div class="card" style="padding:20px">\n          <div class="section-title">Quick Actions</div>\n          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">\n            ${[[\'🔬\',\'Disease Intel\',\'disease\',\'#fff1f2\',\'#e11d48\'],[\'💊\',\'Formulate\',\'formulation\',\'#f0fdfa\',\'#0d9488\'],[\'🛡️\',\'Regulatory\',\'regulatory\',\'#fffbeb\',\'#d97706\'],[\'📄\',\'Dossier\',\'dossier\',\'#eff6ff\',\'#2563eb\']].map(([ic,lbl,pg,bg,c])=>`\n            <div class="card-hover" style="padding:14px;cursor:pointer;background:${bg};border-color:${c}20" onclick="showPage(\'${pg}\')">\n              <div style="font-size:20px;margin-bottom:6px">${ic}</div>\n              <div style="font-size:12px;font-weight:500;color:${c}">${lbl}</div>\n            </div>`).join(\'\')}\n          </div>\n        </div>\n        <div class="card" style="padding:20px;margin-top:16px">\n          <div class="section-title">Diabetes + Herbal Demo</div>\n          <div style="font-size:12px;color:var(--muted);margin-bottom:12px">Pre-loaded demo project with 5 completed intelligence reports</div>\n          ${PROJECTS.find(p=>p.disease_area&&p.disease_area.includes(\'Diabetes\')) ?\n            `<button class="btn btn-primary btn-sm btn-full" onclick="viewProject(\'${PROJECTS.find(p=>p.disease_area&&p.disease_area.includes(\'Diabetes\')).id}\')">View Demo Project →</button>` :\n            `<button class="btn btn-secondary btn-sm btn-full" onclick="showPage(\'projects\')">Create project first</button>`}\n        </div>\n      </div>\n    </div>`;\n  } catch(e) { document.getElementById(\'page-content\').innerHTML = `<div class="empty-state">Failed to load dashboard</div>`; }\n}\n\n// ── Projects ───────────────────────────────────────────────────────────────\nasync function projects() {\n  document.getElementById(\'page-content\').innerHTML = `<div style="color:var(--muted);text-align:center;padding:30px">Loading…</div>`;\n  const data = await api(\'GET\',\'/api/v1/projects/\');\n  PROJECTS = data.results || data || [];\n\n  document.getElementById(\'page-content\').innerHTML = `\n  <div class="flex justify-between items-center mb-4">\n    <div class="flex gap-2" id="status-filters">\n      ${[\'all\',\'active\',\'draft\',\'archived\'].map(s=>`<button class="chip ${s===\'all\'?\'active\':\'\'}" onclick="filterProjects(\'${s}\',this)">${s===\'all\'?\'All\':s}</button>`).join(\'\')}\n    </div>\n    <button class="btn btn-primary" onclick="document.getElementById(\'create-modal\').classList.add(\'open\')">+ New project</button>\n  </div>\n  <div class="projects-grid" id="projects-grid">\n    ${renderProjects(PROJECTS)}\n  </div>`;\n}\n\nfunction renderProjects(list) {\n  if (!list.length) return `<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon">📁</div><div>No projects found</div><button class="btn btn-primary btn-sm mt-3" onclick="document.getElementById(\'create-modal\').classList.add(\'open\')">Create project</button></div>`;\n  return list.map(p=>`\n  <div class="card-hover project-card" onclick="viewProject(\'${p.id}\')">\n    <div class="flex justify-between items-start gap-2">\n      <div class="project-title">${p.title}</div>\n      ${statusBadge(p.status)}\n    </div>\n    ${p.description?`<div class="project-meta" style="margin-top:6px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${p.description}</div>`:\'\'}\n    <div class="project-tags">\n      ${p.disease_area?`<span class="tag tag-rose">${p.disease_area}</span>`:\'\'}\n      ${p.product_type?`<span class="tag tag-teal">${p.product_type}</span>`:\'\'}\n    </div>\n    <div class="flex justify-between items-center" style="border-top:1px solid var(--border);padding-top:10px;margin-top:8px">\n      <span style="font-size:11px;color:var(--faint)">${fmtDate(p.updated_at)} · ${p.job_count||0} jobs</span>\n      <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();archiveProject(\'${p.id}\',\'${p.title}\')" title="Archive" style="font-size:16px;padding:2px 6px">📦</button>\n    </div>\n  </div>`).join(\'\');\n}\n\nfunction filterProjects(status, el) {\n  document.querySelectorAll(\'#status-filters .chip\').forEach(c=>c.classList.remove(\'active\'));\n  el.classList.add(\'active\');\n  const filtered = status===\'all\' ? PROJECTS : PROJECTS.filter(p=>p.status===status);\n  document.getElementById(\'projects-grid\').innerHTML = renderProjects(filtered);\n}\n\nasync function viewProject(pid) {\n  document.getElementById(\'page-content\').innerHTML = `<div style="text-align:center;padding:40px;color:var(--muted)">Loading project…</div>`;\n  document.getElementById(\'page-title\').textContent = \'Project Detail\';\n  document.querySelectorAll(\'.nav-item\').forEach(n=>n.classList.remove(\'active\'));\n  try {\n    const [proj, dash] = await Promise.all([\n      api(\'GET\',`/api/v1/projects/${pid}/`),\n      api(\'GET\',`/api/v1/projects/${pid}/dashboard/`)\n    ]);\n    const byMod = dash.job_by_module || {};\n    const jobs  = dash.recent_jobs  || [];\n    const modColors = {\'disease_intel\':\'#e11d48\',\'formulation_intel\':\'#0d9488\',\'literature_intel\':\'#2563eb\',\'regulatory_intel\':\'#d97706\',\'patent_intel\':\'#7c3aed\',\'stability_intel\':\'#ea580c\',\'dossier\':\'#1d4ed8\',\'research_asst\':\'#db2777\',\'manufacturing_intel\':\'#475569\',\'cost_intel\':\'#059669\',\'analytical_intel\':\'#16a34a\'};\n\n    document.getElementById(\'page-content\').innerHTML = `\n    <div class="flex gap-2 items-center mb-4">\n      <button class="btn btn-secondary btn-sm" onclick="showPage(\'projects\')">← All projects</button>\n      ${proj.status!==\'archived\'?`<button class="btn btn-secondary btn-sm" onclick="archiveProject(\'${pid}\',\'${proj.title}\')">Archive</button>`:\'\'}\n    </div>\n    <div class="card" style="padding:20px;margin-bottom:20px">\n      <div class="flex justify-between items-start">\n        <div>\n          <h2 style="font-size:18px;font-weight:700">${proj.title}</h2>\n          <div style="margin-top:6px;display:flex;gap:8px;flex-wrap:wrap">\n            ${statusBadge(proj.status)}\n            ${proj.disease_area?`<span class="tag tag-rose">${proj.disease_area}</span>`:\'\'}\n            ${proj.product_type?`<span class="tag tag-teal">${proj.product_type}</span>`:\'\'}\n          </div>\n          ${proj.description?`<p style="font-size:13px;color:var(--muted);margin-top:10px">${proj.description}</p>`:\'\'}\n        </div>\n        <div style="text-align:right;flex-shrink:0">\n          <div style="font-size:11px;color:var(--faint)">Created ${fmtDate(proj.created_at)}</div>\n        </div>\n      </div>\n    </div>\n    <div class="stats-grid" style="margin-bottom:20px">\n      <div class="stat-card"><div class="stat-label">AI Jobs</div><div class="stat-value" style="color:var(--brand)">${proj.job_count||0}</div></div>\n      <div class="stat-card"><div class="stat-label">Total Tokens</div><div class="stat-value" style="font-size:18px">${(dash.total_tokens||0).toLocaleString()}</div></div>\n      <div class="stat-card"><div class="stat-label">Modules Run</div><div class="stat-value" style="color:var(--teal)">${Object.keys(byMod).length}</div></div>\n    </div>\n    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">\n      <div class="card" style="padding:20px">\n        <div class="section-title">Module Activity</div>\n        ${Object.keys(byMod).length===0?`<div class="empty-state" style="padding:20px"><div>No jobs yet</div><button class="btn btn-primary btn-sm mt-3" onclick="showPage(\'disease\')">Run first module →</button></div>`:\n          Object.entries(byMod).map(([mod,cnt])=>`\n          <div style="display:flex;align-items:center;justify-content:space-between;padding:8px;margin-bottom:4px;border-radius:6px;background:#f8fafc">\n            <div style="display:flex;align-items:center;gap:8px">\n              <div style="width:8px;height:8px;border-radius:50%;background:${modColors[mod]||\'#94a3b8\'}"></div>\n              <span style="font-size:12px">${mod.replace(/_intel/,\'\').replace(/_/g,\' \')}</span>\n            </div>\n            <span class="badge badge-blue">${cnt} job${cnt>1?\'s\':\'\'}</span>\n          </div>`).join(\'\')}\n      </div>\n      <div class="card" style="padding:20px">\n        <div class="section-title">Recent Jobs</div>\n        ${jobs.length===0?`<div class="empty-state" style="padding:20px"><div>No jobs yet</div></div>`:\n          jobs.slice(0,8).map(j=>`\n          <div style="display:flex;align-items:center;justify-content:space-between;padding:8px;margin-bottom:4px;cursor:pointer;border-radius:6px" onclick="showJobResult(\'${j.id}\')" onmouseover="this.style.background=\'#f8fafc\'" onmouseout="this.style.background=\'\'">\n            <div style="display:flex;align-items:center;gap:8px;min-width:0">\n              <div style="width:6px;height:6px;border-radius:50%;flex-shrink:0;background:${modColors[j.module_type]||\'#94a3b8\'}"></div>\n              <div style="min-width:0">\n                <div style="font-size:12px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${(j.module_type||\'\').replace(/_/g,\' \')}</div>\n                <div style="font-size:10px;color:var(--faint)">${fmtDate(j.created_at)} · ${j.tokens_used||0} tokens</div>\n              </div>\n            </div>\n            ${statusBadge(j.status)}\n          </div>`).join(\'\')}\n      </div>\n    </div>`;\n  } catch(e) { document.getElementById(\'page-content\').innerHTML = `<div class="empty-state">Failed to load project</div>`; }\n}\n\nasync function showJobResult(jid) {\n  try {\n    const j = await api(\'GET\',`/api/v1/projects/jobs/${jid}/result/`);\n    const out = j.output_payload || {};\n    renderReportModal(j.module_type, out, jid);\n  } catch(e) { toast(\'Failed to load report\',\'error\'); }\n}\n\nfunction renderReportModal(moduleType, output, jobId) {\n  const existing = document.getElementById(\'report-modal\');\n  if (existing) existing.remove();\n  const div = document.createElement(\'div\');\n  div.id = \'report-modal\';\n  div.className = \'modal-overlay open\';\n  div.innerHTML = `\n  <div class="modal" style="width:700px;max-height:85vh;overflow-y:auto">\n    <div class="modal-title">\n      <span>${(moduleType||\'\').replace(/_/g,\' \').replace(/\\b\\w/g,c=>c.toUpperCase())} Report</span>\n      <button class="btn btn-ghost btn-sm" onclick="document.getElementById(\'report-modal\').remove()">✕</button>\n    </div>\n    ${renderReport(output)}\n    <div class="flex gap-2 mt-3">\n      ${moduleType===\'dossier\'?`<button class="btn btn-primary btn-sm" onclick="downloadDossierPdf(\'${jobId}\')">⬇ Download PDF</button><button class="btn btn-secondary btn-sm" onclick="downloadDossierJson(\'${jobId}\')">⬇ Download JSON</button>`:\'\'}\n      ${moduleType===\'research_asst\'?`<button class="btn btn-primary btn-sm" onclick="downloadResearchPdf(\'${jobId}\')">⬇ Download PDF</button>`:\'\'}\n      <button class="btn btn-secondary btn-sm" onclick="document.getElementById(\'report-modal\').remove()">Close</button>\n    </div>\n  </div>`;\n  document.body.appendChild(div);\n}\n\nfunction closeModal() { document.getElementById(\'create-modal\').classList.remove(\'open\'); }\n\nasync function createProject() {\n  const title = document.getElementById(\'proj-title\').value.trim();\n  if (!title) { toast(\'Project title is required\',\'error\'); return; }\n  try {\n    const p = await api(\'POST\',\'/api/v1/projects/\',{\n      title, description: document.getElementById(\'proj-desc\').value,\n      disease_area: document.getElementById(\'proj-disease\').value,\n      product_type: document.getElementById(\'proj-product\').value,\n    });\n    if (p.error) { toast(p.message||\'Failed to create project\',\'error\'); return; }\n    closeModal();\n    [\'proj-title\',\'proj-desc\',\'proj-disease\',\'proj-product\'].forEach(id=>document.getElementById(id).value=\'\');\n    toast(\'Project created!\',\'success\');\n    projects();\n  } catch(e) { toast(\'Failed to create project\',\'error\'); }\n}\n\nasync function archiveProject(pid, title) {\n  if (!confirm(`Archive "${title}"?`)) return;\n  await api(\'POST\',`/api/v1/projects/${pid}/archive/`);\n  toast(\'Project archived\',\'success\');\n  projects();\n}\n\n// ── Admin ──────────────────────────────────────────────────────────────────\nasync function admin() {\n  const data = await api(\'GET\',\'/api/v1/users/\');\n  const users = data.results || data || [];\n  document.getElementById(\'page-content\').innerHTML = `\n  <div class="card">\n    <div style="padding:16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between">\n      <span style="font-size:13px;font-weight:500">👥 ${users.length} users</span>\n      <input style="width:240px" placeholder="Search users…" oninput="searchUsers(this.value)">\n    </div>\n    <table id="users-table">\n      <tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Joined</th><th>Actions</th></tr>\n      ${users.map(u=>`\n      <tr>\n        <td><strong>${u.full_name||\'—\'}</strong></td>\n        <td style="color:var(--muted)">${u.email}</td>\n        <td>${roleBadge(u.role)}</td>\n        <td>${u.is_active?\'<span class="badge badge-green">Active</span>\':\'<span class="badge badge-red">Inactive</span>\'}</td>\n        <td style="color:var(--faint)">${fmtDate(u.created_at)}</td>\n        <td>\n          ${u.is_active\n            ? `<button class="btn btn-ghost btn-sm" style="color:#e11d48" onclick="toggleUser(\'${u.id}\',false)">Deactivate</button>`\n            : `<button class="btn btn-ghost btn-sm" style="color:#059669" onclick="toggleUser(\'${u.id}\',true)">Activate</button>`}\n        </td>\n      </tr>`).join(\'\')}\n    </table>\n  </div>`;\n}\n\nasync function toggleUser(uid_, activate) {\n  const ep = activate ? \'activate\' : \'deactivate\';\n  await api(\'POST\',`/api/v1/users/${uid_}/${ep}/`);\n  toast(`User ${activate?\'activated\':\'deactivated\'}`,\'success\');\n  admin();\n}\n\nasync function searchUsers(q_) {\n  const data = await api(\'GET\',`/api/v1/users/?search=${encodeURIComponent(q_)}`);\n  const users = data.results || data || [];\n  // just re-render rows\n  const tbody = document.querySelector(\'#users-table\');\n  if (!tbody) return;\n  const rows = users.map(u=>`<tr><td><strong>${u.full_name||\'—\'}</strong></td><td style="color:var(--muted)">${u.email}</td><td>${roleBadge(u.role)}</td><td>${u.is_active?\'<span class="badge badge-green">Active</span>\':\'<span class="badge badge-red">Inactive</span>\'}</td><td>${fmtDate(u.created_at)}</td><td>${u.is_active?`<button class="btn btn-ghost btn-sm" style="color:#e11d48" onclick="toggleUser(\'${u.id}\',false)">Deactivate</button>`:`<button class="btn btn-ghost btn-sm" style="color:#059669" onclick="toggleUser(\'${u.id}\',true)">Activate</button>`}</td></tr>`).join(\'\');\n  tbody.innerHTML = `<tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Joined</th><th>Actions</th></tr>` + rows;\n}\n\n// ── Report renderer ────────────────────────────────────────────────────────\nfunction renderReport(output) {\n  if (!output || typeof output !== \'object\' || Object.keys(output).length === 0)\n    return \'<div style="color:var(--muted);padding:20px;text-align:center">No output data</div>\';\n  return `<div class="report-viewer"><div class="report-body">${renderObj(output, 0)}</div></div>`;\n}\n\nfunction renderObj(obj, depth) {\n  if (obj === null || obj === undefined) return \'<span style="color:#94a3b8">null</span>\';\n  if (typeof obj === \'boolean\') return `<span style="color:#d97706">${obj}</span>`;\n  if (typeof obj === \'number\')  return `<span style="color:#d97706">${obj}</span>`;\n  if (typeof obj === \'string\')  return `<span style="color:#059669">"${obj}"</span>`;\n  if (Array.isArray(obj)) {\n    if (obj.length === 0) return \'<span style="color:#94a3b8">[]</span>\';\n    if (typeof obj[0] === \'string\') return obj.map(i=>`<div class="r-list-item">${i}</div>`).join(\'\');\n    return obj.map((item,i)=>`<div style="margin-bottom:8px;padding:8px;background:#f8fafc;border-radius:6px;border-left:3px solid var(--brand)"><div style="font-size:10px;color:var(--faint);margin-bottom:4px">#${i+1}</div>${renderObj(item,depth+1)}</div>`).join(\'\');\n  }\n  if (typeof obj === \'object\') {\n    return Object.entries(obj).map(([k,v])=>{\n      const keyFmt = k.replace(/_/g,\' \').replace(/\\b\\w/g,c=>c.toUpperCase());\n      if (typeof v === \'object\' && v !== null) {\n        return `<div class="r-section"><div class="r-title">${keyFmt}</div>${renderObj(v, depth+1)}</div>`;\n      }\n      return `<div class="r-row"><span class="r-key">${keyFmt}</span><span class="r-val">${renderObj(v,depth+1)}</span></div>`;\n    }).join(\'\');\n  }\n  return String(obj);\n}\n\n// ── Intelligence page factory ──────────────────────────────────────────────\nfunction intelPage(pageId, icon, title, style, moduleType, urlModule, action, formFn, hasDl=false) {\n  document.getElementById(\'page-content\').innerHTML = `\n  <div class="module-banner" style="${style}">\n    <span style="font-size:28px">${icon}</span>\n    <div><div style="font-size:14px;font-weight:600">${title}</div><div style="font-size:12px;opacity:.8">${document.getElementById(\'page-sub\').textContent}</div></div>\n  </div>\n  <div class="intel-2col">\n    <div class="card" style="padding:20px">\n      <div class="section-title">Analysis Input</div>\n      <div id="intel-form">${formFn()}</div>\n      <button class="btn btn-primary btn-full" style="margin-top:16px" id="run-btn" onclick="runIntel(\'${urlModule}\',\'${action}\',\'${moduleType}\')">\n        ✨ Run ${title.split(\' \')[0]} Analysis\n      </button>\n    </div>\n    <div style="min-width:0">\n      <div id="job-status-area"></div>\n      <div id="report-area">\n        <div class="card" style="padding:30px;text-align:center;color:var(--muted)">\n          <div style="font-size:32px;margin-bottom:10px">${icon}</div>\n          <div style="font-weight:500">Awaiting analysis</div>\n          <div style="font-size:12px;margin-top:4px">Fill in the form and click Run to generate AI insights</div>\n        </div>\n      </div>\n    </div>\n  </div>\n  <div style="margin-top:24px" id="prev-reports-section"></div>`;\n  loadPrevReports(urlModule, moduleType);\n}\n\nasync function loadPrevReports(urlModule, moduleType) {\n  try {\n    const data = await api(\'GET\',`/api/v1/intelligence/${urlModule}/reports/`);\n    const reports = data.results || data || [];\n    if (!reports.length) return;\n    const section = document.getElementById(\'prev-reports-section\');\n    if (!section) return;\n    section.innerHTML = `\n    <div class="section-title">Previous Reports (${reports.length})</div>\n    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px">\n      ${reports.slice(0,6).map(r=>`\n      <div class="card-hover" style="padding:14px;cursor:pointer" onclick="showJobResult(\'${r.id||r.job_id}\')">\n        <div style="display:flex;justify-content:space-between;align-items:start;gap:8px;margin-bottom:6px">\n          <div style="font-size:12px;font-weight:500">${(r.module_type||moduleType).replace(/_/g,\' \')}</div>\n          ${statusBadge(r.status||\'completed\')}\n        </div>\n        <div style="font-size:11px;color:var(--faint)">${r.project_title||\'\'} · ${fmtDate(r.created_at)}</div>\n        <div style="font-size:11px;color:var(--muted);margin-top:4px">${r.tokens_used||0} tokens</div>\n      </div>`).join(\'\')}\n    </div>`;\n  } catch(e) {}\n}\n\nasync function runIntel(urlModule, action, moduleType) {\n  const projId = document.getElementById(\'intel-project\')?.value;\n  if (!projId) { toast(\'Please select a project first\',\'error\'); return; }\n  const btn = document.getElementById(\'run-btn\');\n  btn.disabled = true; btn.textContent = \'⏳ Running…\';\n\n  const formData = collectFormData();\n  formData.project_id = projId;\n\n  document.getElementById(\'job-status-area\').innerHTML = `\n  <div class="job-card processing" style="margin-bottom:16px">\n    <div class="job-icon" style="background:#f3e8ff">⏳</div>\n    <div style="flex:1">\n      <div style="font-size:13px;font-weight:500">AI analysis running…</div>\n      <div style="font-size:11px;color:var(--muted);margin-top:2px">Processing your request</div>\n      <div class="progress-bar"><div class="progress-fill" style="width:70%"></div></div>\n    </div>\n  </div>`;\n\n  try {\n    let endpoint = `/api/v1/intelligence/${urlModule}/${action}/`;\n    let payload = formData;\n    // Handle file upload for analytical\n    if (urlModule === \'analytical\') {\n      const fileInput = document.getElementById(\'analytic-image\');\n      if (fileInput?.files[0]) {\n        const fd = new FormData();\n        Object.entries(formData).forEach(([k,v])=>fd.append(k,v));\n        fd.append(\'image\', fileInput.files[0]);\n        payload = fd;\n      }\n    }\n    const isForm = payload instanceof FormData;\n    const result = await api(\'POST\', endpoint, payload, isForm);\n\n    if (result.error) { toast(result.message||\'Analysis failed\',\'error\'); btn.disabled=false; btn.textContent=\'✨ Run Analysis\'; return; }\n\n    const jobId = result.job_id;\n    // For demo, status is immediate\n    const jobResult = await api(\'GET\', `/api/v1/projects/jobs/${jobId}/result/`);\n    const output = jobResult.output_payload || {};\n\n    document.getElementById(\'job-status-area\').innerHTML = `\n    <div class="job-card completed" style="margin-bottom:16px">\n      <div class="job-icon" style="background:#dcfce7;font-size:20px">✅</div>\n      <div style="flex:1">\n        <div style="font-size:13px;font-weight:500;color:#15803d">Analysis complete!</div>\n        <div style="font-size:11px;color:var(--muted);margin-top:2px">Job ID: ${jobId.slice(0,8)}… · ${jobResult.tokens_used||0} tokens used</div>\n        ${moduleType===\'dossier\'?`<div style="display:flex;gap:8px;margin-top:10px"><button class="btn btn-primary btn-sm" onclick="downloadDossierPdf(\'${jobId}\')">⬇ PDF</button><button class="btn btn-secondary btn-sm" onclick="downloadDossierJson(\'${jobId}\')">⬇ JSON</button></div>`:\'\'}\n        ${moduleType===\'research_asst\'?`<button class="btn btn-primary btn-sm" style="margin-top:10px" onclick="downloadResearchPdf(\'${jobId}\')">⬇ Download PDF</button>`:\'\'}\n      </div>\n    </div>`;\n    document.getElementById(\'report-area\').innerHTML = renderReport(output);\n    toast(\'Analysis complete!\',\'success\');\n  } catch(e) {\n    console.error(e);\n    document.getElementById(\'job-status-area\').innerHTML = `<div class="job-card failed" style="margin-bottom:16px"><div class="job-icon" style="background:#fee2e2">❌</div><div><div style="font-size:13px;font-weight:500;color:#b91c1c">Analysis failed</div><div style="font-size:11px;color:var(--muted)">Check console for details</div></div></div>`;\n    toast(\'Analysis failed — check server logs\',\'error\');\n  }\n  btn.disabled=false; btn.textContent=\'✨ Run Analysis\';\n}\n\nfunction collectFormData() {\n  const result = {};\n  document.querySelectorAll(\'#intel-form [data-field]\').forEach(el=>{\n    const k = el.getAttribute(\'data-field\');\n    if (el.tagName===\'INPUT\'||el.tagName===\'SELECT\'||el.tagName===\'TEXTAREA\') result[k] = el.value;\n  });\n  // chips (multi-select)\n  document.querySelectorAll(\'#intel-form [data-chips]\').forEach(container=>{\n    const k = container.getAttribute(\'data-chips\');\n    result[k] = Array.from(container.querySelectorAll(\'.chip.active\')).map(c=>c.textContent.trim());\n  });\n  return result;\n}\n\nasync function projectOptions() {\n  try {\n    const data = await api(\'GET\',\'/api/v1/projects/?status=active\');\n    const ps = data.results||data||[];\n    return ps.map(p=>`<option value="${p.id}">${p.title}</option>`).join(\'\');\n  } catch(e) { return \'\'; }\n}\n\n// ── Form templates ─────────────────────────────────────────────────────────\nlet _projOpts = null;\nasync function getProjectSelect() {\n  // Always fetch fresh projects and populate after DOM renders\n  setTimeout(async () => {\n    const sel = document.getElementById(\'intel-project\');\n    if (!sel) return;\n    sel.innerHTML = \'<option value="">Loading projects...</option>\';\n    try {\n      const data = await api(\'GET\', \'/api/v1/projects/\');\n      const ps = (data.results || data || []).filter(p => p.status !== \'archived\');\n      if (ps.length === 0) {\n        sel.innerHTML = \'<option value="">No projects found — create one first</option>\';\n      } else {\n        sel.innerHTML = \'<option value="">— Select project —</option>\' +\n          ps.map(p => `<option value="${p.id}">${p.title}</option>`).join(\'\');\n      }\n    } catch(e) {\n      sel.innerHTML = \'<option value="">Error loading projects</option>\';\n    }\n  }, 200);\n  return `<select id="intel-project"><option value="">Loading...</option></select>`;\n}\n\nfunction diseaseForm() {\n  getProjectSelect();\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Disease / Condition *</label><input data-field="disease" placeholder="e.g. Type 2 Diabetes Mellitus"></div>\n  <div class="form-group"><label>Healthcare Problem *</label><textarea data-field="healthcare_problem" rows="3" placeholder="Describe the specific unmet need or clinical gap"></textarea></div>\n  <div class="form-group"><label>Target Population *</label><input data-field="target_population" placeholder="e.g. Adults 40–70 in rural India"></div>`;\n}\n\nfunction formulationForm() {\n  getProjectSelect();\n  const ptypes = [\'Ayurvedic\',\'Allopathic\',\'Nutraceutical\',\'Homeopathic\',\'Biologic\',\'Cosmeceutical\'];\n  const dforms = [\'Tablet\',\'Capsule\',\'Syrup\',\'Cream\',\'Gel\',\'Injection\',\'Patch\',\'Drops\',\'Suspension\'];\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Disease / Indication *</label><input data-field="disease" placeholder="e.g. Type 2 Diabetes"></div>\n  <div class="grid-2">\n    <div class="form-group"><label>Product Type</label><select data-field="product_type">${ptypes.map(t=>`<option>${t}</option>`).join(\'\')}</select></div>\n    <div class="form-group"><label>Dosage Form</label><select data-field="dosage_form">${dforms.map(d=>`<option>${d}</option>`).join(\'\')}</select></div>\n  </div>\n  <div class="form-group"><label>Additional Context</label><textarea data-field="additional_context" rows="2" placeholder="Focus herbs, special requirements, etc."></textarea></div>`;\n}\n\nfunction literatureForm() {\n  getProjectSelect();\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Research Topic *</label><input data-field="topic" placeholder="e.g. Bitter Melon in Type 2 Diabetes management — RCTs"></div>\n  <div class="form-group"><label>Year Range</label><input data-field="year_range" value="2019-2024" placeholder="e.g. 2019-2024"></div>\n  <div class="form-group"><label>Additional Context</label><textarea data-field="context" rows="2" placeholder="Specific aspects, geographic region, patient population…"></textarea></div>`;\n}\n\nfunction regulatoryForm() {\n  getProjectSelect();\n  const auths = [\'CDSCO\',\'AYUSH\',\'FSSAI\',\'USFDA\',\'EMA\',\'WHO\',\'TGA\',\'Health Canada\'];\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Disease *</label><input data-field="disease" placeholder="e.g. Hypertension"></div>\n  <div class="grid-2">\n    <div class="form-group"><label>Product Type *</label><input data-field="product_type" placeholder="e.g. Fixed Dose Combination"></div>\n    <div class="form-group"><label>Dosage Form</label><input data-field="dosage_form" placeholder="e.g. Tablet"></div>\n  </div>\n  <div class="form-group"><label>Target Regulatory Authorities</label>\n    <div class="chip-group" data-chips="target_markets">${auths.map(a=>`<span class="chip" onclick="this.classList.toggle(\'active\')">${a}</span>`).join(\'\')}</div>\n  </div>`;\n}\n\nfunction patentForm() {\n  getProjectSelect();\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Invention Description *</label><textarea data-field="invention_description" rows="5" placeholder="Describe your invention — composition, method, novel features, technical problem solved…"></textarea></div>\n  <div class="grid-2">\n    <div class="form-group"><label>Disease / Field *</label><input data-field="disease" placeholder="e.g. Diabetes"></div>\n    <div class="form-group"><label>Product Type *</label><input data-field="product_type" placeholder="e.g. Herbal Tablet"></div>\n  </div>`;\n}\n\nfunction stabilityForm() {\n  getProjectSelect();\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Product Name *</label><input data-field="product_name" placeholder="e.g. GlycoHerb-DS Tablet"></div>\n  <div class="form-group"><label>Dosage Form</label><input data-field="dosage_form" placeholder="e.g. Film-Coated Tablet"></div>\n  <div class="form-group"><label>Storage Condition</label><input data-field="storage_condition" value="25°C/60% RH" placeholder="e.g. 25°C/60% RH"></div>\n  <div class="form-group"><label>Active Ingredients (comma-separated)</label><input data-field="active_ingredients" placeholder="e.g. Bitter Melon Extract, Fenugreek Extract"></div>`;\n}\n\nfunction analyticalForm() {\n  getProjectSelect();\n  const types = [\'hplc\',\'hptlc\',\'ftir\',\'uv_vis\'];\n  const labels = {\'hplc\':\'HPLC Chromatogram\',\'hptlc\':\'HPTLC Plate\',\'ftir\':\'FTIR Spectrum\',\'uv_vis\':\'UV-Vis Spectrum\'};\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Image Type *</label>\n    <div class="chip-group" style="flex-direction:column;gap:6px">${types.map((t,i)=>`\n    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:8px;border:1px solid var(--border);border-radius:8px">\n      <input type="radio" name="img-type" value="${t}" ${i===0?\'checked\':\'\'} data-field="image_type" style="width:auto">\n      <div><div style="font-size:12px;font-weight:500">${labels[t]}</div></div>\n    </label>`).join(\'\')}\n    </div>\n  </div>\n  <div class="form-group"><label>Upload Image (PNG/JPEG, max 20MB)</label>\n    <input type="file" id="analytic-image" accept="image/*" style="padding:6px">\n  </div>\n  <div class="form-group"><label>Analysis Context (optional)</label><textarea data-field="context" rows="2" placeholder="e.g. C18 column, 254nm detection, mobile phase…"></textarea></div>`;\n}\n\nfunction manufacturingForm() {\n  getProjectSelect();\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Product Name *</label><input data-field="product_name" placeholder="e.g. GlycoHerb-DS Tablet"></div>\n  <div class="grid-2">\n    <div class="form-group"><label>Dosage Form</label><input data-field="dosage_form" value="Tablet" placeholder="Tablet"></div>\n    <div class="form-group"><label>Batch Size</label><input data-field="batch_size" value="100,000 units" placeholder="100,000 units"></div>\n  </div>\n  <div class="form-group"><label>Product Type</label><input data-field="product_type" value="Pharmaceutical" placeholder="Pharmaceutical"></div>\n  <div class="form-group"><label>Active Ingredients</label><input data-field="active_ingredients" placeholder="e.g. Bitter Melon Extract 250mg, Fenugreek 125mg"></div>`;\n}\n\nfunction costForm() {\n  getProjectSelect();\n  const markets = [\'India\',\'USA\',\'Europe\',\'UK\',\'Australia\',\'Middle East\'];\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Product Name *</label><input data-field="product_name" placeholder="e.g. GlycoHerb-DS Tablet"></div>\n  <div class="grid-2">\n    <div class="form-group"><label>Dosage Form</label><input data-field="dosage_form" value="Tablet"></div>\n    <div class="form-group"><label>Batch Size</label><input data-field="batch_size" value="100,000 units"></div>\n  </div>\n  <div class="form-group"><label>Target Market</label>\n    <div class="chip-group" data-chips="target_market">${markets.map((m,i)=>`<span class="chip ${i===0?\'active\':\'\'}" onclick="document.querySelectorAll(\'[data-chips=target_market] .chip\').forEach(c=>c.classList.remove(\'active\'));this.classList.add(\'active\')">${m}</span>`).join(\'\')}</div>\n  </div>\n  <div class="form-group"><label>Active Ingredients</label><input data-field="active_ingredients" placeholder="e.g. Bitter Melon Extract, Fenugreek Extract"></div>`;\n}\n\nfunction dossierForm() {\n  getProjectSelect();\n  const dtypes = [[\'full_ctd\',\'Full CTD Dossier\',\'5-module Common Technical Document\'],[\'pdr\',\'Product Development Report\',\'ICH Q8 development report\'],[\'executive_summary\',\'Executive Summary\',\'High-level management brief\']];\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="grid-2">\n    <div class="form-group"><label>Product Name *</label><input data-field="product_name" placeholder="e.g. GlycoHerb-DS Tablet"></div>\n    <div class="form-group"><label>Disease *</label><input data-field="disease" placeholder="e.g. Type 2 Diabetes"></div>\n  </div>\n  <div class="grid-2">\n    <div class="form-group"><label>Dosage Form</label><input data-field="dosage_form" placeholder="Film-Coated Tablet"></div>\n    <div class="form-group"><label>Product Type</label><input data-field="product_type" placeholder="Ayurvedic"></div>\n  </div>\n  <div class="form-group"><label>Document Type</label>\n    ${dtypes.map(([v,l,d],i)=>`\n    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:8px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px">\n      <input type="radio" name="doc-type" value="${v}" ${i===0?\'checked\':\'\'} data-field="doc_type" style="width:auto">\n      <div><div style="font-size:12px;font-weight:500">${l}</div><div style="font-size:11px;color:var(--muted)">${d}</div></div>\n    </label>`).join(\'\')}\n  </div>`;\n}\n\nfunction researchForm() {\n  getProjectSelect();\n  const dtypes = [[\'research_proposal\',\'Research Proposal\'],[\'synopsis\',\'Synopsis\'],[\'protocol\',\'Clinical Protocol\'],[\'publication_draft\',\'Publication Draft\'],[\'review_article\',\'Review Article\']];\n  return `\n  <div class="form-group"><label>Project *</label><select id="intel-project"><option value="">— Select project —</option></select></div>\n  <div class="form-group"><label>Document Type</label>\n    <div class="chip-group">${dtypes.map(([v,l],i)=>`<span class="chip ${i===0?\'active\':\'\'}" onclick="document.querySelectorAll(\'#intel-form .chip\').forEach(c=>c.classList.remove(\'active\'));this.classList.add(\'active\');document.querySelector(\'[data-field=doc_type]\').value=\'${v}\'">${l}</span>`).join(\'\')}\n    <input type="hidden" data-field="doc_type" value="research_proposal"></div>\n  </div>\n  <div class="form-group"><label>Research Title *</label><input data-field="title" placeholder="Full research title"></div>\n  <div class="form-group"><label>Disease / Research Area *</label><input data-field="disease" placeholder="e.g. Type 2 Diabetes"></div>\n  <div class="form-group"><label>Primary Objective *</label><textarea data-field="objective" rows="2" placeholder="State the primary research objective…"></textarea></div>\n  <div class="form-group"><label>Background Context</label><textarea data-field="context" rows="2" placeholder="Relevant background, previous work, rationale…"></textarea></div>`;\n}\n\n// ── Downloads ──────────────────────────────────────────────────────────────\nasync function downloadDossierPdf(jobId) {\n  toast(\'Generating PDF…\',\'info\');\n  try {\n    const resp = await fetch(`/api/v1/intelligence/dossier/reports/${jobId}/download/pdf/`,\n      {headers: {Authorization: \'Bearer \'+AUTH.token}});\n    if (!resp.ok) { toast(\'PDF generation failed\',\'error\'); return; }\n    const blob = await resp.blob();\n    const url  = URL.createObjectURL(blob);\n    const a    = document.createElement(\'a\');\n    a.href = url; a.download = \'dossier_report.pdf\';\n    a.click(); URL.revokeObjectURL(url);\n    toast(\'PDF downloaded!\',\'success\');\n  } catch(e) { toast(\'Download failed\',\'error\'); }\n}\n\nasync function downloadDossierJson(jobId) {\n  toast(\'Preparing JSON…\',\'info\');\n  try {\n    const resp = await fetch(`/api/v1/intelligence/dossier/reports/${jobId}/download/json/`,\n      {headers: {Authorization: \'Bearer \'+AUTH.token}});\n    const blob = await resp.blob();\n    const url  = URL.createObjectURL(blob);\n    const a    = document.createElement(\'a\');\n    a.href = url; a.download = \'dossier_report.json\';\n    a.click(); URL.revokeObjectURL(url);\n    toast(\'JSON downloaded!\',\'success\');\n  } catch(e) { toast(\'Download failed\',\'error\'); }\n}\n\nasync function downloadResearchPdf(jobId) {\n  toast(\'Generating PDF…\',\'info\');\n  try {\n    const resp = await fetch(`/api/v1/intelligence/research/documents/${jobId}/download/`,\n      {headers: {Authorization: \'Bearer \'+AUTH.token}});\n    if (!resp.ok) { toast(\'PDF generation failed\',\'error\'); return; }\n    const blob = await resp.blob();\n    const url  = URL.createObjectURL(blob);\n    const a    = document.createElement(\'a\');\n    a.href = url; a.download = \'research_document.pdf\';\n    a.click(); URL.revokeObjectURL(url);\n    toast(\'PDF downloaded!\',\'success\');\n  } catch(e) { toast(\'Download failed\',\'error\'); }\n}\n\n// ── Boot ───────────────────────────────────────────────────────────────────\n(function init() {\n  const token = localStorage.getItem(\'aipbios_token\');\n  const user  = localStorage.getItem(\'aipbios_user\');\n  if (token && user) {\n    AUTH = {token, user: JSON.parse(user)};\n    showDashboard();\n  } else {\n    document.getElementById(\'auth-screen\').style.display=\'flex\';\n  }\n})();\n</script>\n</body>\n</html>\n'

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
