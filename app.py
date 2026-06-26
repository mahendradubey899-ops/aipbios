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
DB_PATH  = os.path.join(BASE_DIR, 'aipbios.db')
SECRET   = 'aipbios-live-prototype-secret-2024'
app      = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))

# ── OpenAI setup ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

def call_openai(system_prompt, user_prompt, model='gpt-4o'):
    """Call OpenAI and return parsed JSON output."""
    if not OPENAI_API_KEY:
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
                "Authorization": f"Bearer {OPENAI_API_KEY}"
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
        if OPENAI_API_KEY and module_name in MODULE_PROMPTS:
            system_prompt, user_prompt_fn = MODULE_PROMPTS[module_name]
            user_prompt = user_prompt_fn(d)
            output, tokens = call_openai(system_prompt, user_prompt)
            if output is None:
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
            'ai_used':  bool(OPENAI_API_KEY),
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
        'ai_mode': 'live' if OPENAI_API_KEY else 'demo'
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        'openai_connected': bool(OPENAI_API_KEY),
        'mode': 'Live AI — any disease supported' if OPENAI_API_KEY else 'Demo mode — add OPENAI_API_KEY for live AI'
    })

# ── STATIC / FRONTEND ─────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    static_dir = os.path.join(BASE_DIR, 'static')
    # Serve API routes normally
    if path.startswith('api/') or path.startswith('health'):
        return jsonify({'error': 'Not found'}), 404
    # Serve static files
    if path and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    # Always serve index.html for everything else
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_dir, 'index.html')
    # Fallback - return basic HTML if index.html missing
    return '''<!DOCTYPE html><html><head><title>AIPBIOS</title></head>
<body style="font-family:sans-serif;text-align:center;padding:50px;background:#0f172a;color:white">
<h1>🧬 AIPBIOS</h1><p>AI Intelligence Platform</p>
<p style="color:#94a3b8">Loading...</p>
<script>setTimeout(()=>location.reload(),3000)</script>
</body></html>''', 200

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
    from server_demo_data import DEMO_REPORTS
    for mod, inp, out, tok in DEMO_REPORTS(main_pid, res_id, t):
        db.execute('INSERT INTO intelligence_jobs(id,project_id,created_by,module_type,status,input_payload,output_payload,tokens_used,created_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (uid(),main_pid,res_id,mod,'completed',json.dumps(inp),json.dumps(out),tok,t,t))
        db.execute('UPDATE projects SET job_count=job_count+1 WHERE id=?',[main_pid])
    db.commit(); db.close()
    print("✓ Demo data seeded")

# Initialize on startup
init_db()
seed()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    mode = 'LIVE AI MODE' if OPENAI_API_KEY else 'DEMO MODE'
    print(f"AIPBIOS starting — {mode} on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
