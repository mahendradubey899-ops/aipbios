"""
AIPBIOS v2.0 — Enterprise AI Intelligence Platform
Technical CEO + Chief Scientific Officer + Regulatory Intelligence Expert
13 Modules | PDF Downloads | Expert-Level Output
"""
import sqlite3, uuid, hashlib, json, datetime, os, io
from flask import Flask, request, jsonify, g, send_from_directory, make_response, Response
import jwt
from functools import wraps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.environ.get('DB_PATH', os.path.join('/tmp', 'aipbios.db'))
SECRET   = os.environ.get('SECRET_KEY', 'aipbios-v2-enterprise-2024')
app      = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))

# ── OpenAI ────────────────────────────────────────────────────────────────────
def call_openai(system_prompt, user_prompt, model='gpt-4o', max_tokens=4096):
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return None, 'OPENAI_API_KEY not set'
    import urllib.request as ur
    try:
        payload = json.dumps({
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_prompt}
            ],
            'temperature': 0.2,
            'max_tokens': max_tokens,
            'response_format': {'type': 'json_object'}
        }).encode()
        req = ur.Request(
            'https://api.openai.com/v1/chat/completions',
            data=payload,
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {api_key}'},
            method='POST'
        )
        with ur.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        content = data['choices'][0]['message']['content']
        tokens  = data.get('usage', {}).get('total_tokens', 0)
        return json.loads(content), tokens
    except Exception as e:
        return None, str(e)

# ── Vision call (for image modules) ──────────────────────────────────────────
def call_openai_vision(system_prompt, user_prompt, image_b64, model='gpt-4o'):
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return None, 'OPENAI_API_KEY not set'
    import urllib.request as ur
    try:
        payload = json.dumps({
            'model': model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': system_prompt + '\n\n' + user_prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}', 'detail': 'high'}}
                ]
            }],
            'max_tokens': 4096,
            'response_format': {'type': 'json_object'}
        }).encode()
        req = ur.Request(
            'https://api.openai.com/v1/chat/completions',
            data=payload,
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {api_key}'},
            method='POST'
        )
        with ur.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        content = data['choices'][0]['message']['content']
        tokens  = data.get('usage', {}).get('total_tokens', 0)
        return json.loads(content), tokens
    except Exception as e:
        return None, str(e)

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA journal_mode=WAL')
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
        organisation TEXT DEFAULT 'AIPBIOS', is_active INTEGER DEFAULT 1,
        created_at TEXT
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
        input_payload TEXT DEFAULT '{}', output_payload TEXT DEFAULT '{}',
        tokens_used INTEGER DEFAULT 0, created_at TEXT, completed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, project_id TEXT, job_id TEXT,
        title TEXT, doc_type TEXT, storage_path TEXT DEFAULT '', created_at TEXT
    );
    """)
    db.commit(); db.close()

def h(pw): return hashlib.sha256(pw.encode()).hexdigest()
def uid(): return str(uuid.uuid4())
def now(): return datetime.datetime.utcnow().isoformat()

def make_token(user):
    exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    return jwt.encode({'user_id': user['id'], 'email': user['email'],
                       'role': user['role'], 'exp': exp}, SECRET, algorithm='HS256')

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

# ── Prompts ───────────────────────────────────────────────────────────────────
try:
    from knowledge_base import build_disease_context_string, get_regulatory_context, verify_pubmed_citation
except ImportError:
    def build_disease_context_string(d): return ''
    def get_regulatory_context(d): return {}
    def verify_pubmed_citation(p): return None

try:
    from prompts import (
        DISEASE_SYSTEM, FORMULATION_SYSTEM, LITERATURE_SYSTEM,
        REGULATORY_SYSTEM, PATENT_SYSTEM, STABILITY_SYSTEM,
        ANALYTICAL_SYSTEM, MANUFACTURING_SYSTEM, COST_SYSTEM,
        DOSSIER_SYSTEM, RESEARCH_SYSTEM, MICROBIOLOGY_SYSTEM,
        STATISTICAL_SYSTEM, PRECLINICAL_SYSTEM, CLINICAL_SYSTEM,
        MODULE_PROMPTS
    )
except ImportError:
    # Fallback minimal prompts
    DISEASE_SYSTEM = FORMULATION_SYSTEM = LITERATURE_SYSTEM = REGULATORY_SYSTEM = \
    PATENT_SYSTEM = STABILITY_SYSTEM = ANALYTICAL_SYSTEM = MANUFACTURING_SYSTEM = \
    COST_SYSTEM = DOSSIER_SYSTEM = RESEARCH_SYSTEM = MICROBIOLOGY_SYSTEM = \
    STATISTICAL_SYSTEM = "You are an expert pharmaceutical analyst. Return ONLY valid JSON with comprehensive analysis."

# ── Fallback demo report ──────────────────────────────────────────────────────
def fallback_report(module, data):
    subject = data.get('disease', data.get('topic', data.get('product_name',
              data.get('title', 'the specified subject'))))
    return {
        '_demo_mode': True,
        '_message': f"Demo report for '{subject}'. Add OPENAI_API_KEY for live AI analysis.",
        'executive_summary': f'This is a demo report for {subject}. Configure OpenAI API key for real analysis.',
        'note': 'Set OPENAI_API_KEY environment variable on Render to enable live AI reports.'
    }

# ── MODULE PROMPTS ────────────────────────────────────────────────────────────
MODULE_PROMPTS = {
    'disease_intel': (DISEASE_SYSTEM, lambda d:
        f"Conduct expert CEO-level disease intelligence analysis for:\n"
        f"Disease: {d.get('disease','')}\n"
        f"Country Focus: {d.get('country','India')}\n"
        f"Additional Context: {d.get('additional_context','')}"),

    'formulation_intel': (FORMULATION_SYSTEM, lambda d:
        f"Design complete pharmaceutical formulation for 500g batch:\n"
        f"Disease/Indication: {d.get('disease','')}\n"
        f"Product Type: {d.get('product_type','Ayurvedic')}\n"
        f"Dosage Form: {d.get('dosage_form','Tablet')}\n"
        f"Country/Pharmacopoeia: {d.get('country','India')}\n"
        f"Preferred Ingredients: {d.get('active_ingredients','')}\n"
        f"Special Requirements: {d.get('additional_context','')}"),

    'literature_intel': (LITERATURE_SYSTEM, lambda d:
        f"Comprehensive literature review:\n"
        f"Topic: {d.get('topic','')}\n"
        f"Year Range: {d.get('year_range','2019-2024')}\n"
        f"Context: {d.get('context','')}"),

    'regulatory_intel': (REGULATORY_SYSTEM, lambda d:
        f"Expert regulatory intelligence for India submission:\n"
        f"Disease: {d.get('disease','')}\n"
        f"Product Type: {d.get('product_type','')}\n"
        f"Dosage Form: {d.get('dosage_form','')}\n"
        f"Target Authorities: {d.get('target_markets',['CDSCO'])}\n"
        f"Country: {d.get('country','India')}"),

    'patent_intel': (PATENT_SYSTEM, lambda d:
        f"Patent intelligence and draft claims:\n"
        f"Invention: {d.get('invention_description','')}\n"
        f"Disease Area: {d.get('disease','')}\n"
        f"Product Type: {d.get('product_type','')}"),

    'stability_intel': (STABILITY_SYSTEM, lambda d:
        f"Complete ICH-compliant stability programme for India Zone IVb:\n"
        f"Product: {d.get('product_name','')}\n"
        f"Dosage Form: {d.get('dosage_form','')}\n"
        f"Active Ingredients: {d.get('active_ingredients','')}\n"
        f"Packaging: {d.get('packaging','')}\n"
        f"Stability Data Context: {d.get('stability_context','')}"),

    'analytical_intel': (ANALYTICAL_SYSTEM, lambda d:
        f"Expert analytical interpretation:\n"
        f"Technique: {d.get('image_type','HPLC')}\n"
        f"Sample: {d.get('product_name','')}\n"
        f"Expected Analytes: {d.get('expected_analytes','')}\n"
        f"Method/Data Context: {d.get('context','')}"),

    'manufacturing_intel': (MANUFACTURING_SYSTEM, lambda d:
        f"Complete GMP manufacturing documentation:\n"
        f"Product: {d.get('product_name','')}\n"
        f"Dosage Form: {d.get('dosage_form','Tablet')}\n"
        f"Batch Size: {d.get('batch_size','100,000 units')}\n"
        f"Active Ingredients: {d.get('active_ingredients','')}"),

    'cost_intel': (COST_SYSTEM, lambda d:
        f"Comprehensive pharmaceutical cost model:\n"
        f"Product: {d.get('product_name','')}\n"
        f"Dosage Form: {d.get('dosage_form','')}\n"
        f"Batch Size: {d.get('batch_size','100,000 units')}\n"
        f"Target Market: {d.get('target_market','India')}\n"
        f"Active Ingredients: {d.get('active_ingredients','')}"),

    'dossier': (DOSSIER_SYSTEM, lambda d:
        f"Submission-ready CTD dossier framework:\n"
        f"Product: {d.get('product_name','')}\n"
        f"Disease: {d.get('disease','')}\n"
        f"Dosage Form: {d.get('dosage_form','')}\n"
        f"Product Type: {d.get('product_type','')}\n"
        f"Submission Authority: {d.get('authority','CDSCO India')}"),

    'research_asst': (RESEARCH_SYSTEM, lambda d:
        f"Generate {d.get('doc_type','research_proposal')} document:\n"
        f"Title: {d.get('title','')}\n"
        f"Disease: {d.get('disease','')}\n"
        f"Objective: {d.get('objective','')}\n"
        f"Context: {d.get('context','')}\n"
        f"Funding Body: {d.get('funding_body','ICMR')}\n"
        f"{'Statistical Validation Plan: Include complete pre-execution statistical validation with G*Power sample size, dosing rationale, sampling strategy, model selection, sensitivity analysis' if d.get('doc_type') == 'statistical_validation' else ''}"),

    'microbiology_intel': (MICROBIOLOGY_SYSTEM, lambda d:
        f"Microbiological analysis:\n"
        f"Sample Type: {d.get('sample_type','pharmaceutical/clinical')}\n"
        f"Culture Medium: {d.get('culture_medium','Nutrient Agar')}\n"
        f"Context: {d.get('context','')}"),

    'statistical_intel': (STATISTICAL_SYSTEM, lambda d:
        f"Statistical analysis and report:\n"
        f"Study Title: {d.get('title','')}\n"
        f"Statistical Model: {d.get('statistical_model','appropriate')}\n"
        f"Data Context: {d.get('context','')}\n"
        f"Study Design: {d.get('study_design','')}"),
}

# ── AUTH ──────────────────────────────────────────────────────────────────────
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
            'role': d.get('role','researcher')}
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
    return jsonify({'access': make_token(user), 'refresh': make_token(user), 'user': user})

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

@app.route('/api/v1/users/<uid_>/activate/', methods=['POST'])
@auth_required
def activate_user(uid_):
    q('UPDATE users SET is_active=1 WHERE id=?', [uid_])
    return jsonify({'message': 'User activated'})

@app.route('/api/v1/users/<uid_>/deactivate/', methods=['POST'])
@auth_required
def deactivate_user(uid_):
    q('UPDATE users SET is_active=0 WHERE id=?', [uid_])
    return jsonify({'message': 'User deactivated'})

# ── PROJECTS ──────────────────────────────────────────────────────────────────
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
    rows = q('SELECT * FROM projects WHERE created_by=? AND status=? ORDER BY updated_at DESC',
             [uid_, status_f]) if status_f and status_f != 'all' else            q('SELECT * FROM projects WHERE created_by=? ORDER BY updated_at DESC', [uid_])
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

@app.route('/api/v1/projects/<pid>/archive/', methods=['POST'])
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
        by_mod[j['module_type']] = by_mod.get(j['module_type'], 0) + 1
        by_stat[j['status']]     = by_stat.get(j['status'], 0) + 1
        total_tok += j.get('tokens_used', 0)
    recent = q('SELECT * FROM intelligence_jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 10', [pid])
    return jsonify({**p, 'job_by_module': by_mod, 'job_by_status': by_stat,
                    'total_tokens': total_tok, 'recent_jobs': recent, 'member_count': 1})

@app.route('/api/v1/projects/<pid>/jobs/', methods=['GET'])
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

# ── INTELLIGENCE HANDLER ──────────────────────────────────────────────────────
def make_intel_route(module_name, url_name, action):
    @app.route(f'/api/v1/intelligence/{url_name}/{action}/',
               methods=['POST'], endpoint=f'intel_{module_name}')
    @auth_required
    def handler():
        # Handle both JSON and multipart
        if request.content_type and 'multipart' in request.content_type:
            d = {k: request.form.get(k,'') for k in request.form}
        else:
            d = request.get_json() or {}

        pid = d.get('project_id')
        if not pid:
            return jsonify({'error': True, 'message': 'project_id required'}), 400
        p = q('SELECT * FROM projects WHERE id=?', [pid], one=True)
        if not p:
            return jsonify({'error': True, 'message': 'Project not found'}), 404

        t = now()
        api_key = os.environ.get('OPENAI_API_KEY','')
        output = None
        tokens = 0

        if api_key and module_name in MODULE_PROMPTS:
            try:
                system_prompt, user_prompt_fn = MODULE_PROMPTS[module_name]
                user_prompt = user_prompt_fn(d)

                # Handle image upload for vision modules
                image_file = request.files.get('image') if request.files else None
                if image_file and module_name in ('analytical_intel', 'microbiology_intel'):
                    import base64
                    image_b64 = base64.b64encode(image_file.read()).decode()
                    output, tokens = call_openai_vision(system_prompt, user_prompt, image_b64)
                else:
                    output, tokens = call_openai(system_prompt, user_prompt)

                if output is None:
                    print(f"OpenAI error for {module_name}: {tokens}")
                    output = fallback_report(module_name, d)
                    tokens = 0
            except Exception as e:
                import traceback
                print(f"Handler error {module_name}: {traceback.format_exc()}")
                output = fallback_report(module_name, d)
                tokens = 0
        else:
            output = fallback_report(module_name, d)
            tokens = 0

        jid = uid()
        q('''INSERT INTO intelligence_jobs(id,project_id,created_by,module_type,status,
           input_payload,output_payload,tokens_used,created_at,completed_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)''',
          [jid, pid, g.user['user_id'], module_name, 'completed',
           json.dumps({k:v for k,v in d.items() if k != 'project_id' and not hasattr(v, 'read')}),
           json.dumps(output), tokens, t, t])
        q('UPDATE projects SET job_count=job_count+1, updated_at=? WHERE id=?', [t, pid])

        return jsonify({
            'job_id':  jid,
            'status':  'completed',
            'message': f'{module_name.replace("_"," ").title()} analysis complete.',
            'ai_used': bool(api_key),
            'poll_url': f'/api/v1/projects/jobs/{jid}/status/'
        }), 202
    return handler

ROUTES = [
    ('disease_intel',      'disease',        'analyse'),
    ('formulation_intel',  'formulation',    'analyse'),
    ('literature_intel',   'literature',     'analyse'),
    ('regulatory_intel',   'regulatory',     'analyse'),
    ('patent_intel',       'patent',         'analyse'),
    ('stability_intel',    'stability',      'analyse'),
    ('analytical_intel',   'analytical',     'analyse'),
    ('manufacturing_intel','manufacturing',  'analyse'),
    ('cost_intel',         'cost',           'analyse'),
    ('dossier',            'dossier',        'build'),
    ('research_asst',      'research',       'generate'),
    ('microbiology_intel', 'microbiology',   'analyse'),
    ('statistical_intel',  'statistical',    'analyse'),
    ('preclinical_intel',  'preclinical',    'analyse'),
    ('clinical_intel',     'clinical',       'analyse'),
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
                FROM intelligence_jobs ij JOIN projects p ON ij.project_id=p.id
                WHERE ij.module_type=? AND ij.created_by=?
                ORDER BY ij.created_at DESC''', [mt, g.user['user_id']])
    for r in rows:
        r['output_payload'] = json.loads(r['output_payload'] or '{}')
        r['input_payload']  = json.loads(r['input_payload']  or '{}')
    return jsonify({'count': len(rows), 'results': rows})

# ── PDF DOWNLOAD ──────────────────────────────────────────────────────────────
def get_pdf(jid):
    """Get job and generate PDF. Returns (pdf_bytes, filename) or raises."""
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [jid], one=True)
    if not j: return None, None
    output = json.loads(j['output_payload'] or '{}')
    inp    = json.loads(j['input_payload']  or '{}')
    module = j['module_type']
    subject = (inp.get('disease') or inp.get('product_name') or
               inp.get('title') or inp.get('topic') or module)
    from pdf_engine import generate_report_pdf
    title_map = {
        'disease_intel': f'Disease Intelligence — {subject}',
        'formulation_intel': f'Formulation Design — {subject}',
        'literature_intel': f'Literature Review — {subject}',
        'regulatory_intel': f'Regulatory Intelligence — {subject}',
        'patent_intel': f'Patent Intelligence — {subject}',
        'stability_intel': f'Stability Programme — {subject}',
        'analytical_intel': f'Analytical Report — {subject}',
        'manufacturing_intel': f'Manufacturing Documentation — {subject}',
        'cost_intel': f'Cost Intelligence — {subject}',
        'dossier': f'Regulatory Dossier — {subject}',
        'research_asst': f'Research Document — {subject}',
        'microbiology_intel': f'Microbiological Report — {subject}',
        'statistical_intel': f'Statistical Analysis — {subject}',
    }
    title = title_map.get(module, f'AIPBIOS Report — {subject}')
    pdf_bytes = generate_report_pdf(output, module, title, inp)
    filename = f'AIPBIOS_{module}_{subject[:30].replace(" ","_")}.pdf'
    return pdf_bytes, filename

@app.route('/api/v1/jobs/<jid>/download/pdf/', methods=['GET'])
@auth_required
def download_pdf(jid):
    pdf_bytes, filename = get_pdf(jid)
    if not pdf_bytes:
        return jsonify({'error': 'Job not found'}), 404
    resp = make_response(pdf_bytes)
    resp.headers['Content-Type'] = 'application/pdf'
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp

# Keep old dossier download endpoint for backwards compatibility
@app.route('/api/v1/intelligence/dossier/reports/<rid>/download/<fmt>/', methods=['GET'])
@auth_required
def download_dossier_compat(rid, fmt):
    if fmt == 'pdf':
        pdf_bytes, filename = get_pdf(rid)
        if not pdf_bytes: return jsonify({'error': 'Not found'}), 404
        resp = make_response(pdf_bytes)
        resp.headers['Content-Type'] = 'application/pdf'
        resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return resp
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [rid], one=True)
    if not j: return jsonify({'error': 'Not found'}), 404
    output = json.loads(j['output_payload'] or '{}')
    resp = make_response(json.dumps(output, indent=2))
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Content-Disposition'] = f'attachment; filename="dossier_{rid}.json"'
    return resp

@app.route('/api/v1/intelligence/research/documents/<rid>/download/', methods=['GET'])
@auth_required
def download_research_compat(rid):
    pdf_bytes, filename = get_pdf(rid)
    if not pdf_bytes: return jsonify({'error': 'Not found'}), 404
    resp = make_response(pdf_bytes)
    resp.headers['Content-Type'] = 'application/pdf'
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp

# ── HEALTH ────────────────────────────────────────────────────────────────────

# ── PDF ENGINE (embedded) ────────────────────────────────────────────────────
"""
AIPBIOS PDF Engine
Professional PDF generation with watermark, logo, and branding.
Used by all 13 intelligence modules.
"""
import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# ── Brand colours ─────────────────────────────────────────────────────────────
BRAND_BLUE   = colors.HexColor('#1a3a5c')
BRAND_TEAL   = colors.HexColor('#0d9488')
BRAND_LIGHT  = colors.HexColor('#eff6ff')
ACCENT_AMBER = colors.HexColor('#d97706')
ACCENT_ROSE  = colors.HexColor('#e11d48')
GREY_DARK    = colors.HexColor('#374151')
GREY_MED     = colors.HexColor('#6b7280')
GREY_LIGHT   = colors.HexColor('#f3f4f6')
BORDER_COLOR = colors.HexColor('#e2e8f0')
WHITE        = colors.white

# ── Watermark canvas maker ─────────────────────────────────────────────────────
def make_watermark_canvas(output_buf, pagesize=A4):
    """Returns a canvas class that adds AIPBIOS watermark + header/footer to every page."""
    class WatermarkCanvas(canvas.Canvas):
        def __init__(self, filename, **kwargs):
            super().__init__(filename, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_watermark_and_chrome(num_pages)
                super().showPage()
            super().save()

        def _draw_watermark_and_chrome(self, total_pages):
            w, h = self._pagesize
            # Watermark
            self.saveState()
            self.setFont('Helvetica-Bold', 52)
            self.setFillColor(colors.HexColor('#1a3a5c'))
            self.setFillAlpha(0.04)
            self.translate(w/2, h/2)
            self.rotate(45)
            self.drawCentredString(0, 0, 'AIPBIOS')
            self.rotate(-45)
            self.translate(-w/2, -h/2)
            self.restoreState()

            # Header bar
            self.saveState()
            self.setFillColor(BRAND_BLUE)
            self.rect(0, h - 1.1*cm, w, 1.1*cm, fill=1, stroke=0)
            self.setFillColor(WHITE)
            self.setFont('Helvetica-Bold', 11)
            self.drawString(1.5*cm, h - 0.75*cm, 'AIPBIOS')
            self.setFont('Helvetica', 8)
            self.drawString(3.5*cm, h - 0.75*cm, '— AI Intelligence Platform for Healthcare & Pharma')
            self.setFont('Helvetica', 7)
            self.drawRightString(w - 1.5*cm, h - 0.75*cm,
                f"Generated: {datetime.datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}")
            self.restoreState()

            # Footer bar
            self.saveState()
            self.setStrokeColor(BRAND_BLUE)
            self.setLineWidth(1.5)
            self.line(1.5*cm, 1.2*cm, w - 1.5*cm, 1.2*cm)
            self.setFillColor(GREY_MED)
            self.setFont('Helvetica', 7)
            self.drawString(1.5*cm, 0.7*cm,
                'CONFIDENTIAL — For professional use only. Verify all content with qualified experts before regulatory submission or clinical use.')
            self.drawRightString(w - 1.5*cm, 0.7*cm, f'Page {self._pageNumber} of {total_pages}')
            self.restoreState()

    return WatermarkCanvas


# ── Style definitions ──────────────────────────────────────────────────────────
def get_styles():
    base = getSampleStyleSheet()
    styles = {
        'title': ParagraphStyle('ReportTitle',
            fontName='Helvetica-Bold', fontSize=22,
            textColor=BRAND_BLUE, spaceAfter=4, alignment=TA_LEFT),
        'subtitle': ParagraphStyle('Subtitle',
            fontName='Helvetica', fontSize=11,
            textColor=GREY_MED, spaceAfter=16, alignment=TA_LEFT),
        'h1': ParagraphStyle('H1',
            fontName='Helvetica-Bold', fontSize=14,
            textColor=BRAND_BLUE, spaceBefore=16, spaceAfter=6,
            borderPad=4, leftIndent=0),
        'h2': ParagraphStyle('H2',
            fontName='Helvetica-Bold', fontSize=11,
            textColor=BRAND_TEAL, spaceBefore=10, spaceAfter=4),
        'h3': ParagraphStyle('H3',
            fontName='Helvetica-Bold', fontSize=10,
            textColor=GREY_DARK, spaceBefore=8, spaceAfter=3),
        'body': ParagraphStyle('Body',
            fontName='Helvetica', fontSize=9.5,
            textColor=GREY_DARK, leading=15, spaceAfter=4,
            alignment=TA_JUSTIFY),
        'bullet': ParagraphStyle('Bullet',
            fontName='Helvetica', fontSize=9.5,
            textColor=GREY_DARK, leading=14, spaceAfter=2,
            leftIndent=16, bulletIndent=6),
        'small': ParagraphStyle('Small',
            fontName='Helvetica', fontSize=8,
            textColor=GREY_MED, spaceAfter=3),
        'badge_label': ParagraphStyle('BadgeLabel',
            fontName='Helvetica-Bold', fontSize=9,
            textColor=WHITE, alignment=TA_CENTER),
        'table_header': ParagraphStyle('TableHeader',
            fontName='Helvetica-Bold', fontSize=9,
            textColor=WHITE),
        'table_body': ParagraphStyle('TableBody',
            fontName='Helvetica', fontSize=9,
            textColor=GREY_DARK, leading=13),
        'reference': ParagraphStyle('Reference',
            fontName='Helvetica', fontSize=8.5,
            textColor=GREY_DARK, leading=13, spaceAfter=3,
            leftIndent=20, firstLineIndent=-20),
    }
    return styles


# ── Helper builders ────────────────────────────────────────────────────────────
def section_header(title, styles):
    """Blue section header with underline."""
    return [
        Paragraph(title, styles['h1']),
        HRFlowable(width='100%', thickness=1.5, color=BRAND_BLUE, spaceAfter=6),
    ]

def sub_header(title, styles):
    return [Paragraph(title, styles['h2'])]

def body_text(text, styles):
    if not text: return []
    return [Paragraph(str(text), styles['body'])]

def bullet_list(items, styles, bullet='•'):
    if not items: return []
    result = []
    for item in items:
        if isinstance(item, dict):
            item = ' | '.join(f"{k.replace('_',' ').title()}: {v}" for k,v in item.items() if v)
        result.append(Paragraph(f"{bullet} {item}", styles['bullet']))
    return result

def kv_table(data_dict, styles, col_widths=None):
    """Two-column key-value table."""
    if not data_dict: return []
    w = A4[0] - 3*cm
    col_widths = col_widths or [5.5*cm, w - 5.5*cm]
    rows = []
    for k, v in data_dict.items():
        if v is None or v == '': continue
        key = k.replace('_',' ').title()
        val = ', '.join(str(i) for i in v) if isinstance(v, list) else str(v)
        rows.append([
            Paragraph(key, styles['h3']),
            Paragraph(val, styles['table_body'])
        ])
    if not rows: return []
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, GREY_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    return [t, Spacer(1, 0.3*cm)]

def data_table(headers, rows_data, styles):
    """Multi-column data table with branded header."""
    if not rows_data: return []
    w = A4[0] - 3*cm
    col_w = w / len(headers)
    header_row = [Paragraph(h, styles['table_header']) for h in headers]
    data_rows = []
    for row in rows_data:
        data_rows.append([Paragraph(str(c or ''), styles['table_body']) for c in row])
    all_rows = [header_row] + data_rows
    t = Table(all_rows, colWidths=[col_w]*len(headers))
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BRAND_BLUE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, GREY_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    return [t, Spacer(1, 0.3*cm)]

def highlight_box(text, styles, bg=None, border=None):
    """Highlighted box for key findings / executive summary."""
    bg = bg or BRAND_LIGHT
    border = border or BRAND_BLUE
    t = Table([[Paragraph(text, styles['body'])]], colWidths=[A4[0]-3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX', (0,0), (-1,-1), 1.5, border),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    return [t, Spacer(1, 0.3*cm)]


# ── MASTER PDF GENERATOR ───────────────────────────────────────────────────────
def generate_report_pdf(output_data: dict, module_type: str, title: str,
                        input_data: dict = None) -> bytes:
    """Professional PDF generation with cover page, watermark, and structured layout."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2.0*cm, bottomMargin=1.8*cm)
    styles = get_styles()
    story = []

    # ── COVER PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 3*cm))

    # Big logo area
    cover_table = Table([[
        Paragraph('<font color="#1a3a5c"><b>AIPBIOS</b></font>', ParagraphStyle(
            'CoverLogo', fontName='Helvetica-Bold', fontSize=36,
            textColor=colors.HexColor('#1a3a5c'), alignment=TA_CENTER))
    ]], colWidths=[A4[0]-3.6*cm])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
        ('BOX', (0,0), (-1,-1), 2, colors.HexColor('#1a3a5c')),
        ('TOPPADDING', (0,0), (-1,-1), 20),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph('AI Intelligence Platform for Healthcare & Pharma',
        ParagraphStyle('CoverSub', fontName='Helvetica', fontSize=12,
            textColor=colors.HexColor('#0d9488'), alignment=TA_CENTER)))
    story.append(Spacer(1, 2*cm))

    # Module badge
    module_labels = {
        'disease_intel': '🔬 Disease Intelligence Report',
        'formulation_intel': '💊 Formulation Intelligence Report',
        'literature_intel': '📚 Literature Intelligence Report',
        'regulatory_intel': '🛡 Regulatory Intelligence Report',
        'patent_intel': '🔎 Patent Intelligence Report',
        'stability_intel': '🌡 Stability Intelligence Report',
        'analytical_intel': '📈 Analytical Intelligence Report',
        'manufacturing_intel': '🏭 Manufacturing Intelligence Report',
        'cost_intel': '💰 Cost Intelligence Report',
        'dossier': '📄 Regulatory Dossier',
        'research_asst': '🎓 Research Intelligence Report',
        'microbiology_intel': '🦠 Microbiological Intelligence Report',
        'statistical_intel': '📊 Statistical Analysis Report',
        'preclinical_intel': '🐭 Preclinical Intelligence Report',
        'clinical_intel': '🏥 Clinical Development Report',
    }
    badge = module_labels.get(module_type, 'Intelligence Report')
    story.append(Paragraph(badge, ParagraphStyle('Badge',
        fontName='Helvetica-Bold', fontSize=16,
        textColor=colors.HexColor('#1a3a5c'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*cm))

    # Report title
    story.append(Paragraph(title, ParagraphStyle('CoverTitle',
        fontName='Helvetica-Bold', fontSize=20,
        textColor=colors.HexColor('#0f172a'), alignment=TA_CENTER,
        spaceAfter=8)))
    story.append(Spacer(1, 2*cm))

    # Meta info table
    subject = ''
    if input_data:
        subject = (input_data.get('disease') or input_data.get('product_name') or
                   input_data.get('title') or input_data.get('topic') or '')

    meta_rows = [
        ['Report Date:', datetime.datetime.utcnow().strftime('%d %B %Y')],
        ['Subject:', subject[:60] if subject else title[:60]],
        ['Platform:', 'AIPBIOS v2.0 — AI Intelligence Platform'],
        ['Classification:', 'CONFIDENTIAL — For Professional Use Only'],
        ['AI Model:', 'GPT-4o (OpenAI)'],
    ]
    if output_data.get('confidence_score'):
        meta_rows.append(['Confidence Score:', f"{output_data['confidence_score']}/10"])

    meta_t = Table(meta_rows, colWidths=[5*cm, A4[0]-3.6*cm-5*cm])
    meta_t.setStyle(TableStyle([
        ('FONT', (0,0), (0,-1), 'Helvetica-Bold', 10),
        ('FONT', (1,0), (1,-1), 'Helvetica', 10),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor('#374151')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 1.5*cm))

    # Disclaimer on cover
    disc = ('This report has been generated by AIPBIOS AI Intelligence Platform using GPT-4o. '
            'All content must be independently verified by qualified pharmaceutical, medical, regulatory, '
            'or legal professionals before use in regulatory submissions, clinical decisions, or commercial activities. '
            'AIPBIOS accepts no liability for decisions made based on this report.')
    story.append(Paragraph(disc, ParagraphStyle('CoverDisc',
        fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#94a3b8'),
        alignment=TA_CENTER, leading=12)))

    story.append(PageBreak())

    # ── EXECUTIVE SUMMARY PAGE ───────────────────────────────────────────────
    exec_sum = (output_data.get('executive_summary') or
                output_data.get('ceo_recommendation') or '')
    if exec_sum:
        story += section_header('Executive Summary', styles)
        story += highlight_box(str(exec_sum), styles)

        # Confidence score bar
        score = output_data.get('confidence_score', 0)
        if score:
            story.append(Spacer(1, 0.3*cm))
            score_color = colors.HexColor('#059669') if score >= 8 else                           colors.HexColor('#d97706') if score >= 6 else                           colors.HexColor('#e11d48')
            # Visual bar
            bar_cells = [['AI Confidence Score:'] +
                         [' '] * 10 + [f'{score}/10']]
            bar_t = Table(bar_cells, colWidths=[3.5*cm] + [0.8*cm]*10 + [1.2*cm])
            style_cmds = [
                ('FONT', (0,0), (0,0), 'Helvetica-Bold', 9),
                ('FONT', (-1,0), (-1,0), 'Helvetica-Bold', 11),
                ('TEXTCOLOR', (-1,0), (-1,0), score_color),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]
            for i in range(1, 11):
                bg = score_color if i <= score else colors.HexColor('#e2e8f0')
                style_cmds.append(('BACKGROUND', (i,0), (i,0), bg))
                style_cmds.append(('ROWHEIGHT', (i,0), (i,0), 0.4*cm))
            bar_t.setStyle(TableStyle(style_cmds))
            story.append(bar_t)
            if output_data.get('confidence_rationale'):
                story.append(Paragraph(f"Confidence note: {output_data['confidence_rationale']}",
                    styles['small']))
        story.append(Spacer(1, 0.5*cm))

    # ── ALL SECTIONS ─────────────────────────────────────────────────────────
    skip = {'executive_summary','ceo_recommendation','confidence_score',
            'confidence_rationale','_demo_mode','_message','report_generated_by',
            'data_quality_note','analyst_recommendation'}
    _render_dict(output_data, story, styles, depth=0, skip_keys=skip)

    # ── ANALYST RECOMMENDATION ───────────────────────────────────────────────
    rec = output_data.get('analyst_recommendation','')
    if rec:
        story.append(Spacer(1, 0.5*cm))
        story += section_header('Strategic Recommendation', styles)
        story += highlight_box(str(rec), styles,
                               bg=colors.HexColor('#f0fdf4'),
                               border=colors.HexColor('#059669'))

    # ── FOOTER DISCLAIMER ────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GREY_MED))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        'Generated by AIPBIOS Intelligence Platform v2.0 | aipbios.onrender.com | '
        'For professional use only. Verify all content with qualified experts.',
        styles['small']))

    # ── BUILD PDF ────────────────────────────────────────────────────────────
    WM = make_watermark_canvas(buf)
    doc.build(story, canvasmaker=WM)
    buf.seek(0)
    return buf.read()

@app.route('/api/v1/intelligence/<module>/reports/<rid>/download/pdf/', methods=['GET'])
@auth_required
def download_report_pdf(module, rid):
    j = q('SELECT * FROM intelligence_jobs WHERE id=?', [rid], one=True)
    if not j: return jsonify({'error': 'Not found'}), 404
    output = json.loads(j['output_payload'] or '{}')
    inp    = json.loads(j['input_payload']   or '{}')
    module_type = j.get('module_type', module)
    name = (inp.get('disease') or inp.get('product_name') or inp.get('title') or
            inp.get('topic') or 'Report')[:50]
    module_labels = {
        'disease_intel':'Disease Intelligence','formulation_intel':'Formulation Intelligence',
        'literature_intel':'Literature Intelligence','regulatory_intel':'Regulatory Intelligence',
        'patent_intel':'Patent Intelligence','stability_intel':'Stability Intelligence',
        'analytical_intel':'Analytical Intelligence','manufacturing_intel':'Manufacturing Intelligence',
        'cost_intel':'Cost Intelligence','dossier':'Regulatory Dossier',
        'research_asst':'Research Intelligence','microbiology_intel':'Microbiological Intelligence',
        'statistical_intel':'Statistical Intelligence','preclinical_intel':'Preclinical Intelligence',
        'clinical_intel':'Clinical Development Intelligence',
    }
    label = module_labels.get(module_type, module_type.replace('_',' ').title())
    title = f"{label} — {name}"
    try:
        pdf_bytes = generate_report_pdf(output, module_type, title, inp)
        resp = make_response(pdf_bytes)
        resp.headers['Content-Type'] = 'application/pdf'
        safe = name.replace(' ','_')[:30]
        resp.headers['Content-Disposition'] = f'attachment; filename="AIPBIOS_{safe}.pdf"'
        return resp
    except Exception as e:
        print(f"PDF error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'PDF failed: {str(e)}'}), 500


@app.route('/health')
def health():
    key = os.environ.get('OPENAI_API_KEY','')
    return jsonify({'status':'healthy','service':'AIPBIOS','version':'2.0.0',
                    'db':'SQLite','ai_mode':'live' if key else 'demo',
                    'modules':13})

@app.route('/api/status')
def api_status():
    key = os.environ.get('OPENAI_API_KEY','')
    return jsonify({'openai_connected':bool(key),
                    'mode':'Live AI — 13 modules supported' if key else 'Demo mode'})

# ── FRONTEND ──────────────────────────────────────────────────────────────────
EMBEDDED_HTML = None
def load_html():
    global EMBEDDED_HTML
    for path in [os.path.join(BASE_DIR,'static','index.html'),
                 os.path.join(BASE_DIR,'index.html'), 'static/index.html', 'index.html']:
        if os.path.exists(path):
            with open(path,'r',encoding='utf-8') as f:
                EMBEDDED_HTML = f.read()
            print(f'HTML loaded: {len(EMBEDDED_HTML)} chars')
            return
    print('WARNING: index.html not found')
load_html()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api/') or path in ('health','api/status'):
        return jsonify({'error':'Not found'}), 404
    static_dir = os.path.join(BASE_DIR, 'static')
    if path and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    if os.path.exists(os.path.join(static_dir, 'index.html')):
        return send_from_directory(static_dir, 'index.html')
    for p in [os.path.join(BASE_DIR,'index.html'),'index.html']:
        if os.path.exists(p):
            with open(p,'r',encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type':'text/html; charset=utf-8'}
    if EMBEDDED_HTML:
        return EMBEDDED_HTML, 200, {'Content-Type':'text/html; charset=utf-8'}
    return '<h1>AIPBIOS v2.0</h1><p>index.html not found</p>', 404

# ── SEED ──────────────────────────────────────────────────────────────────────
def seed():
    db = sqlite3.connect(DB_PATH)
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        db.close(); return
    t = now()
    USERS = [
        (uid(),'admin@aipbios.com',     h('Admin@12345'),    'Platform','Admin',   'superadmin'),
        (uid(),'researcher@aipbios.com',h('Research@12345'),'Dr. Priya','Sharma', 'researcher'),
        (uid(),'student@aipbios.com',   h('Student@12345'), 'Rahul',   'Verma',   'student'),
        (uid(),'industry@aipbios.com',  h('Industry@12345'),'Meera',   'Patel',   'industry_user'),
    ]
    for u in USERS:
        db.execute('INSERT INTO users(id,email,password_hash,first_name,last_name,role,created_at) VALUES(?,?,?,?,?,?,?)', (*u, t))
    res_id = USERS[1][0]
    PROJS = [
        (uid(),'GlycoHerb-DS: Herbal Anti-Diabetic','Ayurvedic anti-diabetic tablet — 5 pre-loaded reports','Type 2 Diabetes Mellitus','Ayurvedic'),
        (uid(),'Dengue Antiviral Research Project','Herbal antiviral for Dengue Fever','Dengue Fever','Herbal'),
        (uid(),'AshwaRelax Sleep Support Capsule','Ayurvedic sleep nutraceutical','Sleep Disorders','Nutraceutical'),
    ]
    for p in PROJS:
        db.execute('INSERT INTO projects(id,title,description,status,disease_area,product_type,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                   (p[0],p[1],p[2],'active',p[3],p[4],res_id,t,t))
    try:
        from server_demo_data import DEMO_REPORTS
        for mod,inp,out,tok in DEMO_REPORTS(PROJS[0][0], res_id, t):
            db.execute('''INSERT INTO intelligence_jobs(id,project_id,created_by,module_type,
                status,input_payload,output_payload,tokens_used,created_at,completed_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (uid(),PROJS[0][0],res_id,mod,'completed',
                 json.dumps(inp),json.dumps(out),tok,t,t))
            db.execute('UPDATE projects SET job_count=job_count+1 WHERE id=?', [PROJS[0][0]])
    except Exception as e:
        print(f'Demo data error: {e}')
    db.commit(); db.close()
    print("✓ Demo data seeded")

# ── STARTUP ───────────────────────────────────────────────────────────────────
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
    mode = 'LIVE AI' if os.environ.get('OPENAI_API_KEY') else 'DEMO'
    print(f"AIPBIOS v2.0 — {mode} MODE — port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
