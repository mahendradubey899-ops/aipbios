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
import zlib as _zlib, base64 as _b64
_HTML_DATA = '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'
EMBEDDED_HTML = _zlib.decompress(_b64.b64decode(_HTML_DATA)).decode('utf-8')
print(f'HTML loaded: {len(EMBEDDED_HTML)} chars')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api/') or path in ('health','api/status'):
        return jsonify({'error':'Not found'}), 404
    # Always use embedded HTML - guaranteed correct version
    return EMBEDDED_HTML, 200, {'Content-Type': 'text/html; charset=utf-8'}

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
