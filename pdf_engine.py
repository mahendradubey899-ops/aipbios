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
    """
    Universal PDF generator for all AIPBIOS modules.
    Renders output_data intelligently based on module_type.
    Returns PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )
    styles = get_styles()
    story = []

    # ── Cover / title block ──────────────────────────────────────────────────
    story.append(Spacer(1, 0.8*cm))
    # Module badge
    module_labels = {
        'disease_intel': 'Disease Intelligence Report',
        'formulation_intel': 'Formulation Intelligence Report',
        'literature_intel': 'Literature Intelligence Report',
        'regulatory_intel': 'Regulatory Intelligence Report',
        'patent_intel': 'Patent Intelligence Report',
        'stability_intel': 'Stability Intelligence Report',
        'analytical_intel': 'Analytical Intelligence Report',
        'manufacturing_intel': 'Manufacturing Intelligence Report',
        'cost_intel': 'Cost Intelligence Report',
        'dossier': 'Regulatory Dossier',
        'research_asst': 'Research Intelligence Report',
        'microbiology_intel': 'Microbiological Intelligence Report',
        'statistical_intel': 'Statistical Analysis Report',
    }
    badge_text = module_labels.get(module_type, 'Intelligence Report')
    badge_table = Table([[Paragraph(badge_text, styles['badge_label'])]],
                        colWidths=[8*cm])
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BRAND_TEAL),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(title, styles['title']))

    if input_data:
        disease = input_data.get('disease', input_data.get('topic', input_data.get('product_name', '')))
        if disease:
            story.append(Paragraph(f"Subject: {disease}", styles['subtitle']))

    story.append(HRFlowable(width='100%', thickness=2, color=BRAND_BLUE, spaceAfter=12))
    story.append(Spacer(1, 0.3*cm))

    # ── Executive Summary ────────────────────────────────────────────────────
    exec_sum = output_data.get('executive_summary', '')
    if exec_sum:
        story += section_header('Executive Summary', styles)
        story += highlight_box(str(exec_sum), styles)

    # ── Render all sections ──────────────────────────────────────────────────
    _render_dict(output_data, story, styles, depth=0, skip_keys={'executive_summary'})

    # ── Disclaimer ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GREY_MED))
    story.append(Spacer(1, 0.2*cm))
    disclaimer = (
        "DISCLAIMER: This report has been generated by AIPBIOS AI Intelligence Platform. "
        "All information is provided for research and informational purposes only. "
        "Content must be verified by qualified pharmaceutical, medical, regulatory, or legal professionals "
        "before use in regulatory submissions, clinical decisions, or commercial activities. "
        "AIPBIOS and its affiliates accept no liability for decisions made based on this report."
    )
    story.append(Paragraph(disclaimer, styles['small']))

    # ── Build ────────────────────────────────────────────────────────────────
    WM = make_watermark_canvas(buf)
    doc.build(story, canvasmaker=WM)
    buf.seek(0)
    return buf.read()


def _render_dict(data, story, styles, depth=0, skip_keys=None):
    """Recursively render a dict/list into PDF story elements."""
    if skip_keys is None: skip_keys = set()
    if not isinstance(data, dict): return

    for key, value in data.items():
        if key in skip_keys: continue
        if key.startswith('_'): continue
        if value is None or value == '' or value == [] or value == {}: continue

        label = key.replace('_', ' ').replace('-', ' ').title()

        if depth == 0:
            story += section_header(label, styles)
        elif depth == 1:
            story += sub_header(label, styles)
        else:
            story.append(Paragraph(f"<b>{label}</b>", styles['h3']))

        if isinstance(value, str):
            story += body_text(value, styles)

        elif isinstance(value, (int, float)):
            story += body_text(str(value), styles)

        elif isinstance(value, list):
            if len(value) == 0: continue
            first = value[0]
            if isinstance(first, str):
                story += bullet_list(value, styles)
            elif isinstance(first, dict):
                # Check if it's a simple list of dicts (table-able)
                keys = list(first.keys())
                if len(keys) <= 5 and all(isinstance(first.get(k,''), (str,int,float)) for k in keys):
                    # Render as table
                    headers = [k.replace('_',' ').title() for k in keys]
                    rows = [[str(item.get(k,'')) for k in keys] for item in value]
                    story += data_table(headers, rows, styles)
                else:
                    for i, item in enumerate(value):
                        story.append(Paragraph(f"<b>Item {i+1}</b>", styles['h3']))
                        _render_dict(item, story, styles, depth+1)
                        story.append(Spacer(1, 0.2*cm))
            else:
                story += bullet_list([str(v) for v in value], styles)

        elif isinstance(value, dict):
            # Check if it looks like a KV block
            has_only_scalars = all(isinstance(v, (str, int, float, type(None)))
                                   for v in value.values())
            if has_only_scalars and len(value) <= 10:
                story += kv_table(value, styles)
            else:
                _render_dict(value, story, styles, depth+1)

        story.append(Spacer(1, 0.1*cm))
