"""
Generate 12_TechnicalManual.pdf and submission_package/source/12_TechnicalManual.docx
Complete engineering documentation, architecture diagrams, API specs, setup scripts,
and deployment runbooks for LipNet-AI (Group 12).
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# Palette
DARK_NAVY = colors.HexColor("#1A365D")
SOMAIYA_RED = colors.HexColor("#A8201A")
ACCENT_BLUE = colors.HexColor("#2B6CB0")
CHARCOAL = colors.HexColor("#2D3748")
LIGHT_BG = colors.HexColor("#F7FAFC")
BORDER_COLOR = colors.HexColor("#CBD5E0")
CODE_BG = colors.HexColor("#EDF2F7")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "LipNet-AI — Technical & Developer Manual | Group 12")
            self.drawRightString(558, 750, "KJSSE / Somaiya Vidyavihar University")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Footer
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Engineering Documentation & Architecture Reference")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(pdf_path, assets_dir):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=DARK_NAVY,
        alignment=1,
        spaceAfter=6
    )
    
    sub_title = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=SOMAIYA_RED,
        alignment=1,
        spaceAfter=15
    )
    
    h1 = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=DARK_NAVY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2 = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=ACCENT_BLUE,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=CHARCOAL,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C")
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=CHARCOAL
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=DARK_NAVY
    )

    story = []
    
    # Title & Metadata
    story.append(Paragraph("LIPNET-AI: TECHNICAL MANUAL & ARCHITECTURE REFERENCE", title_style))
    story.append(Paragraph("SYSTEM SPECIFICATION, INFERENCE APIS, DEPLOYMENT RUNBOOK & AUDIT PROTOCOLS", sub_title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SOMAIYA_RED, spaceAfter=10))
    
    meta_box = [
        [Paragraph("<b>Project:</b>", table_cell_bold), Paragraph("Context-Aware Lip Reading and Speech Generation Using Deep Learning", table_cell),
         Paragraph("<b>Group ID:</b>", table_cell_bold), Paragraph("Group 12 (B.Tech ECE 2026-27)", table_cell)],
        [Paragraph("<b>Authors:</b>", table_cell_bold), Paragraph("Ravikumar Gupta (16014023042), Omkar Gupta (16014023035)", table_cell),
         Paragraph("<b>Mentor:</b>", table_cell_bold), Paragraph("Prof. Umang Patel", table_cell)],
        [Paragraph("<b>Target Stack:</b>", table_cell_bold), Paragraph("FastAPI, PyTorch 2.x, Next.js / React 19, MediaPipe", table_cell),
         Paragraph("<b>Host OS:</b>", table_cell_bold), Paragraph("Windows 11 / Linux Ubuntu 22.04 LTS (x86_64)", table_cell)]
    ]
    t_meta = Table(meta_box, colWidths=[1.1*inch, 2.5*inch, 1.1*inch, 2.3*inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))
    
    # Section 1: System Architecture & Subsystem Diagram
    story.append(Paragraph("1. System Architecture Overview", h1))
    story.append(Paragraph(
        "LipNet-AI is an end-to-end visual speech recognition system engineered as a decoupled client-server microservice. "
        "The architecture bifurcates edge face tracking from server-side deep spatio-temporal tensor inference, minimizing bandwidth "
        "and eliminating heavy GPU client dependencies.",
        body
    ))
    
    arch_img = os.path.join(assets_dir, "system_architecture.png")
    if os.path.exists(arch_img):
        story.append(Spacer(1, 4))
        story.append(Image(arch_img, width=6.5*inch, height=2.4*inch))
        story.append(Paragraph("<font size='7.5'><b>Figure 1:</b> Decoupled LipNet-AI End-to-End System Architecture.</font>", ParagraphStyle('Cap', parent=body, alignment=1, spaceAfter=8)))
    
    arch_table = [
        [Paragraph("<b>Component</b>", table_cell_bold), Paragraph("<b>Technology</b>", table_cell_bold), Paragraph("<b>Key Responsibilities</b>", table_cell_bold)],
        [Paragraph("Edge Client (Web Frontend)", table_cell), Paragraph("Next.js 14, React 19, MediaPipe FaceMesh", table_cell), Paragraph("WebRTC 30 FPS webcam capture, 40-point lip bounding box localization, ROI stabilization, base64 JPEG serialization.", table_cell)],
        [Paragraph("Inference Microservice", table_cell), Paragraph("FastAPI, Uvicorn, Python 3.10", table_cell), Paragraph("Asynchronous REST gateway, validation, CORS middleware, base64 batch decoding, PyTorch model invocation, 59.2 ms CPU inference.", table_cell)],
        [Paragraph("Track A Model Engine", table_cell), Paragraph("WordVideoNet (3D-CNN + GRU)", table_cell), Paragraph("Dynamic instance normalization, spatio-temporal feature extraction, 4-class softmax probability distribution (bin, lay, place, set).", table_cell)],
        [Paragraph("Track B Safety Engine", table_cell), Paragraph("Sequence CTC Gatekeeper", table_cell), Paragraph("Fail-closed security barrier halting hallucinated continuous text generation on unverified weights via descriptive RuntimeError.", table_cell)]
    ]
    t_arch = Table(arch_table, colWidths=[1.8*inch, 1.8*inch, 3.4*inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))
    
    # Section 2: Hardware & Software Prerequisites
    story.append(Paragraph("2. System Requirements & Environment Specification", h1))
    story.append(Paragraph(
        "<b>Minimum Hardware Specifications:</b><br/>"
        "• Processor: Intel Core i5 (8th Gen+) or AMD Ryzen 5 (4 physical cores minimum).<br/>"
        "• Memory: 8 GB RAM minimum (16 GB recommended for batch training).<br/>"
        "• Video Input: 720p / 1080p WebRTC compatible webcam capable of stable 25–30 FPS.<br/>"
        "• Storage: 5 GB free disk space for runtime dependencies and model checkpoints.<br/>"
        "<b>Software & Library Specifications:</b><br/>"
        "• OS: Microsoft Windows 10/11 x64 or Linux (Ubuntu 20.04/22.04 LTS).<br/>"
        "• Python Runtime: Python 3.10.x or 3.11.x (with active virtual environment).<br/>"
        "• Node.js Engine: Node.js v18.x or v20.x LTS with npm 9.x+.<br/>"
        "• Core Python Packages: `torch>=2.0.0`, `torchvision>=0.15.0`, `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`.",
        body
    ))
    story.append(Spacer(1, 8))

    # Section 3: Installation & Service Runbook
    story.append(Paragraph("3. Step-by-Step Installation & Deployment Runbook", h1))
    story.append(Paragraph(
        "Follow this sequential guide to configure the environment, install dependencies, and launch both backend and frontend services on Windows:",
        body
    ))
    
    commands = (
        "# 1. Clone repository & navigate to root\n"
        "cd LipNet-Alphabet-main\\LipNet-Alphabet-main\n\n"
        "# 2. Provision and activate isolated Python virtual environment\n"
        "python -m venv .venv\n"
        ".\\.venv\\Scripts\\activate\n\n"
        "# 3. Install core PyTorch & backend dependencies\n"
        "pip install --upgrade pip\n"
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        "pip install fastapi uvicorn opencv-python pydantic reportlab python-docx python-pptx\n\n"
        "# 4. Launch FastAPI Inference Microservice (runs on port 8000)\n"
        "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload\n\n"
        "# 5. In a separate terminal, install and launch Web Frontend\n"
        "npm install\n"
        "npm run build\n"
        "npm run start  # Serves production UI on http://localhost:3000"
    )
    cmd_table = Table([[Paragraph(f"<pre>{commands}</pre>", code_style)]], colWidths=[7.0*inch])
    cmd_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(cmd_table)
    story.append(Spacer(1, 10))

    # Section 4: Deep Learning Model Architecture
    story.append(Paragraph("4. WordVideoNet Architecture & Mathematical Formulation", h1))
    story.append(Paragraph(
        "<b>Tensor Input Specification:</b> <code>[Batch_Size, 1, 24, 64, 96]</code> representing a grayscale sequence of 24 normalized frames cropped to 64 (height) × 96 (width) pixels.<br/>"
        "<b>Mathematical Normalization Layer:</b> Before 3D convolution, the tensor undergoes channel-wise dynamic instance normalization:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>x̂ = (x - μ_x) / (σ_x + 1e-5)</b><br/>"
        "This dynamic shift centers mean illumination per clip to zero with unit variance, neutralizing inter-speaker skin tone and ambient room lighting differences.<br/>"
        "<b>Spatio-Temporal Pipeline:</b><br/>"
        "1. <b>Conv3D Block 1:</b> 1 → 32 filters, 3×3×3 kernel, padding=1, BatchNorm3d, ReLU, MaxPool3d(1, 2, 2) → <code>[B, 32, 24, 32, 48]</code>.<br/>"
        "2. <b>Conv3D Block 2:</b> 32 → 64 filters, 3×3×3 kernel, padding=1, BatchNorm3d, ReLU, MaxPool3d(1, 2, 2) → <code>[B, 64, 24, 16, 24]</code>.<br/>"
        "3. <b>Conv3D Block 3:</b> 64 → 128 filters, 3×3×3 kernel, padding=1, BatchNorm3d, ReLU, MaxPool3d(2, 2, 2) → <code>[B, 128, 12, 8, 12]</code>.<br/>"
        "4. <b>Spatial Pooling:</b> AdaptiveAvgPool3d(12, 1, 1) flattening spatial dims → <code>[B, 12, 128]</code> sequence.<br/>"
        "5. <b>Temporal BiGRU:</b> 2-layer Bidirectional GRU (hidden_size=64, total 128) with dropout=0.3 → <code>[B, 12, 128]</code>.<br/>"
        "6. <b>Classification Head:</b> Temporal average pooling across 12 frames → Linear(128, 4) producing unnormalized logits.",
        body
    ))
    story.append(Spacer(1, 8))

    # Section 5: API Endpoint Reference & Contracts
    story.append(Paragraph("5. RESTful API Endpoint Reference", h1))
    story.append(Paragraph(
        "The backend exposes four core REST endpoints adhering strictly to OpenAPI 3.0 schemas:",
        body
    ))
    
    api_table = [
        [Paragraph("<b>Endpoint & Method</b>", table_cell_bold), Paragraph("<b>Parameters / Body</b>", table_cell_bold), Paragraph("<b>Response Format & Status</b>", table_cell_bold)],
        [Paragraph("<code>GET /api/models</code>", table_cell), Paragraph("None", table_cell), Paragraph("<code>{\"models\": [...], \"active_mode\": \"word\"}</code> (HTTP 200)", table_cell)],
        [Paragraph("<code>GET /api/word/health</code>", table_cell), Paragraph("None", table_cell), Paragraph("<code>{\"status\": \"healthy\", \"device\": \"cpu\", \"vocab\": [\"bin\",\"lay\",\"place\",\"set\"]}</code> (HTTP 200)", table_cell)],
        [Paragraph("<code>POST /api/predict/word</code>", table_cell), Paragraph("<code>{\"frames\": [\"data:image/jpeg;base64,...\"]}</code><br/>(24 base64 strings)", table_cell), Paragraph("<code>{\"prediction\": \"place\", \"confidence\": 0.642, \"probabilities\": {...}, \"latency_ms\": 59.2}</code> (HTTP 200)", table_cell)],
        [Paragraph("<code>POST /api/predict/sequence</code>", table_cell), Paragraph("Continuous video frames", table_cell), Paragraph("<code>HTTP 503 / 500 RuntimeError:</code> Fail-closed contract halting unverified continuous decoder weights.", table_cell)]
    ]
    t_api = Table(api_table, colWidths=[1.8*inch, 2.4*inch, 2.8*inch])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_api)
    story.append(Spacer(1, 10))

    # Section 6: Unit Testing & Verification Runbook
    story.append(Paragraph("6. Verification & Automated Test Suite Execution", h1))
    story.append(Paragraph(
        "To verify model integrity, regression prevention, and contract compliance across both pipeline tiers, execute the pytest suite:",
        body
    ))
    
    test_run = (
        "# Execute complete automated verification suite (32 tests)\n"
        ".\\.venv\\Scripts\\python.exe -m pytest tests/test_browser_inference.py tests/test_word_recognition.py -v\n\n"
        "# Verify WordVideoNet accuracy and confusion matrix generation\n"
        ".\\.venv\\Scripts\\python.exe scripts/compare_word_accuracy.py"
    )
    t_test = Table([[Paragraph(f"<pre>{test_run}</pre>", code_style)]], colWidths=[7.0*inch])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_test)
    story.append(Spacer(1, 10))
    
    # Section 7: Troubleshooting Guide
    story.append(Paragraph("7. Developer Troubleshooting & Diagnostic Reference", h1))
    troubles = [
        [Paragraph("<b>Symptom / Error</b>", table_cell_bold), Paragraph("<b>Root Cause</b>", table_cell_bold), Paragraph("<b>Resolution Action</b>", table_cell_bold)],
        [Paragraph("<code>RuntimeError: Continuous recognition unavailable</code>", table_cell), Paragraph("Safety contract fails closed on Track B when decoder is unverified.", table_cell), Paragraph("Expected behavior. Switch UI mode to Track A (Isolated Word) for active inference.", table_cell)],
        [Paragraph("Inference timeout or frame buffer mismatch", table_cell), Paragraph("Browser sent &lt; 24 frames or invalid base64 encoding.", table_cell), Paragraph("Ensure camera delivers continuous 25+ FPS; frontend requires exactly 24 frames per word burst.", table_cell)],
        [Paragraph("High false prediction on bilabial plosives (/p/ vs /b/)", table_cell), Paragraph("Natural visual homopheny between 'place' and 'bin'.", table_cell), Paragraph("Inspect Top-2 predictions; WordVideoNet captures the correct viseme cluster with 62.50% accuracy.", table_cell)],
        [Paragraph("<code>ModuleNotFoundError: No module named 'torch'</code>", table_cell), Paragraph("Terminal executing without active virtual environment.", table_cell), Paragraph("Activate environment via <code>.\\.venv\\Scripts\\activate</code> prior to running server.", table_cell)]
    ]
    t_tr = Table(troubles, colWidths=[2.2*inch, 2.3*inch, 2.5*inch])
    t_tr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_tr)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [SUCCESS] Generated Technical Manual PDF: {pdf_path}")

def build_docx(docx_path):
    doc = Document()
    
    # Page Setup
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("LipNet-AI: Technical & Developer Manual")
    r_title.bold = True
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("System Architecture, Endpoints, Runbooks & Diagnostic Standards | Group 12")
    r_sub.bold = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0xA8, 0x20, 0x1A)
    
    doc.add_paragraph(
        "Institution: K. J. Somaiya School of Engineering, Somaiya Vidyavihar University\n"
        "Department: Department of Electronics and Computer Engineering\n"
        "Project: Context-Aware Lip Reading and Speech Generation Using Deep Learning\n"
        "Team: Ravikumar Gupta (16014023042), Omkar Gupta (16014023035) | Guide: Prof. Umang Patel"
    )
    doc.add_heading("1. System Architecture Overview", level=1)
    doc.add_paragraph(
        "LipNet-AI decouples edge client visual capture from deep spatio-temporal neural inference. "
        "The React/Next.js frontend runs MediaPipe FaceMesh in browser WebAssembly to extract and crop 40-point lip regions, "
        "packaging 24 sequential frames into base64 payloads transmitted to an asynchronous FastAPI REST backend. "
        "The backend routes frames to WordVideoNet (Track A) or the CTC Safety Gatekeeper (Track B)."
    )
    
    doc.add_heading("2. Deployment Runbook", level=1)
    doc.add_paragraph(
        "Step 1: cd LipNet-Alphabet-main\\LipNet-Alphabet-main\n"
        "Step 2: python -m venv .venv && .\\.venv\\Scripts\\activate\n"
        "Step 3: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        "Step 4: pip install fastapi uvicorn opencv-python pydantic\n"
        "Step 5: uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload\n"
        "Step 6: npm install && npm run build && npm run start"
    )
    
    doc.add_heading("3. API Specifications", level=1)
    doc.add_paragraph(
        "• GET /api/models: Returns list of available inference models and active configurations.\n"
        "• GET /api/word/health: Returns model operational status, execution device, and supported vocabulary.\n"
        "• POST /api/predict/word: Receives 24 base64 frames, applies dynamic instance normalization, runs 3D-CNN+GRU, returns Top-1 prediction, confidence score, and full distribution.\n"
        "• POST /api/predict/sequence: Evaluates continuous video sequence against Track B CTC decoder; fails closed if unverified."
    )
    
    doc.add_heading("4. Mathematical Formulation & Normalization", level=1)
    doc.add_paragraph(
        "To eliminate inter-speaker lighting discrepancies, the model embeds dynamic instance normalization directly into its PyTorch forward pass: "
        "x̂ = (x - μ_x) / (σ_x + 1e-5). This shifts mean clip luminance to 0 with unit variance, resolving majority-class collapse across unseen test speakers."
    )
    
    doc.save(docx_path)
    print(f"  [SUCCESS] Generated Technical Manual DOCX: {docx_path}")

if __name__ == "__main__":
    pkg_dir = os.path.join(os.path.dirname(__file__), "..", "submission_package")
    assets = os.path.join(pkg_dir, "assets")
    source_dir = os.path.join(pkg_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    pdf_out = os.path.join(pkg_dir, "12_TechnicalManual.pdf")
    docx_out = os.path.join(source_dir, "12_TechnicalManual.docx")
    
    build_pdf(pdf_out, assets)
    build_docx(docx_out)
