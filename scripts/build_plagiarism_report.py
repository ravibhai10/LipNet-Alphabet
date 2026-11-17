"""
Generate 12_PlagiarismReport.pdf
Official University Academic Integrity Dossier, Pre-Submission Plagiarism Audit,
Originality Declaration & Verification Checklist for Somaiya Vidyavihar University / KJSSE.
"""

import os
import sys
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette
DARK_NAVY = colors.HexColor("#1A365D")
SOMAIYA_RED = colors.HexColor("#A8201A")
ACCENT_BLUE = colors.HexColor("#2B6CB0")
CHARCOAL = colors.HexColor("#2D3748")
LIGHT_BG = colors.HexColor("#F7FAFC")
BORDER_COLOR = colors.HexColor("#E2E8F0")
SUCCESS_GREEN = colors.HexColor("#2F855A")

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
            self.drawString(54, 750, "KJSSE B.Tech ECE Capstone Project — Academic Integrity & Plagiarism Dossier")
            self.drawRightString(558, 750, "Group ID: 12")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Footer
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Somaiya Vidyavihar University / K. J. Somaiya School of Engineering")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_plagiarism_report():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "submission_package")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "12_PlagiarismReport.pdf")
    
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
        fontSize=18,
        leading=22,
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
        spaceBefore=12,
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
    
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body,
        fontName='Helvetica-Bold'
    )
    
    callout = ParagraphStyle(
        'Callout',
        parent=body,
        fontSize=9,
        leading=12.5,
        textColor=DARK_NAVY
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
    
    # Header Logos
    assets_dir = os.path.join(out_dir, "assets")
    kjsse_logo = os.path.join(assets_dir, "college_logo.png")
    trust_logo = os.path.join(assets_dir, "somaiya_trust_logo.png")
    
    if os.path.exists(kjsse_logo) and os.path.exists(trust_logo):
        logo_data = [
            [Image(kjsse_logo, width=1.5*inch, height=0.6*inch),
             "",
             Image(trust_logo, width=1.5*inch, height=0.6*inch)]
        ]
        logo_tab = Table(logo_data, colWidths=[2*inch, 3*inch, 2*inch])
        logo_tab.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (0,0), 'LEFT'),
            ('ALIGN', (2,0), (2,0), 'RIGHT'),
        ]))
        story.append(logo_tab)
        story.append(Spacer(1, 10))
    
    # Document Title Block
    story.append(Paragraph("K. J. SOMAIYA SCHOOL OF ENGINEERING", ParagraphStyle('Univ', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=15, alignment=1, textColor=SOMAIYA_RED)))
    story.append(Paragraph("SOMAIYA VIDYAVIHAR UNIVERSITY, MUMBAI — 400077", ParagraphStyle('Dept', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=13, alignment=1, textColor=CHARCOAL)))
    story.append(Paragraph("DEPARTMENT OF ELECTRONICS AND COMPUTER ENGINEERING", ParagraphStyle('Dept2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, alignment=1, textColor=DARK_NAVY, spaceAfter=8)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SOMAIYA_RED, spaceAfter=10))
    
    story.append(Paragraph("ACADEMIC INTEGRITY & ORIGINALITY AUDIT DOSSIER", title_style))
    story.append(Paragraph("FINAL YEAR B.TECH CAPSTONE PROJECT — ACADEMIC YEAR 2026–2027", sub_title))
    
    # Project Info Box
    meta_data = [
        [Paragraph("<b>Project Title:</b>", table_cell_bold), Paragraph("Context-Aware Lip Reading and Speech Generation Using Deep Learning", table_cell)],
        [Paragraph("<b>Group ID:</b>", table_cell_bold), Paragraph("Group 12", table_cell)],
        [Paragraph("<b>Candidate 1:</b>", table_cell_bold), Paragraph("Ravikumar Gupta (Roll No: 16014023042)", table_cell)],
        [Paragraph("<b>Candidate 2:</b>", table_cell_bold), Paragraph("Omkar Gupta (Roll No: 16014023035)", table_cell)],
        [Paragraph("<b>Project Guide:</b>", table_cell_bold), Paragraph("Prof. Umang Patel", table_cell)],
        [Paragraph("<b>Target Document:</b>", table_cell_bold), Paragraph("12_ResearchPaper.pdf / 12_WhiteBook.pdf", table_cell)],
        [Paragraph("<b>Submission Date:</b>", table_cell_bold), Paragraph(datetime.now().strftime("%d October %Y"), table_cell)],
    ]
    meta_table = Table(meta_data, colWidths=[1.5*inch, 5.5*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))
    
    # Section 1: Candidate Declaration of Originality
    story.append(Paragraph("1. Candidate Declaration of Originality", h1))
    story.append(Paragraph(
        "We hereby declare that this capstone project report titled <i>'Context-Aware Lip Reading and Speech Generation Using Deep Learning'</i> "
        "and its accompanying research paper, technical manual, and software implementation represent authentic, original scholarly work "
        "conducted by Group 12 under the guidance of Prof. Umang Patel. Any external algorithmic libraries, datasets, mathematical definitions, "
        "and foundational frameworks utilized have been exhaustively cited and attributed in accordance with IEEE formatting and Somaiya Vidyavihar University guidelines.",
        body
    ))
    story.append(Paragraph(
        "We affirm that no portion of this technical report or codebase has been plagiarized, fabricated, or submitted for any other degree or diploma at this or any other educational institution.",
        body
    ))
    story.append(Spacer(1, 8))
    
    # Section 2: Attribution of Foundational Baseline & Open-Source Works
    story.append(Paragraph("2. Foundational Baselines & Open-Source Attribution", h1))
    story.append(Paragraph(
        "In compliance with Somaiya Vidyavihar University academic ethics, all external datasets, architectures, and open-source models utilized in this project are explicitly itemized below:",
        body
    ))
    
    sources = [
        [Paragraph("<b>Item / Resource</b>", table_cell_bold), Paragraph("<b>Originating Author / Entity</b>", table_cell_bold), Paragraph("<b>Role in Project</b>", table_cell_bold), Paragraph("<b>License / Access</b>", table_cell_bold)],
        [Paragraph("GRID Corpus", table_cell), Paragraph("Cooke, Barker, Cunningham, & Shao (2006)", table_cell), Paragraph("Audiovisual speech dataset (33 speakers, 32,891 clips utilized)", table_cell), Paragraph("Open Academic Research", table_cell)],
        [Paragraph("LipNet Baseline", table_cell), Paragraph("Assael, Shillingford, Whiteson, & de Freitas (2016)", table_cell), Paragraph("Conceptual 3D-CNN + BiGRU architecture inspiration", table_cell), Paragraph("Apache 2.0 / Academic", table_cell)],
        [Paragraph("MediaPipe FaceMesh", table_cell), Paragraph("Lugaresi, Tang, et al. / Google LLC", table_cell), Paragraph("Landmark tracking for lip ROI extraction (40-point lip mesh)", table_cell), Paragraph("Apache 2.0", table_cell)],
        [Paragraph("PyTorch Framework", table_cell), Paragraph("Paszke et al. / Meta AI Research", table_cell), Paragraph("Deep learning computation & tensor runtime engine", table_cell), Paragraph("Modified BSD", table_cell)],
        [Paragraph("FastAPI Engine", table_cell), Paragraph("Tiangolo / Sebastián Ramírez", table_cell), Paragraph("Asynchronous REST microservice inference backend", table_cell), Paragraph("MIT License", table_cell)]
    ]
    t_sources = Table(sources, colWidths=[1.4*inch, 1.8*inch, 2.5*inch, 1.3*inch])
    t_sources.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_sources)
    story.append(Spacer(1, 10))

    # Section 3: Summary of Group 12 Original Engineering Contributions
    story.append(Paragraph("3. Summary of Group 12 Original Contributions", h1))
    story.append(Paragraph(
        "The following aspects of the project constitute original intellectual and engineering contributions developed independently by the students:",
        body
    ))
    
    contribs = [
        "<b>Architectural Normalization for Visual Speech:</b> Diagnosis and resolution of inter-speaker pixel luminance bias ($\pm 10$ intensity units) via embedded <i>Dynamic Instance Normalization</i> ($x \leftarrow (x-\mu)/\sigma$), preventing majority-class collapse across unseen speakers.",
        "<b>Rigorous Generalization Protocol:</b> Establishing a strict speaker-independent evaluation benchmark on held-out subjects (s30–s34) achieving 34.38% Top-1 accuracy and 62.50% Top-2 accuracy across homophenous viseme clusters without data leakage.",
        "<b>Fail-Closed CTC Continuous Decoding Policy:</b> Designing an honest dual-track deployment protocol that exposes production-ready isolated word recognition while failing closed (`RuntimeError`) on unverified continuous decoders rather than returning dummy hallucinated sequences.",
        "<b>Full-Stack Real-Time Browser Microservice:</b> Complete engineering of an asynchronous FastAPI inference server operating at 59.2 ms CPU latency paired with a reactive frontend featuring live ROI stabilization and viseme confidence distributions."
    ]
    for c in contribs:
        story.append(Paragraph(f"• {c}", body))
    story.append(Spacer(1, 10))
    
    # Section 4: University Plagiarism Pre-Submission Check & Verification
    story.append(Paragraph("4. Institutional Similarity & Verification Checklist", h1))
    story.append(Paragraph(
        "To comply with university submission guidelines regarding anti-plagiarism verification (Turnitin / DrillBit / Urkund), "
        "the following parameters and filter configurations must be applied during guide verification:",
        body
    ))
    
    checks = [
        [Paragraph("<b>Verification Criterion</b>", table_cell_bold), Paragraph("<b>Institutional Guideline</b>", table_cell_bold), Paragraph("<b>Group 12 Status</b>", table_cell_bold)],
        [Paragraph("Acceptable Similarity Threshold", table_cell), Paragraph("Overall similarity index &lt; 15% excluding bibliography", table_cell), Paragraph("<font color='#2F855A'><b>NOT ASSESSED ? institutional report required</b></font>", table_cell)],
        [Paragraph("Filter: Exclude Bibliography / References", table_cell), Paragraph("Standard academic references must be excluded from match", table_cell), Paragraph("TO BE VERIFIED", table_cell)],
        [Paragraph("Filter: Exclude Small Matches", table_cell), Paragraph("Matches &lt; 8 consecutive words or &lt; 1% to be filtered", table_cell), Paragraph("TO BE VERIFIED", table_cell)],
        [Paragraph("Filter: Exclude Quoted Material", table_cell), Paragraph("Properly enclosed citations to be excluded", table_cell), Paragraph("TO BE VERIFIED", table_cell)],
        [Paragraph("Verification File Format", table_cell), Paragraph("Single compiled text-searchable PDF (12_ResearchPaper.pdf)", table_cell), Paragraph("VERIFIED (Native PDF)", table_cell)]
    ]
    t_checks = Table(checks, colWidths=[2.2*inch, 3.2*inch, 1.6*inch])
    t_checks.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_checks)
    story.append(Spacer(1, 14))

    # Section 5: Step-by-Step Portal Upload Guide for Project Guide
    story.append(Paragraph("5. Step-by-Step Portal Submission Instructions", h1))
    story.append(Paragraph(
        "The project guide or department coordinator may upload the research paper to the institutional Turnitin/DrillBit account as follows:",
        body
    ))
    story.append(Paragraph(
        "1. Log into the official Somaiya Vidyavihar University Turnitin/DrillBit instructor portal.<br/>"
        "2. Select the active class: <i>B.Tech ECE Final Year Projects 2026-2027</i>.<br/>"
        "3. Create or open submission folder: <b>Group 12 — Context-Aware Lip Reading</b>.<br/>"
        "4. Upload document: <code>12_ResearchPaper.pdf</code>.<br/>"
        "5. Apply standard exclusions: <i>Exclude Bibliography: ON</i>, <i>Exclude Quotes: ON</i>, <i>Exclude matches &lt; 8 words: ON</i>.<br/>"
        "6. Once generated, download the official Digital Receipt & Originality Report and append to this dossier.",
        callout
    ))
    story.append(Spacer(1, 15))
    
    # Section 6: Official Sign-Off Block
    story.append(KeepTogether([
        Paragraph("6. Academic Integrity Endorsement & Sign-Off", h1),
        Spacer(1, 10),
        Table([
            [
                Paragraph("<b>Candidate:</b><br/><br/><br/>_______________________<br/><b>Ravikumar Gupta</b><br/>Roll No: 16014023042<br/>Date: 31/10/2026", table_cell),
                Paragraph("<b>Candidate:</b><br/><br/><br/>_______________________<br/><b>Omkar Gupta</b><br/>Roll No: 16014023035<br/>Date: 31/10/2026", table_cell),
                Paragraph("<b>Project Guide:</b><br/><br/><br/>_______________________<br/><b>Prof. Umang Patel</b><br/>Dept. of ECE, KJSSE<br/>Date: 31/10/2026", table_cell),
            ]
        ], colWidths=[2.3*inch, 2.3*inch, 2.4*inch], style=[
            ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ])
    ]))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [SUCCESS] Generated Plagiarism Report PDF: {pdf_path}")

if __name__ == "__main__":
    build_plagiarism_report()
