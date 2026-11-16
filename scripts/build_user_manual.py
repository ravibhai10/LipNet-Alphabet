"""
Generate 12_UserManual.pdf and submission_package/source/12_UserManual.docx
Non-technical operating manual, UI walkthrough, user best practices, and FAQ for LipNet-AI (Group 12).
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from docx import Document
from docx.shared import Inches, Pt, RGBColor

# Palette
DARK_NAVY = colors.HexColor("#1A365D")
SOMAIYA_RED = colors.HexColor("#A8201A")
ACCENT_BLUE = colors.HexColor("#2B6CB0")
CHARCOAL = colors.HexColor("#2D3748")
LIGHT_BG = colors.HexColor("#F7FAFC")
BORDER_COLOR = colors.HexColor("#CBD5E0")
HIGHLIGHT_BG = colors.HexColor("#EDF2F7")

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
            self.drawString(54, 750, "LipNet-AI — End-User Operating Manual | Group 12")
            self.drawRightString(558, 750, "KJSSE / Somaiya Vidyavihar University")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Footer
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Somaiya Vidyavihar University — Assistive Technology Demonstration Guide")
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
    
    # Header Title
    story.append(Paragraph("LIPNET-AI: END-USER OPERATING MANUAL", title_style))
    story.append(Paragraph("A PRACTICAL GUIDE TO ASSISTIVE LIP READING & REAL-TIME SPEECH GENERATION", sub_title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SOMAIYA_RED, spaceAfter=10))
    
    # Metadata Box
    meta_box = [
        [Paragraph("<b>Project:</b>", table_cell_bold), Paragraph("Context-Aware Lip Reading and Speech Generation Using Deep Learning", table_cell),
         Paragraph("<b>Group ID:</b>", table_cell_bold), Paragraph("Group 12 (B.Tech ECE 2026-27)", table_cell)],
        [Paragraph("<b>Authors:</b>", table_cell_bold), Paragraph("Ravikumar Gupta (16014023042), Omkar Gupta (16014023035)", table_cell),
         Paragraph("<b>Mentor:</b>", table_cell_bold), Paragraph("Prof. Umang Patel", table_cell)],
        [Paragraph("<b>Target Audience:</b>", table_cell_bold), Paragraph("Clinical evaluators, speech clinicians, examiners, end users", table_cell),
         Paragraph("<b>Compatibility:</b>", table_cell_bold), Paragraph("Chrome 110+, Edge 110+, Firefox 115+", table_cell)]
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
    
    # Section 1: Introduction & Purpose
    story.append(Paragraph("1. Introduction & Assistive Purpose", h1))
    story.append(Paragraph(
        "Welcome to <b>LipNet-AI</b>, an artificial intelligence system designed to translate silent speech movements into written text "
        "and synthesized vocal audio. The application is engineered primarily as an assistive communication technology for individuals "
        "recovering from aphonia, laryngeal trauma, or severe vocal cord dysfunction. It also serves as an ambient noise-immune interface "
        "for industrial environments where acoustic speech microphones fail.",
        body
    ))
    story.append(Paragraph(
        "The system operates without requiring any intrusive wearable sensors or microphones. It relies entirely on a standard webcam, "
        "local computer vision face tracking, and deep spatio-temporal neural networks.",
        body
    ))
    story.append(Spacer(1, 8))

    # Section 2: Browser & Camera Setup
    story.append(Paragraph("2. Browser Setup & Camera Permissions", h1))
    story.append(Paragraph(
        "To begin using LipNet-AI, ensure your computer has a connected webcam and an up-to-date modern web browser.",
        body
    ))
    steps = [
        [Paragraph("<b>Step</b>", table_cell_bold), Paragraph("<b>Action Required</b>", table_cell_bold), Paragraph("<b>Visual Confirmation</b>", table_cell_bold)],
        [Paragraph("1. Open Web Portal", table_cell), Paragraph("Launch your browser and navigate to <code>http://localhost:3000</code>.", table_cell), Paragraph("The LipNet-AI dashboard renders with dark slate styling.", table_cell)],
        [Paragraph("2. Allow Camera Access", table_cell), Paragraph("When prompted: <i>'http://localhost:3000 wants to use your camera'</i>, click <b>Allow</b>.", table_cell), Paragraph("Webcam LED lights up and live video appears in viewport.", table_cell)],
        [Paragraph("3. Verify Connection", table_cell), Paragraph("Observe the top-right header status indicator.", table_cell), Paragraph("Badge turns green and displays: <b>Backend Connected</b>.", table_cell)],
        [Paragraph("4. Face Framing", table_cell), Paragraph("Sit 50–70 cm directly in front of the camera with your face centered.", table_cell), Paragraph("A blue bounding box dynamically tracks your lips.", table_cell)]
    ]
    t_steps = Table(steps, colWidths=[1.4*inch, 2.8*inch, 2.8*inch])
    t_steps.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 10))

    # Section 3: Step-by-Step Word Recognition Guide
    story.append(Paragraph("3. How to Perform a Word Recognition Prediction", h1))
    story.append(Paragraph(
        "LipNet-AI Track A is calibrated to recognize four core English vocabulary words: <b>bin</b>, <b>lay</b>, <b>place</b>, and <b>set</b>. "
        "Follow these simple instructions to achieve accurate recognition:",
        body
    ))
    
    story.append(Paragraph(
        "<b>1. Select Track A Mode:</b> In the top navigation bar, ensure <b>'Isolated Word Mode'</b> is active.<br/>"
        "<b>2. Prepare Articulation:</b> Choose one of the 4 supported words to speak. Keep your lips closed in a neutral resting position.<br/>"
        "<b>3. Click 'Record Word':</b> Press the red circular <b>Record</b> button or press the <code>Spacebar</code> shortcut.<br/>"
        "<b>4. Speak Naturally:</b> Pronounce the chosen word silently or with a normal voice over a 1.0-second duration. "
        "Do not exaggerate or gape your mouth unnaturally.<br/>"
        "<b>5. View Instant Result:</b> The system analyzes the 24-frame burst in under 60 milliseconds. "
        "The predicted word appears in bold green text accompanied by a confidence percentage (e.g., <i>'PLACE — 78.4% Confidence'</i>).<br/>"
        "<b>6. Speech Playback:</b> Click the <b>Speaker Icon</b> next to the predicted word to hear an instant clear audio synthesis of your utterance.",
        body
    ))
    story.append(Spacer(1, 8))

    # Section 4: Visual Homopheny & Reading Confidence
    story.append(Paragraph("4. Understanding Visual Ambiguity (Homopheny)", h1))
    story.append(Paragraph(
        "In human speech, different spoken sounds often share identical visual lip formations. This linguistic phenomenon is called <b>homopheny</b>. "
        "For example, the consonant sounds <b>/p/</b> (as in <i>place</i>) and <b>/b/</b> (as in <i>bin</i>) are both bilabial plosives: "
        "both require pressing the upper and lower lips tightly together to stop airflow before release.",
        body
    ))
    story.append(Paragraph(
        "Because a silent video camera captures no audio acoustic cues (voicing vs unvoiced vibration), a purely visual classifier "
        "can occasionally predict <i>bin</i> when you mouthed <i>place</i>. "
        "To assist users, LipNet-AI displays a <b>Top-2 Candidate Ranking</b>. When pronouncing <i>place</i> or <i>bin</i>, "
        "the correct word will almost always appear within the top two candidates (scoring <b>62.50% Top-2 accuracy</b>).",
        body
    ))
    story.append(Spacer(1, 8))

    # Section 5: Environmental Tips for High Accuracy
    story.append(Paragraph("5. Best Practices for Optimal Recognition Accuracy", h1))
    
    tips = [
        [Paragraph("<b>Factor</b>", table_cell_bold), Paragraph("<b>Recommended Best Practice</b>", table_cell_bold), Paragraph("<b>What to Avoid</b>", table_cell_bold)],
        [Paragraph("Ambient Lighting", table_cell), Paragraph("Diffuse frontal lighting illuminating both lips evenly.", table_cell), Paragraph("Strong backlighting (windows behind user) or harsh side shadows.", table_cell)],
        [Paragraph("Camera Angle", table_cell), Paragraph("Straight-on, eye-level orientation directly facing the camera.", table_cell), Paragraph("Steep upward or downward angles tilting lips away from lens.", table_cell)],
        [Paragraph("Head Stability", table_cell), Paragraph("Keep head relatively stationary while speaking.", table_cell), Paragraph("Rapid nodding, turning, or covering mouth with hands.", table_cell)],
        [Paragraph("Speech Cadence", table_cell), Paragraph("Articulate at a standard, conversational pace (approx. 1 word/sec).", table_cell), Paragraph("Hyper-fast mumbling or unnaturally exaggerated slow motion.", table_cell)]
    ]
    t_tips = Table(tips, colWidths=[1.4*inch, 2.8*inch, 2.8*inch])
    t_tips.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_tips)
    story.append(Spacer(1, 10))

    # Section 6: Frequently Asked Questions & Troubleshooting
    story.append(Paragraph("6. Frequently Asked Questions & User Support", h1))
    
    faqs = [
        "<b>Q: Why does the system fail to detect my lips?</b><br/>"
        "<i>A:</i> Ensure your room is well lit. If your face is too close (&lt; 30 cm) or too far (&gt; 1.5 m), MediaPipe FaceMesh cannot resolve all 40 lip landmarks. Position yourself at an arm's length.",
        "<b>Q: What does the 'Continuous Sequence Mode — Research Preview' notice mean?</b><br/>"
        "<i>A:</i> Track B (continuous sentence recognition) is actively under research development. To protect against erroneous hallucinated sentences, the backend enforces a fail-closed safety policy. Always use Track A for live demonstrations.",
        "<b>Q: Can I use LipNet-AI while wearing a surgical mask or heavy facial covering?</b><br/>"
        "<i>A:</i> No. The model relies strictly on lip contour, aperture height, and dental viseme geometry. The mouth must be fully unobstructed.",
        "<b>Q: Can I expand the vocabulary beyond the current four words?</b><br/>"
        "<i>A:</i> Yes. Researchers can consult the companion <i>12_TechnicalManual.pdf</i> to train new classes using the provided automated dataset preparation scripts."
    ]
    for q in faqs:
        story.append(Paragraph(q, body))
        story.append(Spacer(1, 4))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [SUCCESS] Generated User Manual PDF: {pdf_path}")

def build_docx(docx_path):
    doc = Document()
    
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("LipNet-AI: End-User Operating Manual")
    r_title.bold = True
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("Guide to Assistive Visual Speech Recognition & Live Operation | Group 12")
    r_sub.bold = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0xA8, 0x20, 0x1A)
    
    doc.add_paragraph(
        "Institution: K. J. Somaiya School of Engineering, Somaiya Vidyavihar University\n"
        "Department: Department of Electronics and Computer Engineering\n"
        "Project: Context-Aware Lip Reading and Speech Generation Using Deep Learning\n"
        "Team: Ravikumar Gupta (16014023042), Omkar Gupta (16014023035) | Guide: Prof. Umang Patel"
    )
    doc.add_heading("1. Purpose & Application", level=1)
    doc.add_paragraph(
        "LipNet-AI empowers voiceless individuals, post-operative speech patients, and workers in high-decibel acoustic noise "
        "to communicate seamlessly by interpreting silent lip movements through an ordinary computer webcam."
    )
    
    doc.add_heading("2. How to Use the System", level=1)
    doc.add_paragraph(
        "1. Open http://localhost:3000 in Google Chrome or Microsoft Edge.\n"
        "2. Click 'Allow' when prompted for webcam access.\n"
        "3. Confirm that the top indicator shows 'Backend Connected' in green.\n"
        "4. Choose one of the 4 supported words: 'bin', 'lay', 'place', or 'set'.\n"
        "5. Click the Record button and pronounce the word naturally over 1 second.\n"
        "6. View the predicted word and audio playback immediately on screen."
    )
    
    doc.add_heading("3. Understanding Homopheny", level=1)
    doc.add_paragraph(
        "Certain consonants such as /p/ and /b/ look identical on the lips. When pronouncing 'place' or 'bin', "
        "check the Top-2 ranked prediction bar if the first prediction does not match your exact intention."
    )
    
    doc.save(docx_path)
    print(f"  [SUCCESS] Generated User Manual DOCX: {docx_path}")

if __name__ == "__main__":
    pkg_dir = os.path.join(os.path.dirname(__file__), "..", "submission_package")
    assets = os.path.join(pkg_dir, "assets")
    source_dir = os.path.join(pkg_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    pdf_out = os.path.join(pkg_dir, "12_UserManual.pdf")
    docx_out = os.path.join(source_dir, "12_UserManual.docx")
    
    build_pdf(pdf_out, assets)
    build_docx(docx_out)
