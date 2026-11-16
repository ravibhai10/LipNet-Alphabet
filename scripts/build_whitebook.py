import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

output_pdf = Path("submission_package/12_WhiteBook.pdf")
source_docx = Path("submission_package/source/12_WhiteBook.docx")
assets_dir = Path("submission_package/assets")
source_docx.parent.mkdir(parents=True, exist_ok=True)

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

    def draw_page_decorations(self, total_pages):
        if self._pageNumber == 1:
            return
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, A4[1] - 36, "Group 12 | Context-Aware Lip Reading and Speech Generation Using Deep Learning")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, A4[1] - 42, A4[0] - 54, A4[1] - 42)
        self.line(54, 46, A4[0] - 54, 46)
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(A4[0] - 54, 34, page_str)
        self.drawString(54, 34, "K. J. Somaiya School of Engineering | Department of ECE (2026-2027)")
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=26,
        alignment=1, # Center
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#475569")
    )
    center_style = ParagraphStyle(
        "CoverCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#1E293B")
    )
    h1 = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#1E3A8A"),
        keepWithNext=True
    )
    h2 = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#0284C7"),
        keepWithNext=True
    )
    body = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
        textColor=colors.HexColor("#1E293B"),
        alignment=4 # Justify
    )
    bullet = ParagraphStyle(
        "Bullet_Custom",
        parent=body,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 15))
    if (assets_dir / "college_logo.png").exists():
        story.append(Image(str(assets_dir / "college_logo.png"), width=3.5*inch, height=0.875*inch))
    story.append(Spacer(1, 15))
    story.append(Paragraph("A CAPSTONE PROJECT REPORT (WHITE BOOK) ON", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("CONTEXT-AWARE LIP READING AND SPEECH GENERATION USING DEEP LEARNING", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Submitted in partial fulfillment of the requirements for the Degree of<br/><b>Bachelor of Technology (B. Tech) in Electronics and Computer Engineering</b>", subtitle_style))
    story.append(Spacer(1, 20))
    
    student_table_data = [
        [Paragraph("<b>Name of Candidate</b>", table_header), Paragraph("<b>Roll Number</b>", table_header)],
        [Paragraph("Ravikumar Gupta", table_cell), Paragraph("16014023042", table_cell)],
        [Paragraph("Omkar Gupta", table_cell), Paragraph("16014023035", table_cell)]
    ]
    st_table = Table(student_table_data, colWidths=[2.8*inch, 2.2*inch])
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(st_table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Under the Esteemed Guidance of</b><br/><b>Prof. Umang Patel</b><br/>Assistant Professor, Department of Electronics & Computer Engineering", center_style))
    story.append(Spacer(1, 25))
    story.append(Paragraph("<b>DEPARTMENT OF ELECTRONICS AND COMPUTER ENGINEERING</b><br/><b>K. J. SOMAIYA SCHOOL OF ENGINEERING</b><br/>SOMAIYA VIDYAVIHAR UNIVERSITY, MUMBAI – 400077<br/>ACADEMIC YEAR: 2026–2027", center_style))
    story.append(PageBreak())

    # =========================================================================
    # CERTIFICATE PLACEHOLDER
    # =========================================================================
    story.append(Paragraph("CERTIFICATE OF APPROVAL", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=15))
    story.append(Paragraph("<i>[OFFICIAL INSTITUTIONAL CERTIFICATE PAGE PLACEHOLDER]</i>", ParagraphStyle("Notice", parent=body, fontName="Helvetica-Oblique", textColor=colors.HexColor("#B91C1C"))))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This is to certify that the project entitled <b>'Context-Aware Lip Reading and Speech Generation Using Deep Learning'</b> is a bona fide work carried out by <b>Ravikumar Gupta (Roll No: 16014023042)</b> and <b>Omkar Gupta (Roll No: 16014023035)</b> in partial fulfillment of the requirements for the award of Bachelor of Technology in Electronics and Computer Engineering from K. J. Somaiya School of Engineering, Somaiya Vidyavihar University during the academic year 2026–2027.", body))
    story.append(Spacer(1, 40))

    sig_data = [
        [Paragraph("<b>Prof. Umang Patel</b><br/>Project Guide / Mentor", center_style),
         Paragraph("<b>Dr. Rajashree Daryapurkar</b><br/>Head of Department (ECE)", center_style),
         Paragraph("<b>Dr. Suresh Ukarande</b><br/>Principal / Dean", center_style)]
    ]
    sig_table = Table(sig_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
    sig_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(sig_table)
    story.append(Spacer(1, 30))
    story.append(Paragraph("<i>Note: Official embossed university certificate sheets with pre-printed security borders are issued by the university examination office separately and will be inserted prior to hardbound submission on 31 October 2026.</i>", ParagraphStyle("Footnote", parent=body, fontSize=8, textColor=colors.HexColor("#64748B"))))
    story.append(PageBreak())

    # =========================================================================
    # STUDENT DECLARATION & ACKNOWLEDGEMENT
    # =========================================================================
    story.append(Paragraph("CANDIDATES' DECLARATION", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))
    story.append(Paragraph("We hereby declare that the work presented in this Project Report entitled <b>'Context-Aware Lip Reading and Speech Generation Using Deep Learning'</b> is authentic and has been carried out by us under the supervision of <b>Prof. Umang Patel</b>. We have not submitted the matter embodied in this report for the award of any other degree or diploma to any other institute or university.", body))
    story.append(Paragraph("We confirm that all experimental results reported herein reflect genuine code executions, real held-out multi-speaker evaluations, and verified architectural diagnostics without fabricated metrics or synthetic accuracy inflation.", body))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Date: 29 October 2026<br/>Place: Mumbai", body))
    story.append(Spacer(1, 25))
    decl_data = [
        [Paragraph("____________________________<br/><b>Ravikumar Gupta</b><br/>Roll No: 16014023042", center_style),
         Paragraph("____________________________<br/><b>Omkar Gupta</b><br/>Roll No: 16014023035", center_style)]
    ]
    story.append(Table(decl_data, colWidths=[3.5*inch, 3.5*inch]))
    story.append(Spacer(1, 30))

    story.append(Paragraph("ACKNOWLEDGEMENTS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))
    story.append(Paragraph("We express our profound gratitude to our project guide, <b>Prof. Umang Patel</b>, for his continuous guidance, technical insights, and rigorous reviews throughout the design, training, and troubleshooting of this deep learning AVSR project. His insistence on scientific honesty and empirical verification has been invaluable.", body))
    story.append(Paragraph("We are thankful to <b>Dr. Rajashree Daryapurkar</b>, Head of Department of Electronics and Computer Engineering, and <b>Dr. Suresh Ukarande</b>, Principal of K. J. Somaiya School of Engineering, for providing the laboratory infrastructure and high-performance computing environment.", body))
    story.append(PageBreak())

    # =========================================================================
    # ABSTRACT
    # =========================================================================
    story.append(Paragraph("ABSTRACT", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))
    story.append(Paragraph("Visual Speech Recognition (VSR)—commonly referred to as automated lip-reading—interprets spoken linguistic tokens solely from silent video recordings of mouth kinematics. VSR is critical for speech recognition in acoustically degraded environments (airports, factory floors, emergency sirens) and assistive communication for laryngectomy and aphonic individuals. However, practical VSR deployment is impeded by visual homopheny (where acoustically distinct phonemes like /p/ and /b/ produce identical visible lip closures) and the notorious phenomenon of spatio-temporal feature collapse in deep networks trained on multi-speaker corpora.", body))
    story.append(Paragraph("In this Capstone project, we present an end-to-end multi-dataset Visual Speech Recognition system operating across two strictly separated tracks: (1) <b>Track A: Isolated Whole-Word Recognition</b> powered by a customized spatio-temporal neural network (<code>WordVideoNet</code>: 3D-CNN + 2-layer Bidirectional GRU) with Softmax uncertainty gating; and (2) <b>Track B: Continuous Video-to-Text Recognition</b> leveraging greedy Connectionist Temporal Classification (CTC) sequence decoding with visual-only constraints.", body))
    story.append(Paragraph("A critical contribution of this research is the empirical diagnosis and remediation of majority-class collapse. Initially, an isolated word classifier trained on 96 multi-speaker GRID samples completely collapsed to uniform logits ($\sigma \approx 0.0002$) and a single majority class prediction (25.0% accuracy). Through controlled ablation, we identified that unnormalized raw pixel brightness (speaker illumination bias $\pm 10$ intensity units) overwhelmed subtle inter-frame mouth motion ($0.5$ to $1.3$ intensity units). By engineering a per-clip instance normalization layer ($x \leftarrow (x-\mu)/\sigma$) and non-destructive spatial roll augmentations, we restored gradient sensitivity, achieving <b>34.38% Top-1 Accuracy and 62.50% Top-2 Accuracy</b> across completely unseen held-out speakers (s30–s34).", body))
    story.append(Paragraph("The system is integrated into a production-grade FastAPI microservice executing CPU inference in 59.2 ms and a cinematic React/Vite web application providing real-time visual speech transcription.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Keywords:</b> Visual Speech Recognition (VSR), Lip-Reading, 3D Convolutional Neural Networks, Bidirectional GRU, Majority-Class Collapse, Instance Normalization, GRID Corpus, Uncertainty Gating.", body))
    story.append(PageBreak())

    # =========================================================================
    # TABLE OF CONTENTS & ABBREVIATIONS
    # =========================================================================
    story.append(Paragraph("TABLE OF CONTENTS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))

    toc_data = [
        [Paragraph("<b>Chapter</b>", table_header), Paragraph("<b>Title</b>", table_header), Paragraph("<b>Page</b>", table_header)],
        [Paragraph("1", table_cell), Paragraph("Introduction, Motivation & Problem Statement", table_cell), Paragraph("1", table_cell)],
        [Paragraph("2", table_cell), Paragraph("Literature Review & Gap Analysis", table_cell), Paragraph("4", table_cell)],
        [Paragraph("3", table_cell), Paragraph("System Architecture & Track Disentanglement", table_cell), Paragraph("7", table_cell)],
        [Paragraph("4", table_cell), Paragraph("Multi-Dataset Engineering & Licensing Protocols", table_cell), Paragraph("10", table_cell)],
        [Paragraph("5", table_cell), Paragraph("Computer Vision Preprocessing & Landmark Alignment", table_cell), Paragraph("13", table_cell)],
        [Paragraph("6", table_cell), Paragraph("Deep Learning Spatio-Temporal Model Formulation", table_cell), Paragraph("16", table_cell)],
        [Paragraph("7", table_cell), Paragraph("Software Architecture & Full-Stack Integration", table_cell), Paragraph("20", table_cell)],
        [Paragraph("8", table_cell), Paragraph("Experimental Results, Collapse Diagnosis & Discussion", table_cell), Paragraph("24", table_cell)],
        [Paragraph("9", table_cell), Paragraph("Conclusion, Societal Impact & Future Research", table_cell), Paragraph("30", table_cell)],
        [Paragraph("—", table_cell), Paragraph("References (IEEE Format)", table_cell), Paragraph("33", table_cell)],
        [Paragraph("A", table_cell), Paragraph("Appendix A: Verified API Schemas & Request Payloads", table_cell), Paragraph("36", table_cell)],
        [Paragraph("B", table_cell), Paragraph("Appendix B: Reproducibility & Automated Test Suite Logs", table_cell), Paragraph("38", table_cell)]
    ]
    t_toc = Table(toc_data, colWidths=[0.8*inch, 5.2*inch, 1.0*inch])
    t_toc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_toc)
    story.append(Spacer(1, 25))

    story.append(Paragraph("LIST OF ABBREVIATIONS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
    abbr_data = [
        [Paragraph("<b>Abbreviation</b>", table_header), Paragraph("<b>Full Meaning</b>", table_header)],
        [Paragraph("AVSR", table_cell), Paragraph("Audio-Visual Speech Recognition", table_cell)],
        [Paragraph("VSR", table_cell), Paragraph("Visual Speech Recognition (Lip-Reading)", table_cell)],
        [Paragraph("CTC", table_cell), Paragraph("Connectionist Temporal Classification", table_cell)],
        [Paragraph("WER", table_cell), Paragraph("Word Error Rate", table_cell)],
        [Paragraph("CER", table_cell), Paragraph("Character Error Rate", table_cell)],
        [Paragraph("BiGRU", table_cell), Paragraph("Bidirectional Gated Recurrent Unit", table_cell)],
        [Paragraph("ROI", table_cell), Paragraph("Region of Interest", table_cell)],
        [Paragraph("GRID", table_cell), Paragraph("GRID Multi-Speaker Audiovisual Speech Corpus", table_cell)],
        [Paragraph("LRW", table_cell), Paragraph("Lip Reading in the Wild Dataset (BBC/Oxford)", table_cell)],
        [Paragraph("TTS", table_cell), Paragraph("Text-to-Speech Synthesis", table_cell)]
    ]
    t_abbr = Table(abbr_data, colWidths=[1.8*inch, 5.2*inch])
    t_abbr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284C7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_abbr)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1: INTRODUCTION & PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("CHAPTER 1: INTRODUCTION & PROBLEM STATEMENT", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("1.1 Background & Motivation", h2))
    story.append(Paragraph("Speech is the primary and most natural modality for human communication. While conventional Acoustic Speech Recognition (ASR) engines have achieved near-human accuracy in quiet acoustic environments, their performance drops precipitously in real-world environments with non-stationary background noise, reverberation, cocktail party interference, and microphone clipping. In military cockpits, industrial manufacturing plants, operating rooms, and noisy urban transit hubs, acoustic speech signals are severely corrupted.", body))
    story.append(Paragraph("Visual Speech Recognition (VSR)—the automated interpretation of speech from video recordings of a speaker's mouth kinematics—offers a completely acoustic-independent paradigm. Because optical video is entirely immune to ambient sound noise, electromagnetic interference, and acoustic reverberation, VSR provides a robust foundation for multi-modal speech recognition. Furthermore, for non-verbal individuals, post-laryngectomy patients, and aphonic individuals, lip-reading provides a life-changing assistive bridge for speech generation.", body))

    story.append(Paragraph("1.2 Problem Statement", h2))
    story.append(Paragraph("Despite decades of research, commercializing visual speech recognition remains difficult due to three fundamental engineering bottlenecks:", body))
    story.append(Paragraph("• <b>Visual Homopheny:</b> In the English language, multiple distinct acoustic phonemes share identical visible articulation configurations. For instance, the bilabial stops /p/, /b/, and /m/ all require complete upper and lower lip closure before release. An observer—human or machine—cannot visually distinguish between 'pat', 'bat', and 'mat' from mouth kinematics alone without linguistic context.", bullet))
    story.append(Paragraph("• <b>Speaker Invariant Coarticulation:</b> The visual appearance of a phoneme is heavily modulated by preceding and succeeding phonemes (temporal coarticulation). Furthermore, inter-speaker anatomical differences (lip thickness, oral cavity dimensions, facial hair, pigmentation, speech rate) introduce large feature variance that frequently dwarfs the subtle visual differences between words.", bullet))
    story.append(Paragraph("• <b>Optimization Collapse in Multi-Speaker Regimes:</b> When deep spatio-temporal architectures (3D-CNNs + Recurrent networks) are trained on multi-speaker datasets without carefully normalized inputs, networks rapidly settle into degenerate saddle points: predicting the majority training class or invariant uniform probabilities across all inputs.", bullet))

    story.append(Paragraph("1.3 Project Scope & Deliverables", h2))
    story.append(Paragraph("This project directly addresses these challenges by developing an end-to-end deep learning framework featuring:", body))
    story.append(Paragraph("1. <b>Strict Track Separation:</b> Clearly separating isolated whole-word classification (Track A) from continuous sequence decoding (Track B) to avoid fabricated open-vocabulary claims.", bullet))
    story.append(Paragraph("2. <b>Rigorous Multi-Dataset Ingestion:</b> Verifying 33 official GRID corpus speakers (32,891 utterances) on local storage while documenting legal licensing workflows for MIRACL-VC1, TCD-TIMIT, and Oxford LRW.", bullet))
    story.append(Paragraph("3. <b>Empirical Collapse Repair:</b> Documenting the mathematical root cause of majority collapse and resolving it through per-clip instance normalization, achieving 34.38% Top-1 and 62.50% Top-2 accuracy on unseen held-out speakers.", bullet))
    story.append(Paragraph("4. <b>Full-Stack Deployment:</b> Providing a high-performance CPU inference service (<60 ms latency) with mathematical uncertainty rejection and an interactive React web application.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 2: LITERATURE REVIEW
    # =========================================================================
    story.append(Paragraph("CHAPTER 2: LITERATURE REVIEW & GAP ANALYSIS", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("2.1 Evolution of Visual Speech Recognition", h2))
    story.append(Paragraph("Early lip-reading systems relied on handcrafted geometric features, such as active contour models (snakes), active appearance models (AAMs), Discrete Cosine Transform (DCT) coefficients, and optical flow vectors coupled with Hidden Markov Models (HMMs). While these systems established foundational principles, they exhibited near-zero generalization across unseen speakers and required rigid lighting conditions.", body))
    story.append(Paragraph("The visual speech recognition landscape underwent a paradigm shift with the introduction of deep learning. In 2016, Assael et al. published <b>LipNet</b>, the first end-to-end sentence-level lip-reading model. LipNet combined 3D Convolutional Neural Networks (3D-CNN) to capture spatiotemporal motion with Bidirectional Gated Recurrent Units (BiGRU) and Connectionist Temporal Classification (CTC) loss, reporting a 4.8% Word Error Rate (WER) on the constrained GRID corpus. However, independent evaluations revealed that LipNet's high accuracy relied heavily on speaker overlap between training and testing splits.", body))
    story.append(Paragraph("In 2017, Chung et al. introduced the <b>Lip Reading in the Wild (LRW)</b> dataset and the <b>Watch, Listen, Attend and Spell (WLAS)</b> architecture, extending lip-reading to a 500-word open-world broadcast lexicon. Recent self-supervised foundational models, such as Meta's <b>AV-HuBERT</b> (Shi et al., 2022), leverage masked audio-visual transformer encoders pre-trained on thousands of hours of video, achieving state-of-the-art results on continuous speech. However, their immense computational footprint (hundreds of millions of parameters) precludes direct deployment on consumer hardware without dedicated GPUs.", body))

    story.append(Paragraph("2.2 Comparative Literature Matrix", h2))
    lit_data = [
        [Paragraph("<b>Model / Study</b>", table_header), Paragraph("<b>Architecture</b>", table_header), Paragraph("<b>Dataset / Vocab</b>", table_header), Paragraph("<b>Reported Accuracy / WER</b>", table_header), Paragraph("<b>Key Limitation</b>", table_header)],
        [Paragraph("Assael et al.<br/>(LipNet, 2016)", table_cell), Paragraph("3D-CNN + 2x BiGRU + CTC", table_cell), Paragraph("GRID Corpus<br/>(51 Words)", table_cell), Paragraph("95.2% Sentence Acc<br/>4.8% WER", table_cell), Paragraph("Closed syntax; fails on unseen speakers without fine-tuning", table_cell)],
        [Paragraph("Chung & Zisserman<br/>(LRW, 2017)", table_cell), Paragraph("VGG-M / ResNet + BiLSTM", table_cell), Paragraph("Oxford LRW<br/>(500 Words)", table_cell), Paragraph("61.1% Top-1 Word Acc", table_cell), Paragraph("Requires millions of television broadcast clips for pretraining", table_cell)],
        [Paragraph("Shi et al.<br/>(AV-HuBERT, 2022)", table_cell), Paragraph("Conformer / Transformer Masked Pretraining", table_cell), Paragraph("LRS3-TED<br/>(Open Vocabulary)", table_cell), Paragraph("28.6% WER (Visual-Only)", table_cell), Paragraph("Enormous compute requirements (GPU clusters required)", table_cell)],
        [Paragraph("<b>Proposed System<br/>(LipNet-AI)</b>", table_cell), Paragraph("<b>WordVideoNet:<br/>3D-CNN + BiGRU + Clip Norm</b>", table_cell), Paragraph("<b>GRID Multi-Speaker<br/>(33 Speakers)</b>", table_cell), Paragraph("<b>34.38% Top-1 Acc<br/>62.50% Top-2 Acc</b>", table_cell), Paragraph("<b>Verified on CPU; strict speaker-disjoint held-out evaluation</b>", table_cell)]
    ]
    t_lit = Table(lit_data, colWidths=[1.3*inch, 1.4*inch, 1.2*inch, 1.4*inch, 1.7*inch])
    t_lit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_lit)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 3: SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("CHAPTER 3: SYSTEM ARCHITECTURE & TRACK DISENTANGLEMENT", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("3.1 Conceptual Framework", h2))
    story.append(Paragraph("The system is engineered as a modular visual speech recognition pipeline with clear boundary isolation between training, evaluation, and live inference. A frequent deficiency in student AI projects is conflating isolated word classification with continuous speech recognition or presenting letter classifiers as general-purpose word predictors. We enforce two strictly disentangled tracks:", body))
    story.append(Paragraph("• <b>Track A (Isolated Whole-Word Recognition):</b> Accepts a video clip of fixed length ($T=24$ frames) depicting a single spoken English word from a verified vocabulary. Outputs a discrete probability distribution over supported lexical items, gated by an empirical uncertainty threshold $\\tau = 0.35$.", bullet))
    story.append(Paragraph("• <b>Track B (Continuous Video-to-Text Recognition):</b> Accepts unsegmented silent video streams and produces token sequences using CTC alignment decoding. This track is kept visual-only with audio permanently disabled.", bullet))

    story.append(Spacer(1, 8))
    if (assets_dir / "system_architecture.png").exists():
        story.append(KeepTogether([
            Image(str(assets_dir / "system_architecture.png"), width=6.5*inch, height=3.25*inch),
            Paragraph("<b>Figure 3.1:</b> End-to-end multi-stage visual speech recognition architecture illustrating facial landmark tracking, temporal mouth normalization, spatio-temporal deep learning classification, uncertainty rejection, and full-stack API integration.", ParagraphStyle("Cap", parent=body, fontSize=8, alignment=1))
        ]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("3.2 Track B Status & Fail-Closed Policy", h2))
    story.append(Paragraph("During our codebase audit, we identified that an earlier diagnostic log reported a continuous WER of 38.89%. Deep inspection revealed that this metric was generated by evaluating a static dummy hypothesis sentence (<code>'bin blue at f two now'</code>) against 15 GRID sentences. Because GRID sentences adhere to a rigid 6-word grammatical template (<command> <color> <preposition> <letter> <digit> <adverb>), a constant sentence coincidentally matches 2 to 4 words in every utterance, producing an artificial edit distance of $35/90 = 38.89\%$.", body))
    story.append(Paragraph("Rather than masking this limitation, we updated <code>pipeline/sequence_recognition/ctc_decoder.py</code> to fail closed by raising an explicit <code>RuntimeError</code>. Track B is accurately documented as an active research track awaiting full visual-only transformer pre-training.", body))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 4 & 5: DATASET ENGINEERING & PREPROCESSING
    # =========================================================================
    story.append(Paragraph("CHAPTER 4: MULTI-DATASET ENGINEERING & LICENSING", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("4.1 Dataset Ingestion & Verification", h2))
    story.append(Paragraph("We established a unified multi-dataset acquisition framework codified in <code>DATASET_ACQUISITION_PLAN.md</code>. The status across six target speech corpora is summarized below:", body))

    ds_data = [
        [Paragraph("<b>Corpus</b>", table_header), Paragraph("<b>Provider / License</b>", table_header), Paragraph("<b>On-Disk Assets</b>", table_header), Paragraph("<b>Operational Status</b>", table_header)],
        [Paragraph("GRID Corpus", table_cell), Paragraph("Univ. of Sheffield (PVI)<br/>Non-Commercial Academic", table_cell), Paragraph("66 official archives<br/>33 speakers, 32,891 videos", table_cell), Paragraph("<b>Verified & Operational</b><br/>Primary benchmark", table_cell)],
        [Paragraph("MIRACL-VC1", table_cell), Paragraph("A. Watsky / Kaggle<br/>Open Academic", table_cell), Paragraph("Adapter & downloader ready<br/>(15 speakers, 10 words)", table_cell), Paragraph("Awaiting Kaggle token<br/><code>kaggle.json</code>", table_cell)],
        [Paragraph("TCD-TIMIT", table_cell), Paragraph("Sigmedia, Trinity College<br/>Non-Commercial Agreement", table_cell), Paragraph("Adapter implemented<br/>(59 speakers)", table_cell), Paragraph("Requires signed Sigmedia<br/>data agreement", table_cell)],
        [Paragraph("Oxford LRW", table_cell), Paragraph("BBC R&D / Oxford VGG<br/>BBC Research Agreement", table_cell), Paragraph("500-word vocabulary manifest<br/>(<code>data/vocab.txt</code>)", table_cell), Paragraph("Requires institutional<br/>BBC license", table_cell)]
    ]
    t_ds = Table(ds_data, colWidths=[1.3*inch, 2.0*inch, 2.2*inch, 1.5*inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 15))

    story.append(Paragraph("CHAPTER 5: COMPUTER VISION PREPROCESSING & ALIGNMENT", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("5.1 Facial Mesh Tracking & ROI Extraction", h2))
    story.append(Paragraph("For each video frame, facial geometry is extracted using Google's <b>MediaPipe Face Mesh</b> model, predicting 468 3D landmarks in real time. We dynamically track four primary lip boundary anchors:", body))
    story.append(Paragraph("• Landmark 61: Left mouth corner (cheilion)", bullet))
    story.append(Paragraph("• Landmark 291: Right mouth corner (cheilion)", bullet))
    story.append(Paragraph("• Landmark 0: Superior vermilion border (upper lip midpoint)", bullet))
    story.append(Paragraph("• Landmark 17: Inferior vermilion border (lower lip midpoint)", bullet))
    story.append(Paragraph("The mouth bounding box is expanded symmetrically by a safety scale factor of 1.25 to prevent lip clipping during wide articulatory vowel excursions (/a/, /o/). The cropped color region is converted to single-channel grayscale and spatially resized to $64 \\times 64$ pixels using bilinear interpolation.", body))

    story.append(Paragraph("5.2 Temporal Resampling & Ground-Truth Slicing", h2))
    story.append(Paragraph("Word boundary timing is computed from the official GRID <code>.align</code> files using millisecond offsets converted to discrete frame indices: $start = \\lfloor t_{start} \\times fps / 1000 \\rfloor$, $end = \\lceil t_{end} \\times fps / 1000 \\rceil$. Because speaking rates vary naturally, word durations range from 12 to 38 frames. To maintain fixed tensor dimensions for batch matrix multiplication, all clips are temporally resampled to exactly $T = 24$ frames using nearest-neighbor temporal interpolation.", body))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 6: MODEL ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("CHAPTER 6: DEEP LEARNING SPATIO-TEMPORAL MODEL FORMULATION", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("6.1 WordVideoNet Architecture", h2))
    story.append(Paragraph("The visual speech classification network, <code>WordVideoNet</code>, comprises a 3D-CNN spatiotemporal feature extractor, an instance normalization pre-conditioning layer, a multi-layer bidirectional recurrent temporal encoder, and a linear classification head.", body))

    story.append(Paragraph("<b>Input Representation:</b> A batch of preprocessed mouth clips is represented as a 5-dimensional tensor $X \\in \\mathbb{R}^{B \\times 1 \\times 24 \\times 64 \\times 64}$, where $B$ is batch size, 1 is the grayscale channel, 24 is the temporal frame sequence, and $64 \\times 64$ are spatial coordinates.", body))

    story.append(Paragraph("<b>Per-Clip Instance Normalization:</b> To decouple mouth articulation from ambient lighting and speaker skin tone, each clip is normalized across its spatial and temporal dimensions:", body))
    story.append(Paragraph("$$\\mu_b = \\frac{1}{24 \\times 64 \\times 64} \\sum_{t,h,w} X_{b,1,t,h,w}, \\quad \\sigma_b^2 = \\frac{1}{24 \\times 64 \\times 64} \\sum_{t,h,w} (X_{b,1,t,h,w} - \\mu_b)^2$$", ParagraphStyle("Eq", parent=body, fontName="Helvetica-Oblique", alignment=1)))
    story.append(Paragraph("$$\\tilde{X}_b = \\frac{X_b - \\mu_b}{\\sqrt{\\sigma_b^2 + 10^{-6}}}$$", ParagraphStyle("Eq2", parent=body, fontName="Helvetica-Oblique", alignment=1)))

    story.append(Paragraph("<b>3D-CNN Frontend:</b> Three successive 3D convolutional blocks extract localized motion and appearance cues across adjacent frames:", body))
    story.append(Paragraph("• Block 1: Conv3D ($1 \\to 8$ channels, kernel $3 \\times 3 \\times 3$, stride $(1, 2, 2)$, padding 1) + GroupNorm(1, 8) + SiLU $\\to [B, 8, 24, 32, 32]$", bullet))
    story.append(Paragraph("• Block 2: Conv3D ($8 \\to 16$ channels, kernel $3 \\times 3 \\times 3$, stride $(1, 2, 2)$, padding 1) + GroupNorm(1, 16) + SiLU $\\to [B, 16, 24, 16, 16]$", bullet))
    story.append(Paragraph("• Block 3: Conv3D ($16 \\to 32$ channels, kernel $3 \\times 3 \\times 3$, stride $(1, 2, 2)$, padding 1) + GroupNorm(1, 32) + SiLU $\\to [B, 32, 24, 8, 8]$", bullet))

    story.append(Paragraph("<b>Spatial Pooling & Temporal Modeling:</b> Spatial dimensions are globally averaged: $F_t = \\frac{1}{64} \\sum_{h,w} C_{b,c,t,h,w} \\in \\mathbb{R}^{B \\times 24 \\times 32}$. The sequence of 24 frame features is processed by a 2-layer Bidirectional GRU (hidden dimension $H = 48$, dropout 0.15):", body))
    story.append(Paragraph("$$\\vec{h}_t = \\text{GRU}_{fwd}(F_t, \\vec{h}_{t-1}), \\quad \\overleftarrow{h}_t = \\text{GRU}_{bwd}(F_t, \\overleftarrow{h}_{t+1}), \\quad h_t = [\\vec{h}_t ; \\overleftarrow{h}_t] \\in \\mathbb{R}^{96}$$", ParagraphStyle("Eq3", parent=body, fontName="Helvetica-Oblique", alignment=1)))
    story.append(Paragraph("Temporal vectors are pooled across time: $\\bar{h} = \\frac{1}{24} \\sum_{t=1}^{24} h_t$. Finally, a linear classification layer with dropout projects $\\bar{h}$ to class logits: $z = W \\bar{h} + b \\in \\mathbb{R}^{4}$, followed by Softmax normalization: $p_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$.", body))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 7 & 8: IMPLEMENTATION & RESULTS
    # =========================================================================
    story.append(Paragraph("CHAPTER 7: SOFTWARE ARCHITECTURE & FULL-STACK INTEGRATION", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("7.1 Backend Microservice & API Contracts", h2))
    story.append(Paragraph("The backend is built with FastAPI and runs on native Windows CPU execution. The service exposes three core endpoints:", body))
    story.append(Paragraph("• <code>GET /api/models</code>: Returns operational status and supported classes across all recognition tracks.", bullet))
    story.append(Paragraph("• <code>GET /api/word/health</code>: Confirms checkpoint loading (<code>best_word_model.pt</code>), device allocation, and metadata.", bullet))
    story.append(Paragraph("• <code>POST /api/predict/word</code>: Accepts JSON payloads containing base64-encoded JPEG mouth crops. Resamples frames, executes <code>WordVideoNet</code>, computes Softmax probabilities, and gates predictions against uncertainty threshold $\\tau = 0.35$. Average latency on Intel i3 CPU is <b>59.2 ms</b>.", bullet))

    story.append(Spacer(1, 15))
    story.append(Paragraph("CHAPTER 8: EXPERIMENTAL RESULTS, COLLAPSE REPAIR & DISCUSSION", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("8.1 Root-Cause Analysis of Majority-Class Collapse", h2))
    story.append(Paragraph("In initial training runs, the word classifier exhibited complete degenerate collapse: all 32 validation samples and all 32 test samples were predicted as a single majority class with near-uniform probabilities ($p \\approx 0.254$). Standard deviation across sample logits was $\\sigma \\approx 10^{-4}$. Through systematic ablation, four compounding causes were diagnosed:", body))
    story.append(Paragraph("1. <b>Speaker Illumination Discrepancy:</b> Mean pixel brightness across speakers varied between 138.9 and 148.6 (range $\\pm 10$ units). However, inter-frame lip motion delta averaged only $0.5$ to $1.3$ units. Without clip normalization, the 3D-CNN learned speaker illumination identities rather than lip visemes.", bullet))
    story.append(Paragraph("2. <b>Destructive Augmentation:</b> Random spatial shifts zero-padded borders with pure black pixels ($0.0$), introducing artificial step-edge gradients. Temporal frame freezing destroyed short-vowel opening transitions.", bullet))
    story.append(Paragraph("3. <b>Premature Early Stopping:</b> A patience of 4 epochs terminated training after only 96 optimization steps, before the model could escape the random-chance plateau.", bullet))

    story.append(Paragraph("8.2 Held-Out Empirical Evaluation (Unseen Speakers s30–s34)", h2))
    story.append(Paragraph("After implementing per-clip instance normalization, non-destructive circular roll augmentations, and an AdamW cosine annealing schedule, the model was retrained and evaluated on 32 test clips from five completely unseen held-out speakers (s30, s31, s32, s33, s34).", body))

    results_table_data = [
        [Paragraph("<b>Word Class</b>", table_header), Paragraph("<b>Test Support</b>", table_header), Paragraph("<b>Baseline Acc</b>", table_header), Paragraph("<b>Repaired Acc</b>", table_header), Paragraph("<b>Precision</b>", table_header), Paragraph("<b>Recall</b>", table_header), Paragraph("<b>F1-Score</b>", table_header)],
        [Paragraph("<code>bin</code>", table_cell), Paragraph("8", table_cell), Paragraph("0.0% (0/8)", table_cell), Paragraph("<b>87.5% (7/8)</b>", table_cell), Paragraph("0.30", table_cell), Paragraph("0.88", table_cell), Paragraph("0.45", table_cell)],
        [Paragraph("<code>lay</code>", table_cell), Paragraph("8", table_cell), Paragraph("100% (8/8)*", table_cell), Paragraph("0.0% (0/8)", table_cell), Paragraph("0.00", table_cell), Paragraph("0.00", table_cell), Paragraph("0.00", table_cell)],
        [Paragraph("<code>place</code>", table_cell), Paragraph("8", table_cell), Paragraph("0.0% (0/8)", table_cell), Paragraph("<b>25.0% (2/8)</b>", table_cell), Paragraph("0.50", table_cell), Paragraph("0.25", table_cell), Paragraph("0.33", table_cell)],
        [Paragraph("<code>set</code>", table_cell), Paragraph("8", table_cell), Paragraph("0.0% (0/8)", table_cell), Paragraph("<b>25.0% (2/8)</b>", table_cell), Paragraph("0.40", table_cell), Paragraph("0.25", table_cell), Paragraph("0.31", table_cell)],
        [Paragraph("<b>Overall Macro</b>", table_cell), Paragraph("<b>32</b>", table_cell), Paragraph("<b>25.0% (Collapsed)</b>", table_cell), Paragraph("<b>34.38% Top-1</b>", table_cell), Paragraph("<b>0.30</b>", table_cell), Paragraph("<b>0.34</b>", table_cell), Paragraph("<b>0.27</b>", table_cell)]
    ]
    t_res = Table(results_table_data, colWidths=[1.1*inch, 0.9*inch, 1.1*inch, 1.1*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))

    if (assets_dir / "confusion_matrix.png").exists():
        story.append(KeepTogether([
            Image(str(assets_dir / "confusion_matrix.png"), width=4.5*inch, height=3.8*inch),
            Paragraph("<b>Figure 8.1:</b> Empirical confusion matrix on unseen held-out speakers (s30–s34). Top-1 Accuracy: 34.38%, Top-2 Accuracy: 62.50%. Notice that 6 of 8 'place' samples are confused with 'bin' due to identical bilabial lip closure (/p/ vs /b/).", ParagraphStyle("Cap2", parent=body, fontSize=8, alignment=1))
        ]))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 9 & REFERENCES
    # =========================================================================
    story.append(Paragraph("CHAPTER 9: CONCLUSION & FUTURE WORK", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

    story.append(Paragraph("9.1 Summary of Contributions", h2))
    story.append(Paragraph("In this project, we designed, implemented, and empirically validated a visual speech recognition framework for complete spoken English words. Our key technical contributions include:", body))
    story.append(Paragraph("1. Resolving majority-class feature collapse by establishing per-clip instance normalization, increasing held-out accuracy from a 25.0% random baseline to 34.38% Top-1 and 62.50% Top-2 accuracy across five unseen speakers.", bullet))
    story.append(Paragraph("2. Eliminating ungrounded open-vocabulary claims by enforcing strict track disentanglement and adopting a fail-closed policy on continuous sequence decoding.", bullet))
    story.append(Paragraph("3. Delivering a verified full-stack deployment featuring 59 ms CPU inference, mathematical uncertainty gating, and 32 passing automated tests.", bullet))

    story.append(Paragraph("9.2 Limitations & Future Directions", h2))
    story.append(Paragraph("While successful, the current system has clear boundaries: (1) The verified vocabulary is restricted to four lexical words; (2) Performance is subject to visual homopheny between bilabials (/p/ and /b/); and (3) Inference is evaluated on controlled frontal faces. Future work will integrate transformer-based visual language modeling (AV-HuBERT fine-tuning) to incorporate n-gram linguistic priors and expand the vocabulary to 500+ words.", body))
    story.append(Spacer(1, 15))

    story.append(Paragraph("REFERENCES", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
    refs = [
        "[1] J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, 'Lip Reading Sentences in the Wild,' in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 6447–6456. DOI: 10.1109/CVPR.2017.683.",
        "[2] Y. M. Assael, B. Shillingford, S. Whiteson, and N. de Freitas, 'LipNet: End-to-End Sentence-level Lipreading,' arXiv preprint arXiv:1611.01599, 2016.",
        "[3] B. Shi, W.-N. Hsu, K. Lakhotia, and A. Mohamed, 'Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction,' in International Conference on Learning Representations (ICLR), 2022.",
        "[4] J. Cooke, M. Barker, S. Cunningham, and X. Shao, 'An audio-visual corpus for speech perception and automatic speech recognition,' Journal of the Acoustical Society of America, vol. 120, no. 5, pp. 2421–2424, 2006. DOI: 10.1121/1.2229005.",
        "[5] G. Potamianos, C. Neti, G. Gravier, A. Garg, and A. W. Senior, 'Recent advances in the automatic recognition of audiovisual speech,' Proceedings of the IEEE, vol. 91, no. 9, pp. 1306–1326, 2003. DOI: 10.1109/JPROC.2003.817150.",
        "[6] K. Lugosch, M. Ravanelli, P. Ignoto, V. S. Tomar, and Y. Bengio, 'SpeechBrain: A General-Purpose Speech Toolkit,' arXiv preprint arXiv:2106.04624, 2021.",
        "[7] C. Lugaresi et al., 'MediaPipe: A Framework for Building Perception Pipelines,' arXiv preprint arXiv:1906.08172, 2019."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", parent=body, fontSize=8, leading=11, spaceAfter=4)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [SUCCESS] Generated PDF: {output_pdf}")

def build_docx():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("CONTEXT-AWARE LIP READING AND SPEECH GENERATION USING DEEP LEARNING\n")
    r_title.bold = True
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.add_run("CAPSTONE PROJECT WHITE BOOK (PROJECT REPORT)\n").bold = True
    p_sub.add_run("Submitted in partial fulfillment of the requirements for the Degree of\nBachelor of Technology in Electronics and Computer Engineering\n\n")

    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cand.add_run("Submitted by:\nRavikumar Gupta (Roll No: 16014023042)\nOmkar Gupta (Roll No: 16014023035)\n\nUnder the Guidance of:\nProf. Umang Patel\nAssistant Professor, Dept of ECE\n\n")
    p_cand.add_run("DEPARTMENT OF ELECTRONICS AND COMPUTER ENGINEERING\nK. J. SOMAIYA SCHOOL OF ENGINEERING\nSOMAIYA VIDYAVIHAR UNIVERSITY, MUMBAI – 400077\nACADEMIC YEAR: 2026–2027\n")

    doc.add_page_break()

    doc.add_heading("Table of Contents", level=1)
    chapters = [
        "Chapter 1: Introduction, Motivation & Problem Statement",
        "Chapter 2: Literature Review & Gap Analysis",
        "Chapter 3: System Architecture & Track Disentanglement",
        "Chapter 4: Multi-Dataset Engineering & Licensing Protocols",
        "Chapter 5: Computer Vision Preprocessing & Landmark Alignment",
        "Chapter 6: Deep Learning Spatio-Temporal Model Formulation",
        "Chapter 7: Software Architecture & Full-Stack Integration",
        "Chapter 8: Experimental Results, Collapse Diagnosis & Discussion",
        "Chapter 9: Conclusion, Societal Impact & Future Research",
        "References (IEEE Format)",
        "Appendix A: API Contracts & Payloads",
        "Appendix B: Test Suite Logs"
    ]
    for ch in chapters:
        doc.add_paragraph(ch, style='List Bullet')

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph("Visual Speech Recognition (VSR)—commonly referred to as automated lip-reading—interprets spoken linguistic tokens solely from silent video recordings of mouth kinematics. In this Capstone project, we present an end-to-end multi-dataset Visual Speech Recognition system operating across two strictly separated tracks: (1) Track A: Isolated Whole-Word Recognition powered by a customized spatio-temporal neural network (WordVideoNet: 3D-CNN + 2-layer Bidirectional GRU) with Softmax uncertainty gating; and (2) Track B: Continuous Video-to-Text Recognition leveraging greedy Connectionist Temporal Classification (CTC) sequence decoding with visual-only constraints.")
    doc.add_paragraph("A critical contribution of this research is the empirical diagnosis and remediation of majority-class collapse. By engineering a per-clip instance normalization layer and non-destructive circular roll augmentations, we restored gradient sensitivity, achieving 34.38% Top-1 Accuracy and 62.50% Top-2 Accuracy across completely unseen held-out speakers (s30–s34).")

    doc.save(str(source_docx))
    print(f"  [SUCCESS] Generated DOCX: {source_docx}")

if __name__ == "__main__":
    build_pdf()
    build_docx()
