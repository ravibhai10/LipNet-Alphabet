# LIPNET-AI: OFFICIAL CAPSTONE PROJECT SUBMISSION MANIFEST
## K. J. Somaiya School of Engineering | Somaiya Vidyavihar University

**Group ID:** `12`  
**Project Title:** `Context-Aware Lip Reading and Speech Generation Using Deep Learning`  
**Department:** Department of Electronics and Computer Engineering (B.Tech ECE 2026–2027)  
**Project Mentor / Guide:** Prof. Umang Patel  
**Team Members:**  
1. Ravikumar Gupta (Roll No: `16014023042`)  
2. Omkar Gupta (Roll No: `16014023035`)  
**Official Submission Deadline:** 31 October 2026  

---

### 1. Mandatory Deliverables Summary Table

| Deliverable Filename | Official Description | Format | Page / Slide Count | File Size | SHA-256 Checksum (First 16 chars) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `12_WhiteBook.pdf` | Official Final Year Project Report / White Book | `PDF` | 13 Pages | 201.4 KB | `4141ad8865aa5c65...` | **VERIFIED** |
| `12_ESE_Presentation.pptx` | ESE Final Project Defense Presentation (16 Slides + Viva Notes) | `PPTX` | 16 Slides (Viva Notes: Yes) | 84.6 KB | `0ad5cf363df892f1...` | **VERIFIED** |
| `12_ResearchPaper.pdf` | Peer-Review Ready Research Paper (IEEE Transactions Format) | `PDF` | 3 Pages | 11.4 KB | `5d757bc47283caf4...` | **VERIFIED** |
| `12_PlagiarismReport.pdf` | University Academic Integrity & Plagiarism Dossier | `PDF` | 3 Pages | 21.4 KB | `fb54415bb27245a6...` | **VERIFIED** |
| `12_TechnicalManual.pdf` | Engineering Documentation, API Reference & Deployment Runbook | `PDF` | 4 Pages | 107.4 KB | `35729cfe512b32fe...` | **VERIFIED** |
| `12_UserManual.pdf` | Non-Technical End-User Manual & Assistive Operation Guide | `PDF` | 2 Pages | 8.0 KB | `0e3cf834365261e9...` | **VERIFIED** |
| `12_ProjectPoster.jpg` | A4 Landscape Publication-Quality Poster (300 DPI, 3508x2480) | `JPG` | 3508x2480 px, 300 DPI | 1.50 MB | `363a51eee66d64ee...` | **VERIFIED** |
| `source/12_WhiteBook.docx` | Editable Source Document for White Book | `DOCX` | Word OpenXML Document | 37.2 KB | `8b9acac2b7166eb8...` | **VERIFIED** |
| `source/12_ResearchPaper.docx` | Editable Source Document for Research Paper | `DOCX` | Word OpenXML Document | 36.8 KB | `cf63af78dbb8f6e5...` | **VERIFIED** |
| `source/12_TechnicalManual.docx` | Editable Source Document for Technical Manual | `DOCX` | Word OpenXML Document | 37.0 KB | `3a1d7dcc1477e686...` | **VERIFIED** |
| `source/12_UserManual.docx` | Editable Source Document for User Manual | `DOCX` | Word OpenXML Document | 36.6 KB | `d5b33865156f4ecd...` | **VERIFIED** |

---

### 2. Deliverable Details & Verification Evidence

#### Deliverable 1: `12_WhiteBook.pdf` (and `source/12_WhiteBook.docx`)
- **Specification:** Official academic project report adhering strictly to university White Book guidelines.
- **Contents:** Cover page, Certificate page placeholder, Student Declaration, Acknowledgments, Abstract, List of Figures/Tables, and 9 Full Technical Chapters:
  - Chapter 1: Introduction, Visual Speech Recognition, and Assistive Motivation.
  - Chapter 2: Literature Review, Viseme Classifications, and Baseline Limitations.
  - Chapter 3: System Architecture & Decoupled Client-Server Microservice.
  - Chapter 4: Data Engineering, GRID Corpus Structuring & Non-Destructive Augmentation.
  - Chapter 5: WordVideoNet Neural Architecture & Dynamic Instance Normalization.
  - Chapter 6: Track B CTC Continuous Sequence Investigation & Fail-Closed Safety.
  - Chapter 7: Results, Confusion Matrix, and Viseme Homopheny Analysis.
  - Chapter 8: Software Quality Assurance, Benchmarks & Automated Pytest Suite.
  - Chapter 9: Conclusion, Assistive Impact & Future Research Directions.
- **Page Numbering:** Roman numerals for preliminary pages (`i`, `ii`, `iii`...); Arabic numerals (`Page 1 of N`) for technical chapters.

#### Deliverable 2: `12_ESE_Presentation.pptx`
- **Specification:** 16:9 widescreen presentation with high-contrast engineering aesthetic (Deep Navy `#1A365D`, Somaiya Red `#A8201A`, Light Grey cards).
- **Slide Deck Structure:** 16 slides covering Title, Motivation, Architecture, Normalization Breakthrough, WordVideoNet Deep Pipeline, Experimental Setup, Results & Confusion Matrix, Viseme Homopheny Analysis, Track B Fail-Closed Governance, Real-Time Microservice Demo, QA Benchmarks, Team Roles, and Viva Defense Q&A.
- **Viva Speaker Notes:** Every single slide contains word-for-word spoken talking points for external examiners.

#### Deliverable 3: `12_ResearchPaper.pdf` (and `source/12_ResearchPaper.docx`)
- **Specification:** Formatted in 2-column IEEE Transactions scholarly conference style.
- **Key Scientific Highlights:** Detailed mathematical derivation of Dynamic Instance Normalization ($x \leftarrow (x-\mu)/\sigma$), speaker illumination bias analysis, Top-1 (34.38%) vs Top-2 (62.50%) metric explanation across bilabial plosives (/p/ vs /b/), and complete bibliography.

#### Deliverable 4: `12_PlagiarismReport.pdf`
- **Specification:** Somaiya Vidyavihar University Academic Integrity Dossier and Pre-Submission Similarity Audit.
- **Contents:** Student declarations of authorship, exhaustive attribution of open-source baselines (GRID, LipNet, MediaPipe, PyTorch), summary of Group 12 novel contributions, and a step-by-step portal upload guide for the project guide.

#### Deliverable 5: `12_TechnicalManual.pdf` (and `source/12_TechnicalManual.docx`)
- **Specification:** In-depth engineering manual for developers, maintainers, and system evaluators.
- **Contents:** Hardware/software specs, Windows installation runbook, PyTorch tensor dimensions, RESTful API specifications (`GET /api/models`, `GET /api/word/health`, `POST /api/predict/word`, `POST /api/predict/sequence`), automated pytest instructions, and diagnostic tables.

#### Deliverable 6: `12_UserManual.pdf` (and `source/12_UserManual.docx`)
- **Specification:** Non-technical user guide for patients, clinicians, and demonstration evaluators.
- **Contents:** Web portal navigation (`http://localhost:3000`), webcam permission instructions, lighting and camera setup recommendations, 4-word isolated articulation walkthrough, audio playback guide, and troubleshooting FAQ.

#### Deliverable 7: `12_ProjectPoster.jpg`
- **Specification:** High-resolution A4 landscape poster ($3508 \times 2480$ pixels at 300 DPI) for academic display.
- **Visual Elements:** Official K. J. Somaiya School of Engineering logo top-left, Somaiya Trust blue logo top-right, embedded system architecture diagram, embedded confusion matrix, mathematical normalization equation, performance benchmark tables, student roll numbers, and guide attribution.

---

### 3. Verification & Compliance Checklist

- [x] **File Naming Convention:** All final submission files strictly begin with `12_` prefix.
- [x] **Internal Fact Consistency:** Vocabulary (bin, lay, place, set), accuracy (34.38% Top-1, 62.50% Top-2), latency (59.2 ms), and team roll numbers are 100% synchronized across all 7 deliverables.
- [x] **Reproducibility:** Passed 32/32 unit tests with 0 failures.
- [x] **Fail-Closed Continuous Speech Safety:** Track B strictly refuses to return dummy sequences, safeguarding academic integrity.
- [x] **Action Items for Candidates Before Guide Sign-Off:**
  1. Review printed copies of `12_WhiteBook.pdf` and `12_PlagiarismReport.pdf` with Prof. Umang Patel.
  2. Obtain ink signatures on the Certificate and Declaration pages.
  3. Upload `12_ResearchPaper.pdf` to the department Turnitin portal and attach the digital receipt to `12_PlagiarismReport.pdf`.