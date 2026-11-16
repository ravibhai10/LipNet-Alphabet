"""Build an evidence-backed before/after report without running model training."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/word_accuracy/avhubert_miracl8_v1"
BASE = ROOT / "runs/multidataset/miracl8_cpu_v2"
OUT = ROOT / "submission_package"
ASSETS = OUT / "assets/finetuning_comparison"
NAME = "Fine_Tuning_Before_and_After"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from pypdf import PdfReader

    e, paired, selection, protocol = [load(RUN / n) for n in
        ("evaluation.json", "paired_comparison.json", "selection.json", "protocol.json")]
    old, new = rows(RUN / "preserved_scratch_test_predictions.jsonl"), rows(RUN / "test_predictions.jsonl")
    old_by, new_by = {r["id"]: r for r in old}, {r["id"]: r for r in new}
    words = e["vocabulary"]
    assert len(old) == len(new) == len(old_by) == len(new_by) == paired["same_test_clips"] == 192
    assert old_by.keys() == new_by.keys()
    assert not set(new_by).intersection(protocol["quarantined_previous_test_ids"])
    assert paired["same_vocabulary"] == words == protocol["vocabulary"]
    assert all(old_by[k]["ground_truth"] == new_by[k]["ground_truth"] for k in old_by)
    assert sum(r["correct_raw"] for r in old) == paired["preserved_scratch_correct"] == 24
    assert sum(r["correct_raw"] for r in new) == e["correct"] == paired["pretrained_correct"] == 182
    assert sum(not old_by[k]["correct_raw"] and new_by[k]["correct_raw"] for k in old_by) == 158
    assert sum(old_by[k]["correct_raw"] and not new_by[k]["correct_raw"] for k in old_by) == 0
    matrix = np.zeros((8, 8), dtype=int)
    for r in new:
        matrix[words.index(r["ground_truth"]), words.index(r["candidate"])] += 1
    assert matrix.tolist() == e["confusion_matrix"]
    assert sum(words.index(r["ground_truth"]) in np.argsort(r["probabilities"])[-5:] for r in new) == 191
    head = RUN / "head.pt"
    scratch = BASE / "best.pt"
    assert digest(head) == paired["pretrained_head_sha256"] == e["head_sha256"]
    assert digest(scratch) == paired["scratch_checkpoint_sha256"]
    assert digest(ROOT / paired["scratch_model_definition"]) == paired["scratch_model_source_sha256"]
    assert digest(RUN / "features.jsonl") == selection["features_manifest_sha256"]
    assert len(selection["candidates"]) == 27 and selection["test_used"] is False
    training = load(BASE / "training.json")
    current_rejection = load(RUN / "rejection_policy.json")
    preparation, extraction = load(RUN / "preparation.json"), load(RUN / "extraction.json")
    baseline_selected = rows(BASE / "selected.jsonl")
    baseline_counts = Counter(r["split"] for r in baseline_selected)
    assert baseline_counts["train"] == 140
    source_rows = rows(RUN / "sources.jsonl")
    speakers = {s: sorted({r["speaker_id"] for r in source_rows if r["split"] == s and r["role"] == "known_word"})
                for s in ("train", "valid", "test")}
    assert all(not set(speakers[a]) & set(speakers[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")))
    ASSETS.mkdir(parents=True, exist_ok=True)
    date = datetime.now(ZoneInfo("Asia/Calcutta")).strftime("%d %B %Y")
    before, after = 100 * 24 / 192, 100 * 182 / 192
    delta = after - before
    old_correct = {w: sum(r["correct_raw"] for r in old if r["ground_truth"] == w) for w in words}

    # Standalone figures are derived only from recorded predictions.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 2.15), layout="constrained")
    ax.barh(["Before: scratch CNN + BiGRU", "After: frozen AV-HuBERT + head"], [before, after], color=["#93979d", "#399982"], height=.5)
    ax.invert_yaxis(); ax.set_xlim(0, 110); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Top-1 accuracy on the same 192 clips (%)")
    for i, value in enumerate([before, after]):
        ax.text(value + 1, i, f"{value:.2f}%", va="center", weight="bold", fontsize=11)
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    chart = ASSETS / "before_after_accuracy.png"
    fig.savefig(chart, dpi=190, facecolor="white"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 3.35), layout="constrained")
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=24)
    ax.set_xticks(range(8), words, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(8), words, fontsize=8)
    ax.set_xlabel("Predicted word", fontsize=9); ax.set_ylabel("Actual word", fontsize=9)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color="white" if matrix[i, j] > 12 else "#263832")
    cm = ASSETS / "after_confusion_matrix.png"
    fig.savefig(cm, dpi=190, facecolor="white"); plt.close(fig)

    # A common content model keeps the PDF, Word and Markdown claims identical.
    pages = []
    def page(title, subtitle=""):
        content = [("title", title)]
        if subtitle:
            content.append(("subtitle", subtitle))
        pages.append(content)
        return content
    def p(c, text): c.append(("paragraph", text))
    def h(c, text): c.append(("heading", text))
    def table(c, headers, data): c.append(("table", (headers, data)))

    c = page("Fine-Tuning & Model Adaptation", "BEFORE AND AFTER | LipNet-AI | " + date)
    h(c, "Measured result: 12.50% to 94.79%")
    p(c, "The preserved scratch word classifier correctly classified 24 of 192 clips. The new frozen AV-HuBERT encoder with a supervised eight-word classifier correctly classified 182 of those same 192 source clips: an improvement of 82.29 percentage points. [1, 2]")
    c.append(("image", (chart, 6.55, 1.96)))
    table(c, ["Measure", "Before", "After"], [
        ["Correct / total", "24 / 192", "182 / 192"],
        ["Top-1 accuracy", "12.50%", "94.79%"],
        ["Incorrect predictions", "168 / 192", "10 / 192"],
        ["Top-5 accuracy", "Not recorded in paired baseline", "191 / 192 (99.48%)"],
        ["Output vocabulary", "Same eight words", "Same eight words"],
    ])
    h(c, "What was actually trained")
    p(c, "This was supervised classifier-head training on frozen pretrained features (transfer learning / linear probing). AV-HuBERT encoder weights were not fine-tuned. The comparison replaces the feature extractor and changes preprocessing and training-data volume; it does not isolate the effect of one fine-tuning step.")
    h(c, "Scope")
    p(c, "Silent isolated-word recognition using MIRACL-VC1 RGB recordings: choose, connection, hello, navigation, next, previous, start, stop. The 94.79% figure belongs only to this eight-word offline experiment. It is not a 500-word LRW, sentence-recognition, or physical-webcam accuracy result.")
    p(c, "Experiment recorded: 29 October 2026. Report prepared from existing artifacts on " + date + ". No new training or recognition experiment was run to create this document.")

    c = page("Architecture and training changes", "What changed between the two systems")
    table(c, ["Component", "Before: preserved scratch model", "After: pretrained features + head"], [
        ["Feature extractor", "Three Conv3D blocks; channels 8, 16, 32; GroupNorm and SiLU", "Official AV-HuBERT Base encoder; weights frozen"],
        ["Temporal processing", "Two-layer bidirectional GRU; hidden size 48 per direction", "Transformer layer 9 features; temporal mean + standard deviation"],
        ["Classifier input", "96-dimensional pooled GRU representation", "1,536-dimensional pooled encoder representation"],
        ["Output", "Dropout 0.15 and linear eight-class layer", "Training-fitted StandardScaler and eight-class logistic classifier"],
        ["Tensor format", "[B, 1, 24, 64, 64] grayscale", "[B, 1, 29, 88, 88] grayscale"],
        ["Scaling", "Pixels divided by 255", "Pixels / 255, then (x - 0.421) / 0.165"],
    ])
    h(c, "Frozen-feature head fitting [3, 4, 7]")
    p(c, "Measured landmarks align the eye axis and mouth center. The sequence uses a 96 x 96 mouth crop, linear interpolation to 29 frames, and an 88 x 88 center crop. Layers 6, 9 and 12 were cached together as [3, 29, 768] features. Native MIRACL FPS is unknown; interpolation does not establish a frame rate.")
    p(c, "The search tested 3 layers x 3 pooling choices (mean, three temporal means, mean/std) x 3 regularization values C = 0.01, 0.1, 1.0: 27 candidates. Selection used highest validation accuracy, breaking ties with lower validation cross-entropy. The scaler was fitted on training features only.")
    p(c, "Winner: layer 9, mean/std pooling, C = 1.0; L-BFGS logistic regression, class_weight='balanced', random_state=42, max_iter=600. The selected fit recorded 11 solver iterations, not 11 epochs. It has 8 x 1,536 + 8 = 12,296 classifier coefficients/biases, plus stored scaler statistics.")
    h(c, "Baseline training context [5, 6]")
    p(c, f"The baseline log records {len(training['history'])} epochs with best epoch {training['best_epoch']} selected by validation loss, batch size 2, seed 42 and one CPU thread. Its two-repetition protocol planned 144 training clips and retained 140 rows. The new head used 704 usable training clips from repetitions 1-10. Training volume and model selection differ; this is a system comparison, not a controlled equal-data architecture ablation.")

    c = page("Dataset and evaluation protocol", "Real inputs, fixed splits and selection boundaries")
    table(c, ["Partition", "Speakers", "Known-word clips"], [
        ["Training", ", ".join(speakers["train"]), "720 planned; 704 usable"],
        ["Validation", ", ".join(speakers["valid"]), "240 / 240 usable"],
        ["Test", ", ".join(speakers["test"]), "192 / 192 usable; 24 per word"],
        ["Phrase diagnostics", "Separate validation / test phrase sets", "60 validation + 60 test; not word-training labels"],
    ])
    h(c, "Evaluation safeguards [1-4]")
    p(c, "Training, validation and test speakers do not overlap. Test utterances are repetitions 03-10 from F10, F11 and M08. The 48 repetitions 01-02 used by the earlier pilot were explicitly quarantined. Those test speakers had been observed in the pilot: these are fresh utterances on the original held-out speakers, not a completely untouched new speaker population.")
    p(c, "Head and feature choices used word validation results, with selection.json explicitly recording test_used=false. Test predictions were evaluated after selection. Existing selected heads and evaluation outputs are protected from accidental overwrite. Tracking failures are explicit; evaluation failures count as errors rather than disappearing from the denominator.")
    p(c, "For the paired result, both systems received the same 192 source IDs and original RGB recordings. Each used its own actual trained spatial and temporal preprocessing. The old model was not evaluated on downscaled AV-HuBERT feature-cache inputs. [1, 7]")
    h(c, "Measured CPU work [3, 8]")
    table(c, ["Stage", "Recorded time", "Work represented"], [
        ["Tracking / preparation", f"{preparation['seconds']:.2f} s", "1,272 planned word/phrase clips; 1,256 usable"],
        ["Frozen feature extraction", f"{extraction['seconds']:.2f} s", "1,256 encoded word/phrase clips"],
        ["27-head search and fitting", f"{selection['seconds']:.2f} s", "Training and validation selection"],
        ["Paired baseline comparison", f"{paired['seconds']:.2f} s", "192 original RGB test clips"],
    ])
    p(c, "Execution was recorded on the existing Windows CPU environment (Intel i3-1115G4; no CUDA). These times have different workloads and exclude some setup/loading work. They are neither an equal-work speed comparison nor a real-time webcam latency guarantee.")

    c = page("Per-word results and confusion", "Same 192 test source clips | raw top-1 before rejection")
    table(c, ["Word", "Before correct", "After correct", "After accuracy", "After F1"], [
        [w, f"{old_correct[w]} / 24", f"{e['per_word'][w]['correct']} / 24", f"{100*e['per_word'][w]['correct']/24:.2f}%", f"{e['per_word'][w]['f1-score']:.4f}"] for w in words
    ])
    c.append(("image", (cm, 6.2, 3.15)))
    p(c, "Rows are actual labels; columns are predictions. Four 'previous' clips were classified as 'start'. The remaining six errors were connection->choose (1), navigation->connection (1), next->hello (1), next->navigation (1), previous->choose (1), previous->navigation (1). [2]")
    p(c, "After-model macro F1: 0.94668. Single-word substitution error: 10/192 = 5.21% (not sentence WER). By speaker: F10 62/64 (96.88%); F11 57/64 (89.06%); M08 63/64 (98.44%). On paired clips, 158 baseline errors became correct and no baseline-correct clips became incorrect. [1, 2]")

    c = page("Interpretation and current limits", "Keep offline accuracy separate from live acceptance")
    h(c, "Raw accuracy and rejection answer different questions")
    p(c, "The 94.79% result counts every known test word before uncertainty rejection. The frozen evaluation also records confidence >= 0.95 and margin >= 0.20: 137/192 known words accepted (71.35% coverage), all 137 accepted words correct, and 55 rejected. The same policy falsely accepted 8/60 test phrases (13.33%). Accepted accuracy is conditional on rejection and must not be presented as 100% overall recognition. [2]")
    h(c, "Configuration drift found while preparing this report")
    p(c, f"The current rejection_policy.json contains threshold {current_rejection['threshold']:.2f} and margin {current_rejection['margin']:.2f}, while evaluation.json records 0.95 and 0.20. The saved 137/192 acceptance and 8/60 phrase results therefore apply to the evaluated historical policy only; they cannot be assigned to the current configuration. Its claim of webcam calibration is not established by these offline artifacts. This report records the difference without changing the file. [2, 9]")
    h(c, "What the evidence supports")
    p(c, "The new complete eight-word system performs substantially better than the preserved scratch system on the specified paired clips. A richer pretrained representation is a plausible contributor, but data volume, preprocessing and selection also changed. No equal-data ablation establishes their separate causal contributions.")
    p(c, "A saved real API smoke accepted a labeled 'connection' clip at confidence 0.99838. This verifies one actual request path; confidence is not an accuracy estimate. No fresh API inference was run for this report. [10]")
    h(c, "Boundaries and next experiments")
    p(c, "Physical webcam accuracy, new cameras/lighting, unseen isolated words and unrestricted English remain unmeasured. GRID letter-slot training and the independent publisher-trained LRW 500-word model are separate experiments; neither inherits this eight-word score. No verified visual-only sentence decoder is established by these results.")
    p(c, "A defensible next study would freeze the acceptance policy, collect genuinely new speakers and webcam clips, compare both systems on equal training data, and evaluate once on a fresh held-out test set. True encoder fine-tuning would require a separate run with explicit unfrozen layers and optimization logs; it was not performed here.")

    c = page("Evidence and reproducibility", "Repository-relative evidence | report generation is read-only for model artifacts")
    refs = [
        ("1", "Paired source-matched comparison", "runs/word_accuracy/avhubert_miracl8_v1/paired_comparison.json"),
        ("2", "Frozen test metrics and confusion matrix", "runs/word_accuracy/avhubert_miracl8_v1/evaluation.json"),
        ("3", "27 candidates and selected hyperparameters", "runs/word_accuracy/avhubert_miracl8_v1/selection.json"),
        ("4", "Vocabulary, preprocessing and quarantine", "runs/word_accuracy/avhubert_miracl8_v1/protocol.json"),
        ("5", "Preserved scratch architecture", "pipeline/word_recognition/legacy_word_model.py"),
        ("6", "Baseline training log, contract and selected rows", "runs/multidataset/miracl8_cpu_v2/{training.json,contract.json,selected.jsonl,split_inventory.json}"),
        ("7", "Fitting and source-matched evaluation code", "pipeline/word_recognition/pretrained.py; scripts/compare_word_accuracy.py"),
        ("8", "Measured preparation and extraction", "runs/word_accuracy/avhubert_miracl8_v1/{preparation.json,extraction.json}"),
        ("9", "Current acceptance configuration", "runs/word_accuracy/avhubert_miracl8_v1/rejection_policy.json"),
        ("10", "Historical real API smoke", ".qa/word-improvement-real-api.json"),
    ]
    for number, label, path in refs:
        c.append(("reference", f"[{number}] {label}: {path}"))
    h(c, "Artifact identities (SHA-256)")
    for label, value in [("Preserved scratch best.pt", digest(scratch)), ("Trained eight-word head.pt", digest(head)), ("Frozen encoder (recorded by protocol)", protocol["encoder_sha256"])]:
        c.append(("reference", label + ": " + value))
    p(c, "The builder rechecks the two classifier checkpoint hashes, legacy source hash, feature-manifest hash, paired test IDs, quarantine exclusion, confusion counts, top-5 count and disjoint speaker assignments. It reads predictions and logs without loading either neural network or modifying checkpoints. The large encoder hash above is from the recorded protocol, not a fresh rehash.")
    c.append(("reference", "Rebuild document only: .\\.venv\\Scripts\\python.exe scripts/build_finetuning_comparison.py"))
    p(c, "Outputs: PDF, editable DOCX, Markdown, two PNG figures and a document-validation receipt. Existing submission documents, frontend work, model selection and running servers are preserved. No new model accuracy is claimed by report generation.")

    # PDF: charcoal / mint palette, light pages for readable printing.
    pdf_path, docx_path, md_path = [OUT / (NAME + ext) for ext in (".pdf", ".docx", ".md")]
    styles = {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=colors.HexColor("#182724"), spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#4b6f63"), spaceAfter=17),
        "heading": ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#24745f"), spaceBefore=10, spaceAfter=6),
        "paragraph": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.4, leading=13.4, spaceAfter=8),
        "reference": ParagraphStyle("Reference", fontName="Helvetica", fontSize=8.2, leading=11.1, spaceAfter=6, splitLongWords=True),
        "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.4, leading=11),
        "th": ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=8.4, leading=11, textColor=colors.white),
    }
    def decorate(canvas, doc):
        canvas.setFillColor(colors.HexColor("#8fdfcb")); canvas.rect(0, A4[1]-13, A4[0], 13, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#54605c")); canvas.setFont("Helvetica", 8)
        canvas.drawString(43, 25, "LIPNET-AI | Eight-word model adaptation | Evidence-based report")
        canvas.drawRightString(A4[0]-43, 25, str(doc.page))
    flow = []
    for index, content in enumerate(pages):
        if index: flow.append(PageBreak())
        for kind, value in content:
            if kind == "image":
                path, width, height = value
                flow.extend([Image(str(path), width=width*72, height=height*72), Spacer(1, 6)])
            elif kind == "table":
                headers, data = value
                widths = [509/len(headers)] * len(headers)
                body = [[Paragraph(escape(str(x)), styles["th" if i == 0 else "cell"]) for x in row] for i, row in enumerate([headers]+data)]
                t = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183e34")),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f6f3"),colors.white]),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#8fdfcb"))]))
                flow.extend([t, Spacer(1, 7)])
            else:
                flow.append(Paragraph(escape(value), styles[kind]))
    SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=43, leftMargin=43, topMargin=42, bottomMargin=45, title="LipNet-AI: Fine-Tuning Before and After", author="LipNet-AI Project").build(flow, onFirstPage=decorate, onLaterPages=decorate)

    # Editable Word version; the same content and explicit section page breaks.
    doc = Document(); sec = doc.sections[0]
    sec.page_width=Inches(8.27); sec.page_height=Inches(11.69)
    sec.left_margin=sec.right_margin=Inches(.6); sec.top_margin=sec.bottom_margin=Inches(.6)
    normal=doc.styles["Normal"]; normal.font.name="Calibri"; normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6)
    for name in ("Title", "Heading 1", "Heading 2"):
        doc.styles[name].font.color.rgb=RGBColor.from_string("24745F")
    sec.header.paragraphs[0].text="LIPNET-AI  /  RESEARCH EVIDENCE"
    footer=sec.footer.paragraphs[0]; footer.add_run("Eight-word model adaptation | Page ")
    field=OxmlElement("w:fldSimple"); field.set(qn("w:instr"),"PAGE"); footer._p.append(field)
    md=[]
    for index, content in enumerate(pages):
        if index: doc.add_page_break(); md.append("\n---\n")
        for kind, value in content:
            if kind == "title": doc.add_heading(value,0 if index==0 else 1); md.append("# " + value + "\n")
            elif kind == "heading": doc.add_heading(value,2); md.append("## " + value + "\n")
            elif kind == "image":
                path,width,height=value; doc.add_picture(str(path),width=Inches(width),height=Inches(height))
                md.append(f"![{path.stem}](assets/finetuning_comparison/{path.name})\n")
            elif kind == "table":
                headers,data=value; tab=doc.add_table(rows=1,cols=len(headers));tab.style="Light Shading Accent 1"
                for cell,text in zip(tab.rows[0].cells,headers): cell.text=str(text)
                for row in data:
                    for cell,text in zip(tab.add_row().cells,row): cell.text=str(text)
                for row in tab.rows:
                    for cell in row.cells:
                        for par in cell.paragraphs:
                            par.paragraph_format.space_after=Pt(2)
                            for run in par.runs:run.font.size=Pt(9)
                md.extend(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"])
                md.extend("| "+" | ".join(map(str,row))+" |" for row in data);md.append("")
            else:
                para=doc.add_paragraph(value)
                if kind in ("reference","subtitle"):
                    for run in para.runs:run.font.size=Pt(8 if kind=="reference" else 10)
                md.append(value+"\n")
    doc.core_properties.title="LipNet-AI: Fine-Tuning Before and After"
    doc.core_properties.subject="Frozen-feature transfer learning, paired accuracy and evaluation limits"
    doc.save(docx_path);md_path.write_text("\n".join(md),encoding="utf-8")

    pdf = PdfReader(pdf_path)
    text="\n".join(p.extract_text() for p in pdf.pages)
    assert len(pdf.pages)==6, f"Expected six PDF pages, got {len(pdf.pages)}"
    assert all(len(p.extract_text())>300 for p in pdf.pages)
    for expected in ("12.50%", "94.79%", "82.29", "704", "140", "0.60", "0.10", "not fine-tuned"):
        assert expected in text, expected
    reopened=Document(docx_path)
    assert len(reopened.inline_shapes)==2
    assert len(reopened.tables)==5
    evidence=[RUN/n for n in ("evaluation.json","paired_comparison.json","selection.json","protocol.json","rejection_policy.json","test_predictions.jsonl","preserved_scratch_test_predictions.jsonl")]
    receipt={"created_at":datetime.now(ZoneInfo("Asia/Calcutta")).isoformat(),"pdf_pages":len(pdf.pages),"paired_ids_verified":len(old_by),"top1_before":before,"top1_after":after,"delta_percentage_points":delta,"top5_correct":191,"confusion_matrix_verified":True,"classifier_hashes_verified":True,"new_training_run":False,"baseline_selected_train_rows":baseline_counts["train"],"encoder_finetuned":False,"rejection_config_matches_evaluated_policy":current_rejection["threshold"]==e["rejection_policy"]["threshold"] and current_rejection["margin"]==e["rejection_policy"]["margin"],"evidence_sha256":{str(p.relative_to(ROOT)):digest(p) for p in evidence},"outputs":{str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in (pdf_path,docx_path,md_path,chart,cm)}}
    (OUT/(NAME+"_validation.json")).write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps({"pdf_pages":len(pdf.pages),"outputs":[str(p) for p in (pdf_path,docx_path,md_path)],"paired_ids_verified":192,"rejection_config_mismatch_documented":not receipt["rejection_config_matches_evaluated_policy"]},indent=2))


if __name__ == "__main__":
    main()
