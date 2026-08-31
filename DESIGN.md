# Spoken-Alphabet Visual Speech Decoding — Research-Backed Design Document

**Project:** Alphabet-Centric Visual Speech Representation and Decoding
**Independent project** — does not modify the existing Word project.
**Deliverable:** Phase 1 (research + phenomenon definition). Version 1.0, 2026-08-23.
**Scope discipline:** ONE coherent architecture is defined. All switches mentioned are ablation/experiment toggles *inside* that architecture, never competing architectures. Every block has a stated scientific reason (§11).

---

## 1. The visual speech phenomenon we are modeling

A spoken letter is a **temporal articulatory gesture, not a static mouth shape**. Each letter *name* is a fixed phonetic string (Table 1 below); uttering it produces a coordinated **0.2–1.2 s deformation** of the lower face: lip closure and release, jaw lowering, lip spreading/rounding, tongue/teeth exposure, all ordered in time.

Three facts define the problem:

1. **One frame ≠ one letter.** The closed-bilabial state is visually identical for the closure of /b/, /p/, and /m/; many otherwise distinct letters share mouth configurations at the frame level. This is the classical viseme ambiguity (Fisher 1968; Bear & Harvey 2017). Most discriminative power lives in the *transient* signal: onset, closure, hold, release, and the movement into the following vowel ("transient visemes").
2. **Letter names have structure.** The 26 names are not independent images; they are short phonetic strings that cluster (Table 2): the /iː/ family B, C, D, E, G, P, T, V, Z; the /eɪ/ family A, H, J, K; the /ɛ/ family F, L, M, N, S, X; diphthong/juː families I, Y, O, U, Q, W; r-colored R. Within each family the discriminator is usually one consonant at onset or coda — a short, localized articulatory episode.
3. **Voicing is invisible.** /b/ vs /p/ vs /m/ and /d/ vs /t/ differ acoustically by voicing/aspiration/nasality, not by visible articulator configuration. The letter decision is *inherently* ambiguous at the phoneme level. A correct system models the posterior distribution over letters and over letter families, and can deliberately say "AMBIGUOUS".

Consequence: the system models a spatiotemporal sequence of **articulatory visual evidence** → **latent speech unit evidence** → **letter**, with an explicit uncertainty channel. It never pretends a frame is a letter.

## 2. What information is actually observable from the face

**Observable signals (recorded):**

| Signal | Visible content | Letters it best helps |
|---|---|---|
| Lip geometry | Mouth width, height, shape; spreading (/iː/, /eɪ/); rounding (/oʊ/, /juː/, /w/) | Vowels, diphthongs; W, U, O |
| Mouth aperture & jaw | Vertical opening, inside-mouth area, jaw drop/recover | Vowel height, plosive/nasal bursts |
| Intra-oral visibility | Teeth visibility, tongue tip between teeth (/θ/, /ð/), tongue body/shading inside aperture | S, X, F, L, N, plus consonants |
| Articulatory timing | Closure onset/hold/release of lips; nasal closure or held-lips; burst timing; jaw velocity profile | Releasing consonants, B/P/M vs F/V, plosives |
| Facial context (tested, not assumed) | Chin/cheek/jaw movement; speech-linked head micropose | Fine-grained disambiguation, e.g. bilabial puffs |

**NOT observable (hence ambiguity):** voicing (B vs P, D vs T, G vs K), nasality as source/register, amplitude, fundamental frequency, pitch. These limits bound what any visual system can recover and are exactly where the model must stay uncertain.

**Head pose:** informative (speech coupling) and nuisance (identity/appearance change); normalized by alignment (§3) and modeled as metadata (§9), with ablation on pose strata.
## 3. The proposed representation (the core research object)

Three streams derived in parallel from the same aligned video, all **per-frame and temporally ordered** (the model input is a T×D sequence, never T independent images), plus an optional context stream whose value is empirically decided, not assumed.

**1. Appearance stream — mouth pixels.** What the mouth *looks* like: lip contour, lip texture, teeth/tongue visibility, intra-oral shading, jaw appearance. Encodes a 64×64 RGB crop of the mouth region per frame (landmark-derived bbox + 2.2× margin; translation/rotation/scale normalized to a canonical frontal crop), through a compact ResNet-style stem (channels [32,64,128], LayerNorm, GELU, global average-pool) → frame embeddings **v_t ∈ ℝ^96**. This is the "what does the mouth look like" channel.

**2. Geometry stream — where each articulator is.**
- Per frame: 68 MediaPipe FaceMesh landmarks (inner+outer lips, jaw, perioral), normalized to a **mouth-centered coordinate frame with scale = inner-mouth width** (making distances and velocities comparable across speaker size and camera distance). Derive 20 canonical scalars: mouth width W, mouth height H, W/H ratio, inner lip opening, lip corner displacement (spread), upper-lip height, lower-lip height, jaw-to-mouth-center distance, aperture-area proxy, and their z-scored friends.
- MLP(coords + scalars → 96 → 64) → **g_t ∈ ℝ^64**. This stream captures explicit, interpretable articulatory geometry and is *modality-independent* of lighting/appearance.

**3. Motion stream — how the mouth is changing right now.**
- Smoothed first- and second-order temporal derivatives (Savitzky–Golay, window 5) of (i) the normalized landmarks, (ii) the 20 geometry scalars — i.e. ΔW, Δlip-opening, Δjaw-drop, Δlip-spread, Δlandmarks ⇒ velocities/accelerations of the articulators; plus a coarse optical-flow patch (16×16×2, Farnebäck) as an optional additional channel (default ON).
- Small temporal MLP/conv → **m_t ∈ ℝ^64**. This stream exposes closure/release/onset transients that static appearance cannot, i.e. the *transient-viseme* content (§1).

**Optional context stream — full face.** 112×112 full-face grayscale crop through a small conv tower → **c_t ∈ ℝ^64**. The research question "does chin/cheek/head-motion context help?" is tested, not assumed; this block exists only to be measured (§11, §13 ablation grid). It participates through the fusion gate; default ON with a separate ablations to show its delta.

**Fusion — learned gated fusion, not concatenation.** At each frame the fusion mechanism computes per-stream weights α_t = softmax(FPN(v_t; g_t; M_t; c_t)) and produces **f_t = α_v W_v v_t + α_g W_g g_t + α_m W_m M_t + α_c W_c c_t**, each W to ℝ^192. The network learns to *trust geometry and motion while lips are closed, appearance during vowel spread, etc.* — per timestep, per-sample usage of each representation. This replaces the unblinded concat.

## 4. Temporal representation (hierarchy)

The model explicitly constructs a cascade — frame → local mouth features → articulatory movement → letter-scale dynamics — so that onset→articulation→transition→release are represented:

1. **Frame level (L0):** per-frame streams (above) → f_t.
2. **Local temporal (L1, ≈ tens of ms–0.4 s):** stack of 1D temporal convolutions (kernel 7, stride 1, dilation 1 → 2 → 4) + LayerNorm + GELU + residual. Captures closure/hold/release, short articulatory transitions, without destroying temporal resolution (stride-1 convolutions preserve the timeline; this is the LipNet spatio-temporal principle in a GPU-friendly factorized form).
3. **Letter-scale context (L2):** a Conformer encoder (d_model 192, 4 conformer blocks: multi-head self-attention + depthwise-conv FFN, macaron FFN) with sinusoidal + learned positional encoding, giving each timestep **bidirectional long-range context** (what comes before + now + after) — offline mode. A causal-masked, chunked-streaming variant is used at live inference.
4. **Temporal positional/compression:** no compression before L2; a fixed output vocabulary position maps to a letter decision only at the final letter head. Utterance window length T=76 frames @ 30fps ≈ 2.5 s (rest → onset → articulation → release tail), with bucketed padding to length and masks for OOV.

## 5. The proposed latent speech units

The intermediate representation is **phonetic / visemic evidence per frame**, not direct letter scores. Letter names are fixed phone strings (Table 1), so phonetic supervision can be derived *for free* from the letter label (all training alignments run through CTC):

- **Latent unit A — Phonetic evidence head (primary):** linear layer over the temporal encoder output → **P_t ∈ ℝ^45** (44 phones ARPABET + [sil]), trained with **CTC** against the phone string of the letter name (e.g. B → B IY; F → EH F; Q → K Y UW; Table 1). This forces the encoder to localize each phonetic segment in time; the network learns phones as intermediate causes, producing interpretable "closure / release / vowel-spread" structure. It is the viseme-to-phoneme decoder's recognition space and exposes which phonemes the face can and cannot separate.
- **Latent unit B — viseme evidence (optional auxiliary head):** the same encoder output → viseme logits (viseme vocabulary, e.g. 13-class Bear–Harvey-derived map from Table 3) with a CTC/CE auxiliary. This *regularizer* groups visually confusable phones (BIL: s∈{B/P/M}; LD: {F/V}; ALV: {S;Z}; etc.), matches classic viseme categories, and — crucially — is *learnable* by the network. The viseme head is never used to hard-cascade phonemes→visemes; phones are the primary evidence that decoding exploits.
- **Uncertainty in latent space:** the CTC-soft posterior itself expresses ambiguity (e.g. B vs P vs M peak under the bilabial phone), and this soft evidence carries into the letter decoder, which is the mechanism by which "uncertainty about /b/ vs /p/" becomes "uncertainty between B and P" (§7).
## 6. The proposed decoding mechanism

The decoder is a **two-level sequence decoder over the temporal representation**, resolving the visual evidence into a letter *belief*, never into a forced single token at the per-frame level:

1. **Articulation gate (meta-decoder).** Mouth-motion energy (from the motion/gated stream) and the [sil]/[blank] mass of the phone evidence decide whether the window contains speech at all. Non-speech → UNKNOWN path (§7). This keeps the decoder honest (rejects silence/noise), and cleanly segments start/end of an utterance.
2. **Letter CTC head.** From the Conformer output H ∈ ℝ^T×192 → 27 logits (A–Z + CTC blank). CTC trains with the single letter label per utterance; decoding produces one letter (with blanks collapsed) per window. Because our annotation is a single token, this is a *temporal decision*; the CTC training dynamic enforces that the letter evidence emerges from a *sequence* of latent states, not a single frame.
3. **Letter evidence distribution (for uncertainty).** The *frame-averaged* letter logits (log-mean-exp of the per-frame softmax, temperature-calibrated) yield a posterior p ∈ Δ²⁶ per utterance used by the ambiguity engine (§7). Explicit: we **do not** choose the argmax of that pooling alone.
4. **Two-stage visual + language**, kept strictly separate: the visual stage (above) produces letter posteriors; a *lexicon stage* may then disambiguate *within* an ambiguity set for word building (§10) — it can never overwrite a confident visual letter, and it is disabled in the default scientific measurement.

The letter buffer, SPACE, DELETE, CLEAR, UNKNOWN, AMBIGUOUS all operate *below* the CTC stage on the confirmed-letters stream (§10), not inside the vision stack.

## 7. How ambiguity will be represented

Ambiguity is **first-class output data**, not an afterthought.

- **Posterior and statistics per utterance/window:** p ∈ Δ^{26} (calibrated), top-1/top-3/top-5, normalized predictive entropy h, confidence margin μ = p₁ − p₂, and **family probability** f = Σ_{c∈family} p_c for Table 2 families (e.g., bilabial-·/iː/ family {B,P,M}). All of these are stored in the prediction record.
- **Explicit decision rule (never forcing):**
  * no articulation detected → **UNKNOWN** (with reason tag),
  * p₁ ≥ τ_c (0.55, default) AND μ ≥ τ_m (0.20) AND h ≤ τ_h (0.85) → emit **top-1 letter**,
  * otherwise → **AMBIGUOUS** with {top-5 candidates, probabilities, family label, and the frame region of maximal evidence};
  * if p₁ is flat and entropy near max, **ABSTAIN** (= no letter).
- **Calibration.** Temperature scaling fit on speaker-independent validation; reliability diagrams + Expected Calibration Error per letter and per family are part of every run (§12).
- **Uncertainty streaming.** Across windows, posteriors are fused by an exponential-decay weighted average; a letter is *committed* only after κ (default 3) consecutive windows with mathematically stable top choice. In all cases the system emits its belief (set), never a hallucinated winner — matching the "uncertain preserving" requirement.

Buffer memory: each committed letter carries p, entropy, timestamp, candidates — usable by the word/lexicon stage and by failure analysis (§12).
## 8. How data should be collected

The atomic unit is the **uttering recording** (speaker × letter × session × repetition), never a frame or a session-agnostic frame pile.

- **Collection protocol (capture app):** a guided webcam tool shows the letter (randomized order per session), invites a natural voiced utterance ("letter B, beey"), records 720p30 (mono audio stored, optional, used only as optional training supervision — never in inference), tracks the face live (MediaPipe), keeps the crop+landmarks aligned to the raw video, and writes:
  `alphabet_dataset/<LETTER>/<speaker>/<session>/<recording>/video.mp4`, `frames/` (only if storage affordable; else derived on demand), `landmarks.npz`, `metadata.json`.
- **Metadata JSON:** letter; speaker_id; session_id; recording_id; utc timestamp; fps; frame_count; resolution; head-pose (pitch/yaw/roll from landmarks); mouth bbox (per-frame summary); per-frame 68+48 landmark arrays; quality: Laplacian blur energy, brightness mean/std, contrast, duplication hash; lighting tag; accent/language tag (which variant, e.g., "R" rhotic/non-rhotic, "Z" zee/zed); normalization reference (mouth width at rest).
- **Collection plan:** ≥ 30 speakers (any age/gender), ≥3 sessions across ≥2 weeks, ≥8 repetitions of each letter per session ⇒ target build 26 × 30 × 3 × 8 ≈ 18,720 utterances, preferred ≥50,000. Vary distance (1.2–3.0 m), azimuth (±25°), lighting (3 levels), glasses/beard/makeup, rate, loudness; record each face's changes naturally alongside head movement (do not ask robotic fixation beyond the first session).
- **Splitting — absolutely speaker-independent.** Speakers split 70/15/15 by speaker ID; sessions never span splits; no frame shuffling, no same-recording across splits; additionally session-independent: within the val speakers, val sessions never share with train sessions.
- **Derived views only, never destructive cleanup:** dedup frames *within* a recording for storage (near-duplicate suppression for disc savings) but keep original video file; augmentations are applied on-the-fly at training (sequence-level, §9), never stored as new samples.
- **Public data (training pool for Stage A self-supervised only):** we may add mouth-crops from lses datasets (LRS2/LRS3-type) or synthesized faces ONLY for self-supervised pretraining (not eval), each recording tagged as synthetic/pseudo-labeled; excluded from test-time lineage; quality gates applied.

## 9. How the model should learn

**Stage A — self-supervised pretraining ("what does visual speech look like?"), no letter labels:**
1. **Masked frame prediction (primary):** cover ~40% of frame positions (structured runs); the visual encoder + Conformer predicts the masked block embeddings (InfoNCE/normalized MSE) — the AV-HuBERT masked-prediction principle, reimplemented for our own encoder (no offline clustering dependency), teaching *motion-causal* inference from context. Trained on talk videos (mouth-crops).
2. **Masked appearance reconstruction (aux):** patch-level MAE on the mouth image (the appearance stem) — cheap self-supervision for the shallow CNN.
3. **Audio-assisted (optional, training-only supervision):** with optional audio present, predict mel-spectral/acoustic features from masked visual (LiRA/AV-HuBERT pattern) — audio is **never required at inference**; strict gate marking it is always OFF in eval.

**Stage B — supervised multi-objective training on the alphabet corpus:**
- **Primary loss:** CTC on phonetic evidence (45 classes) + CTC on letter (27).
- **Auxiliary:** (i) pooled letter CE (arg supervision on the post-Conformer features); (ii) appearance↔geometry consistency: predict the 20 geometric scalars from the appearance stem (aux LINEAR head, hinge MSE) — this *ties* the appearance to interpretable articulatory coordinates, and yields a measurable appearance-vs-geometry trace; (iii) viseme CTC (optional §5); (iv) fine: temporal frame-rate invariance (0.8×/1.25× resampling same-letter contrastive/consistency loss); (v) gradient penalty-free.
- Weights are tunable knobs in config; default λφ/λℓ/λgeo/λviseme/λfr = 1.0/0.8/0.3/0.2/0.1.
- **Augmentation — sequence level only**, never per-frame independent (per-frame rotation would fabricate motion): global small translation (≤3px), rotation (≤4°), scale (0.9–1.1), brightness/contrast/gamma, mild blur σ≤1.2, temporal gap/dropout (≤8% frames), temporal (strided) resampling (0.8–1.25×), light/noise.
- **Optiminfra:** PyTorch; AdamW; cosine schedule + 10% warmup; gradient clip 1.0; AMP (BF16 on Ampere+); bucketed padding batching; tf.data-like background prefetch; reproducible seeds (all); early-stopping on speaker-independent MACRO-F1; resume, best-ckpt save.
- **Checkpoint content** (everything needed to reproduce inference): model {weights, class map}, normalization/alignment parameters, dataset version hash, training config (yaml), representation config, decoder config, random seed, augmentation snapshot.

**Decay of methods:** report learning curves, per-family breakdown, and — critically — the results of the ablation grid (§13, Table 5) that isolates the contribution of each of: appearance, geometry, motion, context stream, phone-supervision, viseme auxiliary, self-supervised initialization, and corpus size. These are answered-the research questions, not extras.
## 10. How the system will decode A–Z end-to-end

**Offline evaluation pipeline (single utterance):**
video → face tracking → mouth alignment (§3 normalization) → three-stream representation → spatial-temporal encoder (§4) → Conformer (§4) → phone/viseme evidence (§5) → CTC letter head (§6) → calibrated posterior + ambiguity engine (§7) → `LETTER | AMBIGUOUS{...} | UNKNOWN`.

**Live pipeline (thread-safe, never blocked):**
```
camera thread ──> face tracker (MediaPipe, its own thread)
      └─► rolling temporal buffer (bounded ring of the last T=76 aligned frames, stored as crop + landmarks, plus coarse per-frame mouth-motion to mark "speaking")
              └─► inference worker (models preloaded once; reads bounded queue, drops stale frames, never blocks camera)
                      └─► periodic window inference: representation ± encoder ± decoder ⇒ letter posterior
                              └─► uncertainty/confirmation gate (κ consistent windows, §7)
                                      └─► letter buffer + command engine ─► word composition UI
```
Invariants: no model load in the inference thread; bounded queues; stale-frame drop (never backlog); window stride ≤ 3 frames (overlap ≈ 96%).

**Letter buffer & word composition:**
- committed letters accumulate with their (p, entropy, timestamp, top-5);
- **duplicate suppression** (a letter is added once until a *next* distinct letter is committed with margin);
- **letter timeout** (after t_idle ms, the window closes; next articulatory event opens a new letter);
- commands: SPACE, DELETE (last), CLEAR, plus special outputs UNKNOWN / AMBIGUOUS{candidates}.
- **Word composition (two-stage language use):** the letter buffer is first displayed as raw letters; a *lexicon-only* stage (e.g., small spellable-word list) may then propose completions *solely within* AMBIGUOUS sets: `H E L {L/P/M}` → candidates {HELLO, HELP, HELM} — shown as suggestions, applied explicitly, and always labeled as "language-assisted" in the transcript, keeping visual recognition scientifically separable from ideological context. This stage is disabled by default in measurement.

Output statement per letter is always `letter, p1, {top3}, entropy, margin, family, source(visual|lexicon)`.

## 11. Why each block exists

| Block | Why it exists (scientific/engineering reason) |
|---|---|
| Face detection & tracking | Defines the ROI from which *all* articulatory evidence is extracted; failure here corrupts every downstream measurement (evaluation metrics must treat it as gate). |
| Mouth alignment/normalization | Makes the representations comcomparable across pose, distance, identity; removes nuisance variance (Chung & Zisserman). |
| Appearance stream | Captures intra-oral & lip-visible articulators (teeth/tongue/shading) and vowel spread; the historical driver of lip-reading accuracy. |
| Geometry stream | Explicit articulatory configuration (width/height/opening/jaw) that is invariant to lighting/appearance; gives interpretable landmarks to probe "which geometry is used". |
| Motion stream | Encodes closure/release/onset velocity — the transient-viseme info static frames lack (Bear & Harvey). |
| Context stream (tested) | Enables the *research question* "does full-face context help mouth-only?" via ablation; default gated in, measured out. |
| Gated fusion | Learns per-timestep weighting of streams (motion during closure, geometry during rounded vowels, appearance during spread) instead of blind concat. |
| Temporal conv stack (L1) | Local articulatory dynamics; preserves temporal resolution (LipNet principle, factorized). |
| Conformer (L2) | Long-range bidirectional temporal context; the encoder family dominating current VSR (Gulati et al.; Chang et al.). |
| Phonetic evidence CTC | Dense, label-free phonetic supervision; removes the "26 classes memorized" shortcut; makes ambiguity *in the unit space* visible. |
| Viseme auxiliary | Regularizes toward articulatory similarity; interpretable confusability chart; supports the classic viseme question empirically. |
| Letter CTC + posterior | Sequence-level letter decision decoupled from frames; gives p, top-k, entropy for calibration/ambiguity. |
| Ambiguity engine | Turns posterior into decision that is allowed to be AMBIGUOUS/UNKNOWN — the honest answer under invisible voicing. |
| Letter buffer/composition | Time-domain statefulness: one gesture = one letter confirmed by κ windows; enables word building; separation of visual vs lexical data. |
| Two-stage (visual then lexicon) | Keeps the science interpretable; language never injects visual hallucination. |
## 12. Relevant recent research supporting each design decision

Full reference list (consulted 2026-08, arXiv IDs given):

**R1. LipNet** — Assael, Shillingford, Whiteson, de Freitas, *LipNet: End-to-End Sentence-level Lipreading*, arXiv:1611.01599 (2016). Spatiotemporal convs + BiGRU + CTC; 95.2% sentence accuracy on GRID with *overlapped* speakers. Supports: sequence-level modeling & CTC end-to-end; also the known fact that unseen-speaker generalization is substantially harder (motivates our strict splits).
**R2. AV-HuBERT** (Shi, Hsu, Lakhotia, Mohamed, ICLR 2022, arXiv:2201.02184). Self-supervised masked multimodal cluster prediction; lipreading WER 26.9% on LRS3 vs 33.6% for a system trained on ~31k h; audio used in pretraining only. Supports: Stage A masked frame prediction (we reimplement the principle without the clustering machinery) and train-time-only audio.
**R3. LiRA** (Ma, Mira, Petridis, Schuller, Pantic, Interspeech 2021, arXiv:2106.09171). Audio→visual self-supervised pretraining by predicting acoustic features; visual-only at inference. Supports: optional audio-assisted Stage A; justifies "audio never at inference".
**R4. Conformer** (Gulati et al., NeurIPS 2020, arXiv:2005.08100) — conv-augmented transformer standard. **R5.** *Conformers are All You Need for VSR* (Chang et al., 2023, arXiv:2302.10915) — light front-end + strong Conformer is the modern pattern (linear/ResNet front-end suffices). **R6. AVEC** (Burchi & Timofte, 2023, arXiv:2301.01456) — ResNet-18 front-end + Efficient-Conformer + inter-CTC supervision.
**R7. VALLR** (Thomas, Fish, Bowden, 2025, arXiv:2503.21408). Two-stage Visual ASR: video → phoneme string (CTC) → LLM; LRS3 WER 18.7 with 99.4% less labeled data. Mirrors our phoneme-core + separate language stage. **R8. VALLR-Pin** (2025) extends the same two-stage principle to Mandarin w/ Pinyin, confirming the trend.
**R9.** Bear & Harvey, *Phoneme-to-viseme mappings: the good, the bad, and the ugly* (Speech Communication 2017; arXiv:1805.02934); also *Which phoneme-to-viseme maps best improve visual-only computer lip-reading* (2014) and *Decoding visemes* (arXiv:1710.0128) — viseme maps many-to-few, context-dependent, *transient* events carry info; data-derived maps ("Bear visemes") beat hand maps. Supports: our learned phonetic evidence + data-driven viseme head rather than fixed phoneme→viseme priors.
**R10.** TVTA (2026, arXiv:2607.08236) event-based lip reading: trajectory-aware, viseme-guided temporal aggregation with a CTC decoder — evidence that viseme-guided CTC structures the timeline.
**R11.** VisG AV-HuBERT (Papadopoulos, Jain & Harte, ICPR 2026, arXiv:2604.00982): auxiliary viseme classification regularizes audio-visual HuBERT (WER 13.59→6.60 at −10 dB), improving speech-unit discrimination. Supports our viseme auxiliary head.
**R12.** Chung & Zisserman, *Lip Reading Sentences in the Wild* (2016)/*Learning to Read Lips from Limited Data* — mouth-crop primacy. **R13.** Oghbaie et al., *Advances and Challenges in Deep Lip Reading* (2021, arXiv:2110.07879) — survey: data-scale & speaker diversity dominate.
**R14.** Tran et al., *A Closer Look at Spatiotemporal Convolutions for Action Recognition* (2018) — justification of the (2+1)D-style factorization (space then time).
**R15.** Ephrat et al., *Looking to Listen at the Cocktail Party* (SIGGRAPH 2018) — attention-gated audio-visual fusion precedent.

**Plain-language takeaways that shaped the design:**
1. Sequence-level CTC decoding beats per-frame classification (R1).
2. Phonetic-intermediate supervision (VALLR) is the current recipe for data-efficient, speaker-robust visual decoding (R7, R8).
3. Viseme ambiguity is real, context-dependent, and transient; the model must represent it, not hide it (R9, R1); viseme-guided auxiliary heads measurably help (R10, R11).
4. Light visual front-end + strong Conformer encoder is the modern fit (R5, R6).
5. Self-supervised / audio-assisted pretraining lowers data need drastically (R2, R3).
6. Diverse speakers, not frames, bound generalization (R12, R13).

No claim of global uniqueness or state-of-the-art is made; per spec, any such claim must be re-checked against the literature at the time of results.
## 13. Exact implementation plan

Build order per spec §28. PyTorch codebase (CTC/attention and pretraining APIs are native to PyTorch; this choice is recorded in configs). Every phase has a gate and produces runnable, tested code.

- **Phase 1 — Research + phenomenon definition.** This document; research notes with arXiv IDs; risk register. *[DONE]*
- **Phase 2 — Dataset format + recording metadata.** `alphabet_dataset/<LETTER>/<speaker>/<session>/<recording>/` with `video.mp4`, `landmarks.npz`, `metadata.json`; `data/schema.py`; `data/recorder.py` (guided capture app); `data/quality_report.py` (blur/brightness/pose/dup checks); `data/splits.py` — speaker-independent with session-leakage assertions. **Gate:** 3-speaker pilot dataset; schema validates.
- **Phase 3 — Face tracking + mouth alignment.** `tracking/aligner.py`: MediaPipe FaceMesh per frame; mouth-crop normalization (§3); motion-energy VAD; head-pose estimation. **Gate:** alignment RMSE report.
- **Phase 4 — Representations.** `repr/appearance.py`, `repr/geometry.py`, `repr/motion.py`, `repr/fusion.py` (gated fusion); stream toggles per config. **Gate:** stream traces semantically meaningful (lip-opening ↔ vowel).
- **Phase 5 — Temporal sequence loader.** `data/sequence_loader.py`: windowing, bucketed padding, masks, sequence-level augmentation, caching/prefetch. **Gate:** deterministic replay.
- **Phase 6 — Spatial–temporal encoder.** `model/frontend.py` (appearance stem) + `model/encoder.py` (L1 temporal convs + L2 Conformer). **Gate:** forward/backward shapes, AMP safe.
- **Phase 7 — Temporal representation model.** The Conformer backbone (shares Phase 6 file); depth/width via config.
- **Phase 8 — Latent visual speech representation.** `model/latent.py`: phonetic evidence head (45 logits, CTC), viseme aux head, letter-name→phone dictionary (ARPABET) with accent variants.
- **Phase 9 — Sequence decoder.** `model/decoder.py`: letter CTC head (27) + pooled posterior + articulation gate.
- **Phase 10 — Ambiguity estimation.** `model/uncertainty.py`: temperature calibration, entropy/margin rules, AMBIGUOUS/UNKNOWN emitter, streaming κ-window confirmation. **Gate:** unit tests on synthetic posteriors.
- **Phase 11 — Training.** `train.py`, `configs/*.yaml` (all §9 losses & toggles), AMP/bf16, cosine + warmup, grad-clip, checkpoint/resume, seeded determinism. **Gate:** 1 GPU run converges on pilot.
- **Phase 12 — Speaker-independent evaluation.** `eval/eval_metrics.py`: accuracy, CER, macro-F1, top-1/3/5, ECE/reliability, confusion matrix, family accuracy, per-speaker; ablation grid (Table 5 below). **Gate:** results recorded on an unseen-speaker test set.
- **Phase 13 — Failure analysis.** `eval/failure.py`: per-misclassification record (true/pred/topk/conf/speaker/length/pose/quality/temporal region) with attribution {appearance, motion, geometry, pose, ambiguity}.
- **Phase 14 — Live streaming inference.** `app/live.py`: camera thread, tracker thread, bounded ring buffer, inference worker (preloaded models), stale-frame drop, κ-confirmation. **Gate:** latency + no thread-block.
- **Phase 15 — Word composition.** `app/composer.py`: letter buffer, SPACE/DELETE/CLEAR, timeout, duplicate suppression, optional lexicon-only word resolution (two-stage), word display.

**Ablation / experiment grid (all on the same unseen-speaker test set):**
E0 full system; E1 −motion; E2 −geometry; E3 −appearance; E4 −context stream; E5 −phonetic-CTC; E6 −viseme head; E7 −self-supervised Stage A; E8 −audio-assist; E9 small-data (15 speakers); E10 augmentation sweep; E11 streaming-causal mode.
Metrics per row: accuracy, CER, sequence acc, macro-F1, ECE, top-k, family error, per-speaker spread (§21), failure attribution (§26).

**Definition of success (honest):** (a) speaker-independent accuracy above strong frame-class baselines; (b) quantified marginal value of each stream (which face information helps — the core research question); (c) calibrated, behavior-plausible uncertainty with B/P/M-family demonstrably among the harder families; (d) full reproducibility from config + seed; (e) no fabricated confidence — AMBIGUOUS/UNKNOWN outputs audited on held-out spot-checks.

**Infra & constraints:** Nvidia GPU ≥8 GB; Python ≥3.11; MediaPipe, OpenCV, PyTorch ≥2.2; logs to wandb or JSON. Any external pretrained model (e.g., AV-HuBERT) is only adopted after license, compatibility, VRAM, input-format and temporal-resolution checks (spec §15); the default path trains from scratch with our own Stage A. No UI work before Phase 13. No modification of the existing Word project — this project lives in its own workspace directory.
## Appendix — Reference tables used throughout

**Table 1 — Letter name → phone string (ARPABET; canonical US default; accent variants availated in config)**
| Letter | Phones | Letter | Phones |
|---|---|---|---|
| A | EY | N | EH N |
| B | B IY | O | OW |
| C | S IY | P | P IY |
| D | D IY | Q | K Y UW |
| E | IY | R | AA R |
| F | EH F | S | EH S |
| G | JH IY | T | T IY |
| H | EY CH | U | Y UW |
| I | AY | V | V IY |
| J | JH EY | W | D AH B AH L Y UW |
| K | K EY | X | EH K S |
| L | EH L | Y | W AY |
| M | EH M | Z | Z IY (or, if zed accent: Z EH D) |

**Table 2 — Letter families (shared phonetic structure)** — used for family-level confidence, confusions and AMBIGUOUS labels:
- `/iː/ tail:` B, C, D, E, G, P, T, V, Z
- `/eɪ/ onset:` A, H, J, K
- `/ɛ/ initial:` F, L, M, N, S, X
- `/juː/:` Q, U, W; `/aɪ/:` I, Y; `/oʊ/:` O; `/ɑr/:` R
- Bilabial onset: B, P, M; labiodental: F, V; sibilant: C, S, Z

**Table 3 — Viseme label topology (initial; final grouping is re-fit from the model's own phone-confusion matrix — data-driven "Bear visemes", R9):**
V1 {B,P,M}; V2 {F,V}; V3 {TH,DH}; V4 {S,Z}; V5 {SH,ZH,CH,JH}; V6 {T,D,N}; V7 {K,G,Y}; V8 {R}; V9 {L}; V10 {W}; V11 front vowels {IY,IH,EH,AE}; V12 back/rounded vowels {AA,AO,OW,UW}; V13 diphthongs {EY,AY,OY,AW}; plus [SIL]. The exact grouping is learned from the empirical phone-confusion matrix rather than fixed a priori; this table only initializes the label topology (R9).
