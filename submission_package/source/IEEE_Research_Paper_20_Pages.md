# Abstract

Visual word recognition must be evaluated against the vocabulary, speakers, input geometry, and training protocol that produced its predictions. This paper presents a CPU-executable research system combining measured mouth alignment with frozen AV-HuBERT representations and a supervised linear classifier. The principal experiment uses eight resolved MIRACL-VC1 word labels. A validation-only search over three encoder layers, three temporal pooling functions, and three regularization settings selects layer 9 with concatenated temporal mean and standard deviation. On 192 specified test utterances, the selected model obtains 182 correct predictions, 94.79% top-1 accuracy, 99.48% top-5 accuracy, and 0.9467 macro F1. A preserved scratch 3D-CNN–BiGRU baseline obtains 24/192 correct predictions on the same source utterances, giving an 82.29-percentage-point system-level difference. The comparison is not an equal-data or single-variable ablation. The encoder remains frozen throughout downstream training. Additional repository components include an isolated alphabet classifier, a publisher 500-word model, and an experimental CTC path; their capabilities are evaluated separately. This extended manuscript reports dataset provenance, preprocessing, model selection, complete confusion matrices, per-class and per-speaker results, recorded CPU costs, and implementation limitations. Physical-webcam accuracy, continuous transcription, language-model correction, and speech synthesis are not established by the available evidence.

**Index Terms—** Visual speech recognition, lip reading, AV-HuBERT, frozen-feature transfer learning, isolated words, CPU inference, reproducible evaluation.

# I. Introduction

Visual speech recognition maps a sequence of facial or mouth images to linguistic labels. The problem is difficult because the camera observes only part of the speech-production process. Tongue position, voicing, and other acoustic distinctions may be weakly represented or invisible. Differences in pose, facial appearance, recording conditions, and speaking duration further change the input distribution. A useful system therefore needs both a temporal representation and an evaluation protocol that distinguishes recognition evidence from interface behavior.

This work investigates limited-vocabulary recognition on a Windows computer with an Intel i3 CPU and approximately 7.74 GiB of memory. The motivating engineering constraint is to reuse a strong representation without repeatedly optimizing a large neural encoder. The implemented solution extracts frozen AV-HuBERT features and fits a small classifier. Its central research question is whether this complete pipeline improves over an existing small scratch model on the same real source clips. The question is narrower than demonstrating universal lip reading or a new state-of-the-art benchmark.

The principal contribution is an auditable integration and comparison. Source identifiers, speaker partitions, preprocessing constants, checkpoint fingerprints, and per-clip predictions are preserved. These artifacts allow the reported metrics and confusion matrix to be independently reconstructed. The system also maintains distinct alphabet, eight-word, and 500-word routes, preventing one task’s measured accuracy from being attached to another model.

Three aspects make the study useful for implementation-focused research. It records the exact spatial and temporal contract required by each checkpoint, evaluates a small supervised head without updating the pretrained encoder, and exposes errors through class and speaker decompositions rather than a headline score alone. The accompanying package reconstructs plots directly from the saved predictions. This connects the methodological description to inspectable numerical evidence and makes subsequent improvements comparable with the preserved result.

The remainder of the manuscript describes related methods, data selection, spatial and temporal processing, model geometry, optimization, quantitative results, selective prediction, CPU execution, software integration, and limitations. Detailed experimental records and reconstruction instructions accompany the analysis. Although the original project motivation included contextual correction and generated speech, executable LLM and TTS components were not found in the audited implementation. They are discussed only as future extensions.

<!-- PAGE -->
# II. Related Work and Scope of Comparison

## A. Sentence and word recognition are different tasks

LipNet combines spatiotemporal visual processing, recurrent modeling, and connectionist temporal classification to map video to text [1]. Its sequence-level objective differs from a classifier that assigns one word to an already segmented clip. This distinction matters because a fixed-vocabulary classifier can be evaluated with a confusion matrix, whereas continuous decoding must account for insertions, deletions, and substitutions. The presence of a recurrent layer alone does not turn a clip classifier into a sentence recognizer.

CTC permits optimization over alternative alignments between a frame sequence and a shorter target sequence [2]. It introduces a blank label and a collapse operation. In the present repository, loss and greedy-decoding utilities exist, but the deployed alphabet checkpoint returns a single 26-class vector. This paper consequently does not attach a sentence-recognition claim to the alphabet interface or use utility tests as evidence of trained decoding.

## B. Pretrained visual speech representations

AV-HuBERT learns representations through masked multimodal cluster prediction [3]. The local implementation reuses its official Base checkpoint as a frozen feature extractor. Audio is not supplied during the reported downstream inference. The important connection to prior work is representation transfer: the large encoder has already learned speech-related structure, while the local dataset supplies supervision for a small fixed vocabulary. The study evaluates the local downstream classifier, not a reproduction of the publisher’s original pretraining experiment.

Temporal convolutional lip-reading models offer another route to isolated-word recognition [4]. The repository includes a publisher ResNet18/DC-TCN implementation and checkpoint. The author repository also documents later training strategies [5]. Because the local checkpoint is a Dense TCN variant, it is identified by its architecture configuration and artifact hash rather than being assumed identical to every model described in the earlier TCN paper. Loading a public model is a useful integration result, but its published benchmark score is not a local measurement.

## C. Dataset relevance

MIRACL-VC1 provides synchronized RGB and depth speech recordings [6]. Only RGB images are used here. GRID provides constrained sentence material and word timing annotations [7]. LRW is an isolated-word benchmark with 500 classes [8]. These resources support different scientific questions. GRID letter-slot classification tests discrimination of a known position in a constrained sentence, while LRW tests words occurring in broadcast-derived clips. MIRACL’s selected eight-word vocabulary is much smaller. Cross-dataset accuracy rankings would therefore be misleading without matched protocols.

## D. Contribution and novelty boundary

The study does not introduce AV-HuBERT, CTC, convolution, recurrent units, landmark tracking, or logistic regression. Its contribution is a reproducible CPU pipeline and a transparent account of what the available artifacts establish. The primary comparison changes several system components simultaneously, including representation, input normalization, sequence length, and the amount of supervised training material. It is reported as a system comparison rather than as proof that a single architectural change caused the improvement.

The source audit also provides a practical methodological contribution: it separates implementation, trained weights, configured experiments, recorded measurements, and unsupported claims. This separation prevents a vocabulary file from being treated as evidence of a dataset, a dataset from being treated as a trained model, and a successful endpoint response from being treated as an accuracy benchmark. These distinctions are essential when a research workspace contains multiple generations of experiments.

| Task | Output unit | Evidence required |
|---|---|---|
| Alphabet clip | One A–Z label | 26-way checkpoint and labeled clips |
| Isolated word | One vocabulary item | Ordered class map and word test set |
| Continuous text | Variable token sequence | Trained decoder and sequence references |
| Context correction | Revised text | Raw/corrected paired evaluation |

## E. Questions answered by the present study

The available artifacts answer three concrete questions. First, can the official encoder execute on this Windows CPU and produce finite features from authentic tracked video? Recorded extraction and API smoke evidence support that claim. Second, can a supervised head discriminate the eight retained words on the specified speaker partition? The 192 saved predictions provide direct evidence. Third, can these artifacts coexist with other research models without conflating their label maps? Separate model adapters and endpoints implement that distinction.

The artifacts do not answer whether the system outperforms published methods on their original benchmark splits, whether a person using a physical webcam will obtain the same score, or whether contextual correction improves transcription. These are different experimental questions. This framing avoids using the literature as a substitute for missing local evidence while still explaining why pretrained representations and temporal models are appropriate technical foundations.

The paper is intentionally an extended empirical and implementation study. Its findings can guide a subsequent controlled comparison, but claims of a new foundational architecture, clinical efficacy, or universal speech reconstruction would require different evidence. Published methods are credited, and local integration choices are identified explicitly.

<!-- PAGE -->
# III. Data Provenance and Corpus Inventory

## A. Primary MIRACL data

The author-provided MIRACL resource contains RGB and depth image sequences. The present pipeline uses original RGB frames and ignores depth and audio. The local resolved manifest contains 2,700 entries: 1,200 isolated-word utterances and 1,500 phrase utterances from 15 speakers. Each decoded RGB image is expected to have shape 480×640×3. The manifest records native frame rate as unknown; the pipeline does not invent a capture rate merely because it later interpolates sequences to a fixed number of frames.

Eight word labels are retained: choose, connection, hello, navigation, next, previous, start, and stop. Two author label families have unresolved naming inconsistencies and are excluded rather than silently renamed. This exclusion changes the task from the author resource’s ten-word description to a locally defined eight-class experiment. Every result in this manuscript uses the latter definition. The complete retained word manifest has 150 utterances per class before the experimental quarantine and tracking exclusions.

## B. Evidence hierarchy

The primary evidence consists of source manifests, checkpoint tensors, stored prediction rows, selection records, and evaluation JSON. Narrative reports are supporting material. This ordering is necessary because a long-lived workspace may contain documentation produced before data acquisition or after later code changes. The paper therefore reconstructs result counts from prediction files instead of copying a headline accuracy without checking its denominator.

Source fingerprints identify data and model versions. A locally calculated SHA-256 hash demonstrates that two inspected files have matching bytes; it does not independently prove authenticity or replace a publisher-supplied checksum. Acquisition records and official source links establish provenance separately. The distinction is important for pretrained artifacts whose publisher supplies a download but no comparison checksum.

## C. Other locally available resources

The GRID preparation contains 33,000 source videos for all 33 speakers with available video. The saved inventory records 32,891 usable video/annotation pairs and 109 exclusions for premature decoding termination. GRID’s fourth content token is used in a separate letter-slot manifest. Its 25 labels omit W, so that task cannot supply a complete 26-class alphabet head without additional labeled data.

The authorized full LRW dataset is absent. The local vocabulary contains 500 unique labels and the publisher model has 500 outputs, but neither establishes that the corpus was downloaded. The local `data/lrw` development directory contains 45 MP4 files with only one distinct byte hash across its nominal splits. These fixtures are unsuitable for recognition evaluation. One genuine public Oxford AFTERNOON clip is available for execution checks.

| Resource | Local evidence | Role in this paper |
|---|---|---|
| MIRACL resolved words | 1,200 utterances, eight labels | Main experiment |
| MIRACL phrases | 1,500 manifest entries | Small rejection diagnostic subset |
| GRID | 32,891 usable pairs | Separate corpus/integration study |
| Complete official LRW | Not present | No local full benchmark |
| Public AFTERNOON | One genuine clip | Publisher-model smoke test |
| LRW development fixture | 45 byte-identical videos | Excluded from accuracy |

## D. Research-use boundaries

Corpus and checkpoint terms must be retained with any reproduced experiment. The official LRW download requires the applicable BBC Research & Development agreement [8]. Publicly available model weights do not confer access to that corpus. The author-provided MIRACL page describes research use [6]. This manuscript reports a research implementation and does not infer commercial deployment rights, redistribution permissions, or clinical suitability from technical availability.

The validated source manifest and explicit exclusions make the experiment inspectable. They also expose what remains unknown: recording frame rate for MIRACL, performance on new cameras, and generalization beyond the selected word inventory. Those unknowns are retained as limitations rather than replaced with assumptions.

## E. Input probes and storage records

Three representative videos were decoded during the audit. The genuine GRID sample has resolution 360×288, 25 FPS, 75 decoded frames, and duration three seconds. The public Oxford word sample is 256×256 with 29 frames at 25 FPS, or 1.16 seconds. The duplicate local LRW fixture is 46×140 with 29 frames at 25 FPS. Matching nominal duration does not make the fixture authentic; source identity and repeated-byte detection override such superficial similarities.

The GRID acquisition records approximately 13.59 billion archive bytes and 13.87 billion extracted video/alignment bytes across all available speakers. Archives were retained for integrity and repeatable extraction. The source report distinguishes these logical file sizes from filesystem allocation and pre-existing audio. No data deletion or fresh bulk download was needed to prepare this paper.

For reproducibility, a video-decoding pass and an archive checksum answer different questions. A valid archive can contain a video that the installed decoder cannot completely read. Conversely, a fully decoded stream may have a container frame-count estimate that is off by one. The preparation policy records actual decoded length and exclusion reason, rather than relying only on file existence or a nominal header count. This is why the usable corpus count is reported separately from the 33,000 source files.

<!-- PAGE -->
# IV. Experimental Partition and Leakage Controls

## A. Speaker assignment

The principal experiment separates speakers between training, validation, and test. Training uses F01, F02, F04, F05, F06, F07, M01, M02, and M04. Validation uses F08, F09, and M07. The specified test speakers are F10, F11, and M08. These sets contain nine, three, and three speakers respectively. The training partition supplies scaler statistics and supervised head fitting; validation supplies model selection and rejection-policy selection; test supplies final known-word and phrase-diagnostic results.

The protocol initially plans 720 training word utterances and 240 validation word utterances. Of the 240 possible word utterances on the three test speakers, repetitions 01–02 are quarantined, leaving repetitions 03–10 and 192 test utterances. Each retained test class has 24 examples: eight repetitions from each of three speakers. This balance makes macro recall equal overall accuracy, but precision still depends on which classes receive false positives.

@figure split_counts | Fig. 1. Word-utterance accounting. The training bar includes 704 usable clips and 16 recorded tracking exclusions. Quarantined test repetitions are excluded from all reported 192-clip test metrics.

## B. Pilot exposure and its consequence

The 48 quarantined utterances were used in an earlier pilot. The final test speaker identities had therefore already been observed in prior project work. Removing the exact pilot utterances prevents their direct reuse in the reported test, but it does not create a completely untouched speaker population. We describe the result as a fresh-utterance evaluation on the original held-out speakers. Claims of fully prospective population generalization would require an additional unseen cohort.

## C. Tracking exclusions and denominators

Sixteen training word utterances fail the recorded tracking criterion, leaving 704 usable training examples. All 240 validation words and all 192 test words are usable. Tracking failures remain explicitly recorded; they are not replaced with generated crops. The evaluation code counts an unusable evaluation clip as incorrect/rejected rather than removing it from the denominator. Thus the absence of test tracking failures in this run is a measured property of the retained clips, not an implicit filtering advantage.

Sixty validation phrases and sixty test phrases are separately selected for rejection diagnostics. They do not become supervised word labels. The total prepared workload is 1,272 clips: 720 training words, 240 validation words, 192 test words, and 120 diagnostic phrases. After the sixteen training failures, 1,256 clips yield cached features. This arithmetic reconciles extraction timing with the classification partition sizes.

## D. Locked selection policy

The experiment records the source manifest hash, speaker assignments, preprocessing constants, layer candidates, seed, and quarantined identifiers before head selection. The selected head is written with its vocabulary and feature geometry. Test rows are not consulted by the head-search loop. The selection record explicitly stores `test_used=false`. The manuscript build additionally verifies that paired baseline and transfer prediction identifiers match and that none overlaps the quarantine list.

These controls support the reported comparison, but they do not eliminate all bias. Twenty-seven candidates share the same validation speakers, and the experiment constitutes one dataset/protocol realization. Repeated random seeds, an untouched speaker cohort, and matched-data ablations would strengthen the evidence. No such additional measurements are claimed here.

## E. Partition arithmetic and source units

The planned 720 training words arise from nine speakers, eight words, and ten repetitions per speaker-word pair. Validation has three speakers under the same eight-word, ten-repetition structure, giving 240. Test retains three speakers, eight words, and eight repetitions, giving 192. The 48 quarantined examples correspond to the remaining two repetitions per test speaker-word pair. These products reconcile the protocol without treating individual RGB frames as independent labeled utterances.

The unit of classification is one segmented utterance. A clip may contain many image files, but those frames do not increase the statistical sample count. Similarly, the three cached encoder layers do not create three independent training examples: they are alternative representations of the same clip. Keeping utterance identity intact prevents inflated denominators when comparing frame-based storage with clip-based metrics.

The diagnostic phrase selection uses two repetitions per validation phrase and two specified repetitions per test phrase, resulting in sixty examples for each split. Phrase labels remain outside the eight-class supervised vocabulary. Their only role in the reported policy is to probe rejection. Combining those phrases into the known-word test denominator would change the task and make the headline 182/192 result uninterpretable.

<!-- PAGE -->
# V. Spatial and Temporal Preprocessing

## A. Source decoding and landmark measurements

The preprocessing pipeline reads ordered RGB frames and verifies their source hashes. For MIRACL, each frame must have the expected 480×640 geometry. MediaPipe measurements supply facial landmarks; eye-axis orientation and mouth position determine a canonical crop. The procedure uses real measured landmarks, not landmark coordinates inferred from labels or predictions. Tracking metadata records source-frame count, tracked-frame count, and longest missing segment for each utterance.

A valid clip requires at least two tracked frames, at least 80% tracking coverage, and a gap statistic no greater than five frames. The measured eye axis compensates for in-plane head rotation. The mouth center and width determine a square crop with scale margin 2.2. This square is sampled at 96×96 pixels and converted to grayscale. The choice removes color variation from the encoder input while retaining a reproducible mouth-centered field of view.

## B. Temporal interpolation

Word utterances vary in the number of source frames. The MIRACL path maps the ordered tracked sequence to 29 samples using linear interpolation. For continuous index u between neighboring frames, the interpolation is:

@equation temporal

Here α is the fractional part of u. The operation normalizes sequence length; it does not establish the original recording frame rate. Consequently, the paper does not convert MIRACL source-frame counts into seconds or claim that all utterances were recorded at 25 FPS. A motion-duration experiment would need reliable native timestamps or a separately documented frame rate.

## C. Checkpoint-specific normalization

The 96×96 crop is center-cropped to 88×88 before AV-HuBERT input. Pixels are scaled from byte intensities to [0,1] and normalized using the checkpoint’s mean and standard deviation:

@equation normalization

The final tensor is [B,1,29,88,88], where B is the batch dimension and the singleton channel is grayscale. The encoder returns a temporal feature sequence with 768 coordinates per time step. The normalization constants are part of the artifact contract; substituting another crop or scaling convention can change predictions even if the classifier weights are unchanged.

| Stage | Geometry or rule |
|---|---|
| Source RGB | 640×480, three channels |
| Measured mouth crop | 96×96, margin 2.2 |
| Temporal mapping | 29 interpolated samples |
| Encoder crop | Center 88×88 |
| Intensity transform | [0,1], mean .421, std .165 |
| Encoder input | B×1×29×88×88 |
| Selected feature | 29×768 |

## D. Distinct preprocessing for other models

The preserved scratch model consumes 24 grayscale frames at 64×64 with values in [0,1]. Its paired comparison starts from the same original RGB utterances and applies that model’s own required preprocessing. It does not resize an AV-HuBERT feature cache and pretend the result is a scratch-model input. This preserves the meaning of a complete-system comparison while also introducing a preprocessing confound that must be acknowledged.

The publisher LRW route uses mean-face similarity alignment and real-time 25-Hz resampling rather than the MIRACL fixed-length convention. The alphabet adapter resamples to 30 frames at height 140 and width 46. These inputs are incompatible by design. A single shared mouth video endpoint cannot safely select an arbitrary checkpoint without applying its recorded spatial, temporal, and normalization contract.

Generic training utilities support coherent crop jitter, photometric changes, and frame masking. Their existence is not evidence that every completed experiment used them. The reported frozen MIRACL feature run is described by its actual preparation and extraction records, and no unrecorded augmentation gain is included in the results.

## E. Coordinate and transform specification

The shared Python canonicalization converts normalized landmark coordinates into pixels using the actual frame width and height. Landmarks 61 and 291 define mouth width; landmarks 13 and 14 define the midpoint of the inner vertical mouth opening. The vector from landmark 33 to landmark 263 defines the eye axis. If that axis is too small, the implementation falls back to the mouth-corner axis. Invalid coordinates or a mouth width below two pixels are rejected.

The rotation center is the measured mouth midpoint. The scale is the requested output size divided by mouth width times the 2.2 margin. Translation moves the center to the middle of the crop. OpenCV performs bilinear affine warping with replicated image borders. These details are part of the actual preprocessing, not an interchangeable “face crop” description. Changing the landmarks, border convention, or transform center changes the pixels seen by the encoder.

The representation loader also rejects nonfinite arrays and floating input outside [0,1]. Integer arrays are scaled by 255, while floating arrays are assumed already scaled. This distinction prevents a double-normalization error. It verifies temporal and channel dimensions before returning a contiguous tensor. The resulting safeguards establish input validity, although they cannot guarantee linguistic correctness or detect every visually poor crop.

<!-- PAGE -->
# VI. Frozen Representation Architecture

## A. Loaded representation model

The representation checkpoint is the official AV-HuBERT Base artifact `base_vox_iter5.pt`. Its local size is 1,240,185,329 bytes. Inspection of the loaded model reports 103,330,280 parameters, all with gradients disabled, and 9,620 buffer elements. This count includes modules retained in the checkpoint object, including the audio feature extractor and pretraining projection. It should not be interpreted as the number of parameters necessarily exercised by the selected-layer video-only call.

The visual path begins with a 3D convolutional frontend and residual visual encoder, then produces features for the transformer stack. The inspected frontend convolution has one input channel, 64 output channels, kernel (5,7,7), stride (1,2,2), and padding (2,3,3). The model’s representation dimension is 768. The local adapter uses the official downstream `extract_finetune` method with audio absent and masking disabled. The method name does not imply that weights are being fine-tuned: the call occurs in an inference-only frozen path.

@figure architecture | Fig. 2. Implemented eight-word transfer pipeline. The large representation model is fixed. Only the downstream eight-class logistic coefficients and biases are learned on the local training split.

## B. Parameter accounting

The inspected object attributes 11,580,672 parameters to the video extractor, 80,640 to the audio extractor, 1,180,416 to post-extraction projection, 89,775,488 to the transformer encoder, 3,072 to layer normalization, and 196,864 to the final projection. Their sum is 103,330,280. These component counts describe the loaded artifact, including modules that are not part of the final layer-9 classifier output. No arithmetic combination is presented as a newly trained visual backbone.

## C. Selected intermediate layer

Layers 6, 9, and 12 are collected during a shared extraction pass. Each retained clip provides three temporal feature matrices, each of shape 29×768. This enables head experiments without rerunning landmark detection or loading the encoder repeatedly. Cached features are tied to source, preprocessing, and encoder fingerprints. A cache is valid only for the exact representation contract that generated it.

The choice of layer 9 is empirical within the recorded validation search. It should not be interpreted as a universal property of AV-HuBERT or proof that deeper layers are inferior for all visual speech tasks. Different vocabularies, source distributions, pooling schemes, and training sizes can change the best layer. The paper therefore reports all tested candidates rather than only the winning score.

## D. Frozen computation and memory

The Windows implementation uses memory-mapped loading and meta-device construction to avoid allocating an unnecessary second large encoder. A compatibility patch skips random initialization for meta tensors before strict assignment of checkpoint weights. Ordinary loading remains available for other callers. This is an engineering memory optimization; it does not change the learned representation or constitute model compression, pruning, quantization, or knowledge distillation.

The distinction between loaded parameters, trainable parameters, checkpoint bytes, and runtime resident memory is retained throughout the paper. A file may contain training state or unused modules, while memory measurements depend on paging and process history. None of these quantities alone establishes the real-time speed of the browser pipeline.

| Loaded component | Parameter count |
|---|---:|
| Video extractor | 11,580,672 |
| Audio extractor retained in object | 80,640 |
| Post-extraction projection | 1,180,416 |
| Transformer encoder | 89,775,488 |
| Layer normalization | 3,072 |
| Final projection | 196,864 |
| Total inspected object | 103,330,280 |

## E. Intermediate-state integrity

Forward hooks collect the measured intermediate outputs of layers six and nine, while the final call supplies the layer-twelve representation. Each cache item must have shape [3,29,768] and finite values. Cache receipts combine source identity, ROI identity, encoder fingerprint, chosen layers, preprocessing, and landmarker fingerprint. A previously written cache is reused only when its receipt and content hash match. Unexpected existing files are protected rather than silently overwritten.

The extraction function can stop after a time or sample budget and retain completed cache items. This matters on limited hardware because interruption should not require re-encoding every source video. Resumability is a computation-management feature, not a change to the train/validation/test policy. A resumed run must retain the same source and preprocessing contract.

<!-- PAGE -->
# VII. Downstream Head Training and Selection

## A. Temporal pooling

The frozen encoder yields H with 29 rows and 768 feature coordinates. Three pooling functions are considered. Mean pooling produces 768 values. Three-bin temporal pooling divides the sequence into three contiguous parts, averages each part, and concatenates the resulting vectors to obtain 2,304 values. Mean-and-standard-deviation pooling concatenates the global temporal mean and standard deviation, producing 1,536 values:

@equation pooling

Mean pooling discards explicit ordering after the encoder has already modeled temporal context. Three-bin pooling retains coarse progression. Mean/std pooling describes central tendency and temporal variation. These interpretations explain the candidate choices; they are not independent measurements of what any feature coordinate encodes. The observed winner is mean/std at layer 9.

## B. Training-only scaling and classifier

A StandardScaler is fitted on training vectors for each layer/pooling pair and reused on validation vectors. It subtracts training means and divides by training scales. The eight-class multinomial logistic model then maps the standardized vector to logits and probabilities:

@equation head

W has shape 8×1,536 and b has eight elements. The selected classifier therefore has 12,296 learned coefficients and biases. Scaler means and scales are fitted statistics, stored separately from this learned-parameter count. They are essential to prediction compatibility even though they are not optimized through encoder backpropagation.

## C. Optimization and search space

The solver is L-BFGS with a maximum of 600 iterations, balanced class weights, and random-state value 42. Regularization C is tested at 0.01, 0.1, and 1.0. C controls inverse regularization strength; it is not a neural-network learning rate. No minibatch size, warm-up epoch count, or AdamW schedule should be attributed to this selected logistic-head experiment. Those settings belong to other repository trainers.

@equation objective

The equation expresses a class-weighted regularized negative log-likelihood; the exact numerical scaling follows the scikit-learn solver. The coefficient a for each target class reflects balanced weighting. This formulation updates W and b while the encoder remains fixed. Calling the completed run “end-to-end fine-tuning” would be inaccurate.

Selection is lexicographic: maximize validation accuracy with tracking failures included in the denominator, then minimize usable-example validation loss when accuracies tie. Three layers, three poolings, and three C values yield 27 candidates. The selected configuration is layer 9, mean/std, C=1.0, with 228/240 correct validation predictions and log loss 0.18884. Its recorded solver iteration count is 11.

## D. Artifact protection

The selected head stores vocabulary, preprocessing contract, encoder identity, scaler statistics, weight matrix, bias, and selection record. The head-search function refuses to overwrite an existing selection file. This discourages repeated tuning after test inspection, although it cannot replace sound experimental practice or guarantee that all earlier human decisions were blind to the data.

The retained experiment uses 704 usable training words. Its training duration of approximately 19.29 seconds covers the recorded head-search stage, not a neural-encoder fine-tuning run. There is no epoch-wise encoder loss curve because no encoder optimizer ran. The valid learning evidence is the head-search table and the baseline’s separately recorded training curve.

## E. Computational geometry of pooling alternatives

The three pooling functions yield different classifier sizes. An eight-way mean-pooled head has 8×768+8=6,152 coefficients and biases. The selected mean/std head has 8×1,536+8=12,296. A three-bin head has 8×2,304+8=18,440. These are calculated classifier dimensions; they exclude scaler statistics and all frozen representation parameters. Increasing head width is therefore another part of the candidate search, even though every candidate is linear in its own pooled representation.

Mean/std pooling uses the standard deviation across the temporal axis, as implemented by NumPy’s default population-standard-deviation calculation. It does not concatenate framewise logits or average independent word predictions. The transformation is computed before classification, and the stored scaler is applied to the concatenated vector. The saved artifact checks the weight, mean, and scale dimensions and requires positive scales when computing probabilities.

Balanced class weights account for the training labels remaining after measured exclusions. They do not rebalance the validation or test data or alter their denominators. The post-exclusion training support and original source identities remain inspectable in the feature manifest. This enables a future matched-data study to choose the same examples explicitly rather than approximating them through aggregate sample counts.

<!-- PAGE -->
# VIII. Scratch Baseline and Alphabet Comparator

## A. Preserved scratch whole-word model

The paired baseline is a separately trained WordVideoNet with 84,048 parameters. Its input has shape [B,1,24,64,64]. Three 3D-convolution blocks increase channels from 1 to 8, 16, and 32; each uses a 3×3×3 kernel, spatial stride two, padding one, GroupNorm, and SiLU. Temporal resolution remains 24 samples. Spatial averaging produces a 24×32 sequence, which enters a two-layer bidirectional GRU with hidden size 48 per direction and inter-layer dropout 0.15.

The recurrent output has 96 coordinates per time step. Temporal averaging, dropout, and a linear 96→8 layer produce one eight-way word distribution. The exact historical model definition is preserved in `legacy_word_model.py`, and the paired-comparison record stores its source hash. This matters because independently modified current model code is not necessarily the architecture that produced an older checkpoint.

@figure scratch_loss | Fig. 3. Actual recorded scratch-model training and validation losses over twelve epochs. Epoch 11 was selected by validation loss. This is not an AV-HuBERT fine-tuning curve.

## B. Historical training evidence

The scratch run used a selected set of 140 training examples, 48 validation examples, batch size two, seed 42, and one CPU thread. Its recorded maximum was twelve epochs, its selected epoch was eleven, and training took 58.44 seconds. The history records an initial learning rate of 0.001 and later reductions. The repository trainer implements cross-entropy, AdamW, validation-loss selection, and scheduling, but later source revisions are not proof that every present default applied to this historical run.

The paired test in this paper uses 192 new source utterances under the transfer experiment’s specified protocol rather than the older 48-example pilot test. This avoids comparing percentages computed on different test sets. However, the training sets remain unequal: 140 usable baseline examples versus 704 usable transfer examples. The resulting improvement combines representation, preprocessing, and supervision changes.

## C. Separate deployed alphabet model

The alphabet checkpoint is a 913,274-parameter 2D CNN–BiLSTM with 26 outputs. It accepts [B,30,1,140,46]. Its forward pass is unusual and must be documented literally: convolution 1→16, BatchNorm/ReLU, adaptive pooling to 4×4; convolution 16→32 and max-pooling to 2×2; convolution 32→64 and max-pooling to 1×1; a second adaptive pooling call to 4×4; then flattening to 1,024 features per frame. The final adaptive operation does not recover spatial detail already discarded by earlier pooling.

A one-layer bidirectional LSTM with hidden size 96 produces 192 features. The final time step passes through a 192→128 layer, ReLU, dropout 0.5, and a 128→26 classifier. The checkpoint emits a single class vector, not CTC logits across time. Its recorded test score is 12/100, with best validation accuracy 20%. It is retained as a distinct alphabet capability and is not the 12.50% eight-word baseline used in the principal comparison.

| Scratch stage | Tensor geometry after stage |
|---|---|
| Input | B×1×24×64×64 |
| Conv3D block 1 | B×8×24×32×32 |
| Conv3D block 2 | B×16×24×16×16 |
| Conv3D block 3 | B×32×24×8×8 |
| Spatial mean | B×24×32 |
| Bidirectional GRU | B×24×96 |
| Temporal mean and linear head | B×8 |

## D. What the baseline loss curve establishes

The recorded validation accuracy remains low while loss changes only slightly. The curve supports a description of poor learned discrimination on that historical run. It does not, by itself, prove vanishing gradients, illumination dominance, class imbalance, or a particular normalization failure. Those mechanisms require targeted measurements. Earlier narrative documents contain stronger causal claims, but the paired comparison here relies only on the preserved architecture, training history, and actual predictions.

The baseline remains scientifically useful because it reflects the existing small-model system that the transfer pipeline replaced. A stronger scratch baseline with matched supervision, more hyperparameter search, and repeated seeds might perform differently. Reporting this possibility does not erase the observed result; it defines the comparison’s scope and motivates an appropriate follow-up rather than claiming that scratch learning is universally inadequate.

<!-- PAGE -->
# IX. Mathematical Task Definitions and Sequence Boundary

## A. Closed-vocabulary classification

For an isolated clip X and vocabulary of K labels, the classifier produces logits z and probabilities p. The prediction is the largest-probability class. In the main experiment K=8; in the alphabet mode K=26; in the publisher LRW mode K=500. Identical softmax code does not make these tasks equivalent because their data distributions and label inventories differ.

@equation softmax

Top-1 accuracy compares one predicted class with one reference class for each utterance. Top-5 accuracy checks whether the reference appears among the five highest probabilities. With only eight classes, top-5 is a relatively permissive diagnostic and should not be confused with a five-word transcript. The main paper reports it alongside top-1 rather than as the primary success criterion.

@equation accuracy

## B. Per-class metrics

For each class k, true positives, false positives, and false negatives determine precision, recall, and F1. Macro averages weight every class equally; weighted averages weight by support. The test set contains 24 examples per class, so macro and weighted F1 coincide up to floating-point precision. This does not imply that precision equals recall for each class.

@equation metrics

The confusion matrix uses reference classes as rows and predicted classes as columns. Its diagonal contains correct counts. The off-diagonal row entries describe where a reference word was misclassified, while column entries explain reduced precision of the receiving class. Tables and plots use the same vocabulary order as the head artifact, avoiding accidental permutation of labels.

## C. Sequence CTC path

For a genuine character-sequence task, output length is not fixed to one class. The repository’s experimental CTC vocabulary places blank at zero and A–Z at indices 1–26. CTC sums the probabilities of all frame-level paths that collapse to a target sequence:

@equation ctc

The implemented loss accepts logits [B,T,V], converts them to float32 log probabilities [T,B,V], supplies explicit sequence lengths, and invokes PyTorch CTC with blank zero and zero-infinity handling. Greedy decoding takes the framewise argmax, merges adjacent repeats, and removes blank. A blank between repeated letters preserves both letters. These are implemented utilities with targeted tests, not a measured trained recognizer.

The configured sequence encoder uses fused visual features of dimension 192, four Conformer blocks, four attention heads, feed-forward width 768, and convolution kernel seven. The letter head produces 27 logits at each time step. No matching trained active checkpoint with held-out recognition evidence was found. Beam search and language-model fusion are also absent from the verified path.

## D. WER and CER interpretation

Word error rate is the sum of substitutions, deletions, and insertions divided by reference word count. Character error rate uses the same edit-distance definition over characters:

@equation wer

For the eight-word experiment every sample has one target word and the raw classifier returns one word, so the reported 5.21% error is a substitution rate. It is not a sentence WER result. The availability of edit-distance functions or GRID sentence annotations cannot justify reporting continuous-recognition metrics without actual model-generated sequences and references.

## E. Temporal classification versus alignment learning

Both a BiGRU clip classifier and a CTC decoder can process a temporal input, but their supervision differs. The clip classifier is trained to emit one class after temporal pooling. It does not learn word boundaries or output ordering among several words. CTC instead receives a sequence target and sums over possible alignments, which makes its blank and repeat behavior part of the learning objective. Replacing a final linear layer without the corresponding sequence labels and training objective cannot bridge that distinction.

The current Conformer configuration also permits optional phone and viseme heads. They are disabled by default, and no measured auxiliary-supervision result is present. Spoken letter names should not be assigned phoneme clusters merely from their written first letters. Any future viseme experiment needs explicit annotation provenance and a definition appropriate to the spoken units being recognized.

<!-- PAGE -->
# X. Paired Whole-Word Recognition Results

## A. Reconstruction from saved predictions

The manuscript builder reads both prediction JSONL files, verifies 192 unique matching identifiers, checks matching reference labels, and recomputes the number of correct predictions. It also reconstructs the transfer confusion matrix and verifies equality with the saved evaluation matrix. The head and preserved scratch-checkpoint hashes match their recorded experiment identities. This reconstruction is an arithmetic validation of saved evidence; it does not claim that a new inference run or training experiment was performed for manuscript production.

@table scores

The transfer model classifies 182 clips correctly and misses ten. The scratch model classifies 24 correctly and misses 168. The observed difference is 158/192, or 82.29 percentage points. Both systems use the same raw source utterances, but their own recorded preprocessing and learned artifacts. This makes the comparison useful for an application-level replacement decision while leaving causal attribution to individual changes unresolved.

@figure paired_accuracy | Fig. 4. Paired test-set top-1 accuracy. Error bars are descriptive 95% Wilson intervals treating clips as Bernoulli observations. Repeated utterances within only three speakers violate a population-independent interpretation of these bars.

## B. Paired outcome structure

All 24 clips correctly recognized by the baseline are also correctly recognized by the transfer model. The transfer model additionally corrects 158 baseline errors; ten clips remain wrong for both systems. No clip is correct only for the baseline. The paired contingency table is therefore more informative than comparing two independent percentages.

@table paired

A descriptive exact paired test under independent discordant observations would yield 2×0.5^158. This arithmetic is stored in the evidence-validation receipt, but we do not use the resulting very small number to claim broad population significance. The 192 observations are nested within three speakers and repeated words, and the test speaker identities were seen in a pilot. A larger independently sampled cohort is needed for robust population-level inference.

## C. Validity of the improvement claim

The headline result is an observed difference between two complete pipelines. It is not an 82.29-point gain from encoder fine-tuning, because no encoder weights were updated. It is not an isolated gain from layer 9, because training data, model class, and preprocessing also differ. It is not a prediction of physical-webcam accuracy. The result remains substantial within the explicitly defined experiment without requiring any of those stronger interpretations.

The saved validation score is 228/240, or 95.00%, while the held-out test score is 94.79%. Their closeness is descriptive. It does not prove absence of overfitting or eliminate selection bias from evaluating 27 candidates on the same validation set. The complete search results are reported later to show the alternatives actually examined.

## D. Top-five and failure accounting

The true label appears among the five highest probabilities for 191 of 192 test clips. This means that nine of the ten top-one errors still retain the reference among the five candidates. It does not mean that a postprocessor can reliably choose the correct alternative without additional evidence. Selecting the right candidate using the test reference would be an oracle operation, not a valid recognition method.

The raw model always returns a known-vocabulary candidate for a usable clip. Abstention is introduced only by the later policy. Consequently, the top-one denominator is 192 regardless of whether a candidate is displayed. The test evaluator is designed to count tracking failures as errors, but the retained test happens to have none. Future reports should preserve this all-planned-samples convention even when a different capture setup yields more failures.

The paired table also shows that the comparison is not driven by a small number of improved examples: 158 additional clips are correct. Nevertheless, those clips are repeated utterances of only eight words from three people. The appropriate empirical conclusion remains a strong observed improvement on this defined task, with broader generalization left to a new experiment.

<!-- PAGE -->
# XI. Complete Confusion Matrices and Error Analysis

## A. Preserved baseline distribution

The baseline matrix is reconstructed from the same 192 paired prediction rows used in Section X. Every reference row has support 24. This uniform support makes a classifier that always predicts one class achieve exactly 12.50% accuracy. The matrix, rather than the percentage alone, reveals the actual concentration of the saved baseline predictions. The plot does not diagnose the optimization cause of that behavior; such a diagnosis would require controlled interventions or intermediate-feature measurements.

@figure confusion_baseline | Fig. 5. Complete baseline confusion matrix on the paired 192-utterance set. Rows are reference words; columns are predicted words. Every cell is an observed count, including zeros.

## B. Matrix reconstruction and normalization

The builder increments each cell from the saved reference/candidate pair. Matrix sums must equal 192, row sums must equal 24, and the transfer trace must equal 182. These checks detect omitted examples and reordered label maps. Row-normalized values would divide every cell by 24; integer counts retain the exact observations. Future unequal-support tests should report both counts and normalized values, alongside sample IDs for error inspection.

## C. Transfer distribution

The transfer matrix has a diagonal sum of 182. Choose, hello, start, and stop each have 24 correct predictions. Connection and navigation each have one error. Next has two errors and previous has six. These row counts expose a residual class imbalance in performance despite identical test support. The model is highly accurate on this inventory but not uniformly reliable across words.

@figure confusion_transfer | Fig. 6. Complete frozen-AV-HuBERT head confusion matrix. The strongest residual confusion is previous→start, with four observations. The image is generated from saved per-clip predictions, not from rounded percentages.

## D. Interpreting individual errors

Connection is misclassified once as choose. Navigation is misclassified once as connection. Next is assigned once to hello and once to navigation. Previous is assigned once to choose, once to navigation, and four times to start. These ten events account for all observed transfer errors. No test tracking failures need to be added to this matrix in the current run.

The matrix cannot establish that two words share a particular phonetic articulation or that a specific frame caused an error. Such interpretations require inspecting source videos, linguistic timing, and model responses under controlled perturbation. We therefore report the empirical confusion and identify previous as a priority for future data collection, without claiming a proven causal visual similarity.

Column analysis explains why start has perfect recall but reduced precision. All 24 actual start clips are correct, while four previous clips also enter the start column. Its precision is therefore 24/28, or 0.8571. Conversely, previous has precision 1.0 but recall 18/24: the model rarely predicts previous incorrectly, yet misses six true examples. A single accuracy value conceals this distinction.

<!-- PAGE -->
# XII. Per-Word and Per-Speaker Performance

## A. Class-level statistics

The following table reports precision, recall, and F1 from the complete transfer matrix. The average macro precision is 0.95232, macro recall is 0.94792, and macro F1 is 0.94668. All counts are based on raw top-one candidates before confidence rejection. Selective prediction has a different denominator and is examined separately in Section XIV.

@table perclass

@figure per_word | Fig. 7. Before/after recall for each of the eight words on identical source utterances. Equal test support makes the bars directly interpretable as correct counts divided by 24; it does not make the training procedures equivalent.

The weakest recall is previous at 75%. The strongest F1 is stop at 1.0 for 24 observations. A perfect observed class score does not imply that the word is always recognized. Twenty-four clips offer limited coverage of accent, head pose, lighting, camera optics, and speaking style. Future reports should retain class support alongside percentages and include failure cases instead of displaying only favorable examples.

## B. Speaker-level statistics

The three test-speaker scores are F10: 62/64 (96.875%), F11: 57/64 (89.0625%), and M08: 63/64 (98.4375%). The overall 182/192 score is their equally weighted average because each speaker contributes 64 utterances. F11 accounts for seven of the ten errors, suggesting that aggregate accuracy masks variation across people. The study does not infer a demographic explanation for that variation.

@figure speakers | Fig. 8. Recorded test accuracy by speaker. Only three speakers contribute to the final evaluation; the bars are individual observed outcomes rather than estimates of demographic-group performance.

## C. Limits of subgroup analysis

Three speakers are insufficient to estimate robust between-speaker variance across a broad population. Their identifiers encode corpus organization, not a validated demographic sampling design. Any future fairness or accessibility claim would require an appropriate recruitment protocol, consent, and enough independent participants to characterize the intended groups.

The class and speaker decompositions guide future collection without changing the frozen result. A new training expansion should emphasize naturally recorded previous utterances and difficult capture conditions, but its evaluation must use a fresh locked set. Reusing these errors repeatedly for hyperparameter selection would turn the current test into development data. The existing head and result artifacts should remain preserved as the historical comparison point.

## D. Complementary interpretation of precision and recall

Choose has perfect recall but precision 24/26 because it receives one error from connection and one from previous. Hello has precision 24/25 after receiving one next clip. Navigation has precision 23/25 after receiving errors from next and previous, while losing one of its own clips to connection. These relationships explain why the macro precision is not identical to macro recall even in a perfectly balanced test set.

The balanced support also clarifies the aggregate score: an error on any class changes overall accuracy by the same 1/192 increment. Per-class recall changes in larger steps of 1/24, approximately 4.17 percentage points. A difference of one correct example can therefore noticeably alter a class bar. Reporting three decimal places in the table aids arithmetic reproducibility; it should not be read as three-decimal statistical certainty about future users.

<!-- PAGE -->
# XIII. Validation Search and Model Selection Evidence

## A. All evaluated candidates

The search space contains layers 6, 9, and 12; pooling by mean, three temporal bins, or mean/std; and C values 0.01, 0.1, and 1.0. The table lists all 27 recorded candidates. “Valid. %” uses the 240 planned validation words. The loss column is the mean negative log probability of the true label on usable validation examples. All validation words were usable in this run, so the two denominators coincide here.

@table candidates

## B. Reading the search surface

The selected layer-9 mean/std configuration at C=1 has the highest validation accuracy, 95.00%, and loss 0.18884. The table also contains nearby candidates with slightly lower accuracy. A small numerical advantage on one validation partition should not be interpreted as a universal ranking of feature layers. It identifies the choice made under this protocol and records how many alternatives were considered.

@figure validation_search | Fig. 9. Validation accuracy for every layer/pooling/C combination. Each cell derives from `selection.json`; no test outcome appears in this selection surface.

## C. Selection discipline

The code compares accuracy first and negative loss second. Thus loss acts as a tie-breaker, not as the sole selection criterion. This detail differs from the scratch trainer, which selects by validation loss. Reproductions must preserve the selection rule, not merely the candidate list, because different tie-breaking can choose a different artifact.

The search fits each scaler only on training vectors. Validation means and scales are never used to standardize the training representation. The final test is transformed by the selected stored scaler. This prevents an otherwise easy form of leakage in frozen-feature experiments: estimating normalization statistics across all partitions before classifier fitting.

## D. No fabricated training curve

The frozen encoder has no downstream epoch history because its weights never change. The logistic solver’s convergence count and candidate losses are the relevant optimization records. Figure 9 is a model-selection surface, not a learning curve. Figure 3 is the actual scratch-model epoch curve. Keeping these figure types distinct avoids implying that the pretrained model underwent a full supervised neural fine-tuning schedule.

The saved head is protected after selection, and the record states that test results were not used. These checks are necessary but should be combined with a written protocol and an independently held test set in future experiments. The existing search record supports this experiment’s provenance without resolving the broader limitations of pilot exposure and small speaker count.

<!-- PAGE -->
# XIV. Confidence Rejection and Open-Set Limitations

## A. Evaluated acceptance rule

The saved evaluation applies two conditions to the raw eight-way distribution: the highest probability must exceed a confidence threshold and its gap to the second-highest probability must exceed a margin. The evaluated policy uses confidence threshold 0.95 and margin 0.20:

@equation reject

The thresholds were selected using validation words and a small validation-phrase diagnostic. The test is used only to report the resulting coverage and error behavior. The distribution remains a closed-set softmax; thresholding does not make it a fully validated detector of arbitrary unseen English words.

## B. Coverage and accepted accuracy

The evaluated policy accepts 137 of 192 known test words, giving 71.35% coverage. All 137 accepted words are correct. The remaining 55 clips are rejected, comprising 45 otherwise-correct raw classifications and all ten raw errors. Therefore, the accepted accuracy of 100% trades away a substantial fraction of known-word opportunities. It is not a claim of perfect recognition across all inputs.

@figure rejection_counts | Fig. 10. Known-word outcomes under the evaluated 0.95-confidence/0.20-margin policy. Rejection removes all ten errors but also withholds 45 correct raw predictions. These counts do not describe the current relaxed runtime policy.

## C. Phrase diagnostic

Eight of sixty held-out phrases are falsely accepted as known words, a 13.33% false-accept rate in that limited diagnostic. Phrases are not the same distribution as unseen isolated words. Their duration, transitions, and word content can differ substantially. The result is useful as a stress test but cannot establish broad open-set reliability. In particular, a phrase containing one of the target words may activate a known-word representation without constituting correct phrase recognition.

An adequate future evaluation would include unknown isolated words, silence, non-speech facial motion, incomplete words, and known words under varied capture conditions. Threshold selection should use validation examples from these conditions, while final performance should be measured on a separate locked set. Coverage, accepted error, false acceptance, and false rejection should all be reported.

## D. Current runtime-policy discrepancy

The inspected current rejection-policy file contains confidence threshold 0.60 and margin 0.10. Its prose describes live-webcam calibration, but no corresponding physical-webcam calibration experiment was found. The historical evaluation metrics above remain attached to 0.95/0.20. They cannot be carried over to the newer settings merely because the head checkpoint is unchanged.

The current service also returns the top candidate in word/display fields when the model marks the output uncertain. Combined with the observed frontend consensus regression, that behavior can expose a candidate before the intended confirmation checks. These implementation findings do not alter the saved offline accuracy, but they prevent treating that offline score as proof of current UI reliability.

## E. Probability interpretation

Softmax confidence is a model score, not a calibrated probability of correctness. No calibration error, reliability diagram, or isolated-OOV benchmark is reported for the current runtime configuration. A future temperature-scaling or other calibration step must fit only on validation data and be recorded as a new artifact. Any post-test threshold analysis should be labeled diagnostic and should not redefine the original locked headline result.

## F. Distinct denominators

Accepted accuracy is the number of correct accepted known words divided by accepted known words: 137/137. Coverage is accepted known words divided by all planned known words: 137/192. Overall accepted-correct coverage is also 137/192 in this run because no accepted known word is wrong. Raw accuracy remains 182/192. Phrase false acceptance is 8/60, with its own reference population. Combining these denominators into a single “reliability percentage” would obscure the trade-off.

Observed zero accepted errors do not establish zero future risk. The selected policy was tested on only 137 accepted known examples, drawn from the same limited vocabulary and speakers as the larger test. In addition, unknown phrases still generate false accepts. An interface should communicate uncertainty and allow a rejected result rather than automatically substituting the most likely class. Any implementation change that forces a display candidate changes the practical meaning of rejection, even if the numerical threshold is unchanged.

<!-- PAGE -->
# XV. CPU Execution and Resource Measurements

## A. Hardware and environment

The audited machine has an Intel Core i3-1115G4 CPU with two physical cores and four logical processors, approximately 7.74 GiB RAM, and no available CUDA device. The completed word experiment uses a Python 3.11.9/PyTorch 2.2.2+cpu environment. Its package inventory records NumPy 1.26.4, scikit-learn 1.5.2, and MediaPipe 0.10.14. Separate backend and audit environments contain other dependency versions; they should not be collapsed into a single invented environment lock.

The preparation/extraction runs use one CPU thread according to the recorded experiment. The local feature cache stores three 29×768 arrays per usable clip. This permits repeated small classifier fits after a one-time visual-encoding cost. The approach reduces the cost of downstream experimentation, but it does not remove the encoder cost at inference or demonstrate real-time webcam operation.

@figure runtime | Fig. 11. Recorded wall times for MIRACL ROI preparation, frozen feature extraction, and 27-candidate head search. The stages have different workloads and are not interchangeable per-frame latency measures.

## B. Recorded workload

ROI preparation takes 626.28 seconds across the planned 1,272 clips. Feature extraction takes 856.29 seconds and produces 1,256 usable cached items after the sixteen tracking exclusions. Head search takes 19.29 seconds. The paired baseline comparison takes 52.03 seconds. These are local wall-clock observations from preserved records, not averages across repeated hardware trials or guarantees for another computer.

The extraction total divided by usable clips is a descriptive aggregate cost, not an end-to-end capture-to-display latency. It excludes real user utterance duration, browser scheduling, transport, and UI confirmation. Likewise, classifier-fitting time should not be compared directly with the baseline’s total neural training time as if they represent equal computation: AV-HuBERT’s original pretraining cost is external to the local experiment.

## C. Separate LRW endpoint timing

The saved three-request LRW trace records round-trip times approximately 4,702.52, 1,026.07, and 918.37 ms. Corresponding server wall times are 4,479.10, 970.40, and 898.12 ms. All three requests concern the same public clip. The first request includes different initialization conditions, and the trace is too small to estimate a reliable latency percentile or sustained-load capacity.

@figure latency | Fig. 12. Three recorded publisher-model requests, with round-trip and server times shown separately. These are execution diagnostics for a specific clip, not a physical-webcam accuracy or throughput benchmark.

## D. Memory and timing interpretation

Other single-request traces report resident and peak process memory, but memory varies with lazy loading, paging, model coexistence, and process history. The official encoder checkpoint is approximately 1.24 GB on disk, while the selected word head is only tens of kilobytes. File size is not equivalent to resident working set. The manuscript retains this distinction instead of claiming that small head size makes the whole system lightweight.

Future measurements should separate cold load, warm preprocessing, encoder execution, classification, network transport, and display confirmation. They should repeat each condition and report distributions together with frame counts, image sizes, thread settings, and background load. LLM and TTS latency remain undefined because those components are absent from the verified implementation.

| Quantity | Recorded value | Scope |
|---|---|---|
| Physical/logical CPU cores | 2 / 4 | Audited machine |
| Installed memory | ~7.74 GiB | System snapshot |
| Available CUDA | No | Runtime probe |
| Usable feature items | 1,256 | Main extraction |
| Candidate heads | 27 | Validation search |
| Measured LRW requests | 3 | Separate timing trace |

The three-request trace reports a median round trip of 1,026.07 ms, but three observations do not establish a dependable service-level target. The measured sequential request rate includes the slow first request. It should not be extrapolated to concurrent camera streams, since the word CPU lock deliberately serializes work. A sustained-load experiment would need to count rejections and queue behavior as well as completed model calls.

<!-- PAGE -->
# XVI. Browser, Backend, and Live Inference Design

## A. Browser acquisition

The application uses React, TypeScript, and Vite. The browser requests a video-only camera stream after a user action, with ideal resolution 960×540, ideal frame rate 15 FPS, and maximum 24 FPS. MediaPipe Face Landmarker runs in VIDEO mode on CPU and allows detection of up to two faces. User-facing capture must distinguish a single usable speaker from no face or multiple faces. Browser permission and media-track lifecycle are separate concerns from model accuracy.

The live pipeline measures mouth aperture and recent movement, aligns a crop, and accumulates frames. Alphabet and eight-word modes use a maximum buffer of 29 frames, a minimum of eight, and inference considered every four captured frames. Visual activity gating uses measured movement rather than interpreting a static open mouth as speech. These heuristics help define candidate windows but have not been independently calibrated as a voice-activity detector on a labeled webcam corpus.

@figure runtime_flow | Fig. 13. Browser-to-backend flow. This is the implemented research pathway; the final confirmation behavior has a documented regression. Audio capture, LLM correction, and TTS are absent from this flow.

## B. Mode-specific inference

FastAPI exposes separate routes for alphabet, eight-word, and 500-word inference. The word backend is selected independently of the alphabet backend. This permits the measured eight-word model to remain available alongside the publisher 500-word path. Readiness endpoints report model loading and vocabulary. A nonblocking shared CPU lock prevents concurrent word requests from silently oversubscribing the limited CPU; the service returns HTTP 429 when busy.

| Route | Intended task |
|---|---|
| POST /api/predict | Isolated A–Z label |
| POST /api/predict/word | Selected eight-word model |
| POST /api/predict/word/lrw500 | Publisher 500-class model |
| GET /api/word/health | Eight-word readiness |
| GET /api/word/lrw500/health | 500-word readiness |

## C. Stabilization and current defect

The probability stabilizer supports an exponential moving average with α=0.6 and two consistent top-one windows. The separate word-consensus implementation currently overwrites candidate and utterance state before testing whether they changed, then confirms a non-null candidate immediately. Four targeted frontend tests fail for this behavior. Consequently, the paper does not claim that the current word display reliably enforces two accepted agreeing windows.

@equation ema

The defect is an implementation issue, not evidence that the offline representation is inaccurate. Conversely, high offline accuracy does not excuse the defect: premature publication can change what a user sees, especially for uncertain inputs. This distinction motivates separate tests for recognition, rejection, state transitions, and UI lifecycle. The manuscript-generation task records the issue without changing application code or restarting a server.

## D. Validation evidence boundary

A recorded real API smoke accepts a genuine connection clip at confidence 0.99838. This establishes one successful route execution with real visual input. Browser checks verify rendering, readiness, mode selection, and camera-off behavior. They do not constitute a physical-webcam recognition benchmark. The word “live” refers to the capture interface, not a measured population accuracy claim.

## E. Activity and probability-state details

The activity detector keeps at most 29 aperture measurements. It evaluates the aperture range and the root-mean-square difference over the six most recent samples. The recorded thresholds are range 0.045, recent energy 0.008, at least eight samples, and a 350-ms movement hold. A time gap above 500 ms or a backward timestamp resets the activity state. These constants are heuristics, and no held-out detector sensitivity or specificity is reported.

The probability stabilizer only applies EMA when it receives a complete valid distribution: finite values within [0,1], unique labels, and a sum within 0.001 of one. If the vocabulary changes, the stored averages and repetition state reset. If a full distribution is unavailable, the implementation falls back to top-one consensus. These checks are useful software contracts, but their success cannot compensate for the separate word-publication defect described above.

<!-- PAGE -->
# XVII. Additional Recognition Modes and GRID Integration

## A. GRID acquisition and validation

The saved GRID acquisition includes 66 video/alignment archives for 33 speakers. The publisher does not supply video for s21. The local inventory records 33,000 source videos, 32,891 usable video/annotation pairs, and 109 exclusions for premature decoder termination. Train, validation, and test partitions use separate speakers: 23 speakers with 22,891 clips, five speakers with 5,000 clips, and five speakers with 5,000 clips respectively. Source and decoded-thumbnail fingerprints are recorded for duplicate screening.

GRID annotations contain six content positions: command, color, preposition, letter, digit, and adverb. The letter-slot task selects the fourth token after removing silence/pause annotations. Alignment units are 25,000 per second. The observed labels are A–V and X, Y, Z, omitting W. Each validation/test letter class has 200 examples. A 25-class head learned from this manifest must not be passed to the browser’s exact A–Z adapter.

The acquired corpus enables future experiments; it does not create a trained continuous decoder. A separate small command-word experiment covers bin, lay, place, and set, with 12/32 test examples correct in its recorded improved run. Its 37.50% score is neither a sentence score nor part of the eight-word MIRACL result. The tasks have different vocabularies, speakers, and segmentation assumptions.

## B. Publisher 500-class architecture

The genuine publisher word model combines a 3D visual frontend, ResNet18 trunk, and Dense TCN. The configuration has four blocks of three layers, growth 384, temporal kernels 3/5/7, dilations 1/2/5, reduced size 512, squeeze-excitation enabled, dropout 0.2, and no word-boundary input. Inspection reports 52,548,532 parameters and a final classifier matrix of shape 500×1,664. The ordered author vocabulary matches 500 unique local entries.

Strict loading and a correct public AFTERNOON prediction establish execution compatibility. The original smoke reports confidence 0.95890. A broader tiny local diagnostic contains only four known examples across two classes, with one correct and one failed case; it is inadequate for estimating 500-class performance. Neither result inherits the 94.79% MIRACL score.

## C. Alignment and acquisition limitations

The LRW adapter uses a stored mean face, robust similarity fitting, a 256×256 aligned image, a 96×96 mouth region, and an 88×88 center crop. Browser MediaPipe points approximate the publisher’s landmark convention. The live preprocessing also uses timestamps to resample at 25 Hz and enforces duration, frame-count, face-count, lighting, and motion constraints. These checks improve input validity but do not prove publisher-equivalent distribution or calibrated rejection.

| Mode | Labels | Local evidence | Unavailable evidence |
|---|---:|---|---|
| Alphabet CNN–BiLSTM | 26 | Checkpoint; 12/100 historical test | Strong webcam generalization |
| MIRACL transfer | 8 | 182/192 specified test | General-English accuracy |
| Publisher DC-TCN | 500 | Strict load; real smoke | Full authorized LRW test |
| GRID letter slot | 25 | Validated corpus/manifests | Completed corresponding head |
| Conformer/CTC | 27 incl. blank | Implemented loss/decoder tests | Trained active sequence model |

## D. Preserving capability boundaries

A backend registry should identify task, vocabulary, checkpoint, preprocessing, and evaluation scope together. A 500-dimensional output cannot be interpreted using an eight-word map, and a 25-class GRID label set cannot be silently expanded to 26. The repository’s task separation is therefore scientifically useful as well as an engineering safeguard. The present study retains all modes while attaching quantitative claims only to their own artifacts.

## E. GRID split implementation

The training speakers are s1–s20 and s22–s24. Validation uses s25–s29; test uses s30–s34. All 33 available speakers are represented. The annotation manifest stores integer start and exclusive end frame bounds for the selected letter interval while pointing to the original full-face video. It does not require duplicating or re-encoding the corpus into a second collection of cropped video files. This preserves source identity and makes the segment definition auditable.

The preparation report records 108 exclusions for s8 and one for s4. It also retains fully decoded streams whose container frame-count estimate was slightly low. Excluded videos remain on disk with their reasons in the inventory; they are not replaced by synthetic data. The declared speaker-independent split applies to future runs that actually use these assignments. It does not retroactively establish speaker independence for an older checkpoint whose training speakers were undocumented.

For LRW, the official class list and the checkpoint’s output dimension are necessary compatibility checks but insufficient evaluation evidence. A complete local study still needs the authorized videos and metadata, valid source splits, consistent word-centered preprocessing, and predictions over the held-out benchmark. The correct public example is retained as a small execution check while that broader work remains unavailable.

<!-- PAGE -->
# XVIII. Threats to Validity and Future Research

## A. Internal validity

The principal baseline comparison changes more than one factor. The baseline was trained on 140 usable examples, while the transfer model uses 704. Their image sizes, temporal lengths, normalization, model capacities, and optimization methods differ. The 82.29-point improvement is an observed complete-pipeline result. To isolate the value of pretrained representation, a future study should train both approaches on identical source training IDs, preserve matched validation/test partitions, and report preprocessing ablations separately.

The test speaker identities were observed during a prior pilot even though repetitions 01–02 were quarantined. This means that exact test utterances are fresh relative to that pilot but the population is not untouched. Informal development decisions may have benefited from earlier speaker exposure. The limitation cannot be repaired retrospectively by renaming the split. A new cohort, recruited and locked before tuning, is the appropriate next evaluation.

The validation search contains 27 candidates. Although test results were not used in the head-search loop, repeated comparisons can adapt the selected model to validation idiosyncrasies. The recorded search surface makes this multiplicity visible. Nested validation or an additional development cohort would provide a stronger estimate of the selection procedure’s robustness, especially when speaker count is small.

## B. External validity

Eight controlled word labels do not represent arbitrary English. The classifier cannot emit a word outside its vocabulary, distinguish all homophenic alternatives, or transcribe an unconstrained sentence. Its softmax distribution always allocates probability among known classes. Open-set evaluation must therefore include unknown isolated words and non-speech activity, rather than relying solely on phrase rejection.

The MIRACL RGB frames differ from browser webcam streams in capture conditions and timing. Native MIRACL FPS is not recorded in the manifest, and interpolation to 29 frames normalizes ordering rather than physical duration. A future study should collect reliable timestamps, document camera resolution and exposure, and evaluate pose, distance, lighting, accent, speaking rate, and occlusion. Physical webcam accuracy remains unmeasured in the present artifacts.

Only three speakers contribute to the final 192 clips. Clip-level confidence intervals consequently understate population uncertainty if interpreted as independent observations. Repeated words and utterances within a person create dependence. Future analysis should preserve speaker as the sampling unit, include more speakers, and report both pooled and per-speaker metrics. The current descriptive intervals are explicitly qualified for this reason.

## C. Implementation validity

The current consensus regression and relaxed rejection-policy file demonstrate that deployed behavior can drift after an offline evaluation. Model hashes alone are insufficient to identify the complete runtime. A deployable release should bind frontend state logic, service response semantics, preprocessing, thresholds, model weights, and evaluation evidence to a versioned manifest. Otherwise an unchanged head may produce different user-visible behavior.

The source audit also found stale documents and a synthetic LRW fixture directory. These artifacts are useful for tracing project development but must not become scientific evidence merely because they contain plausible labels or training terminology. The manuscript uses primary saved predictions and explicit provenance to avoid that error.

## D. Context correction and speech generation

No executable LLM provider, model identifier, prompt template, token settings, correction endpoint, or correction evaluation was found. No TTS engine, voice, waveform output, sample rate, or synthesis latency was found. These components remain proposed. A future LLM module should receive the raw candidate distribution, preserve uncertainty, allow abstention, and log both original and corrected text. Its benefit must be tested against harmful overcorrection on a locked set.

A future TTS stage should consume confirmed text and expose explicit playback control. Evaluation would need to separate recognition mistakes from synthesis intelligibility, report the voice and language, and measure capture-to-text and text-to-audio latency independently. Neither fluent corrected text nor audible output would establish improved visual recognition by itself.

## E. Research next steps

The highest-priority implementation task is repairing confirmation and uncertainty handling, followed by a versioned validation of the chosen thresholds. The highest-priority scientific task is a matched-data experiment on untouched speakers. Full authorized LRW evaluation, genuine character-sequence training, calibration analysis, and carefully constrained LLM/TTS extensions can then be investigated as separate experiments with their own denominators and evidence.

| Proposed experiment | Controlled comparison | Required outcome |
|---|---|---|
| Matched supervision | Same source training IDs | Top-1 and class metrics |
| Alignment ablation | Canonical vs. fixed crop | Errors plus tracking failures |
| Pooling ablation | Fixed layer and split | Validation/test separation |
| New speaker cohort | Locked model and policy | Speaker-level uncertainty |
| OOV evaluation | Unknown isolated words | False accept/reject rates |
| Live state repair | Same recorded responses | Confirmation correctness |

These rows are a future experimental design, not completed measurements. They identify what would be needed to move from the present integration result toward stronger causal and deployment claims. The current paper deliberately does not populate the proposed rows with estimated accuracy gains.

<!-- PAGE -->
# XIX. Reproducibility and Artifact Appendix

## A. Primary experiment records

The main experiment directory is `runs/word_accuracy/avhubert_miracl8_v1`. Its protocol file records source manifest identity, vocabulary, preprocessing, encoder identity, source-list hash, partition sizes, and quarantined identifiers. `selection.json` contains the selected candidate and all 27 alternatives. `evaluation.json` contains aggregate, per-word, and per-speaker results. Two prediction JSONL files provide the preserved baseline and transfer predictions for the same 192 utterances.

The head checkpoint SHA-256 is bc07a9eca0979e556af573a387733396a891fe41faa10715c9f4d323e9c4cc74. The frozen encoder fingerprint is dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc. The preserved scratch checkpoint fingerprint is 3c7c420d30ac111d4b752322318d92c3b9dfa2077ba2f3d04bc6a71b6aa937ed. These are locally calculated identities; the manuscript does not claim publisher checksum verification where none was supplied.

## B. Reconstruction procedure

The manuscript builder performs the following checks before generating figures: read both JSONL files; require exactly 192 unique IDs per file; require identical ID sets and reference labels; reject overlap with quarantined pilot IDs; recompute 24 baseline and 182 transfer correct predictions; reconstruct both 8×8 matrices; verify the transfer matrix equals the evaluation artifact; verify top-five membership for 191 examples; and check selected-head, baseline-checkpoint, model-source, and feature-manifest fingerprints.

It also checks that all 27 head candidates exist and that the selection record reports no test use. It computes descriptive Wilson intervals and the paired contingency counts from saved outcomes. These are new arithmetic summaries of existing predictions, not new recognition experiments. The build writes an evidence receipt so the source hashes and calculations can be inspected independently of the PDF.

| Component | Source entry point |
|---|---|
| Main protocol and head | pipeline/word_recognition/pretrained.py |
| Frozen encoder adapter | pipeline/avhubert/loader.py |
| Historical baseline | pipeline/word_recognition/legacy_word_model.py |
| Word service | app/services/pretrained_word_inference.py |
| Publisher model | pipeline/lrw_tcn/inference.py |
| LRW alignment | pipeline/lrw_tcn/preprocessing.py |
| Sequence loss | alphabet/losses.py |
| Greedy decoder | alphabet/decoding.py |
| Frontend state | frontend/src/live-demo/wordConsensus.ts |

## C. Commands and preservation

From the repository root, the manuscript is rebuilt with `.venv\Scripts\python.exe scripts/build_ieee_evidence_paper.py`. It reads existing artifacts and writes the manuscript, editable document, figure files, and validation receipts. It does not train a model, change a checkpoint, alter backend selection, or restart the live server. The recorded original experiment used the pretrained module’s init, preparation/extraction, train, and evaluate stages, followed by the paired baseline comparison. Completed run directories should remain frozen.

A new experiment should use a new directory, a written selection protocol, and an independently held test partition. Copying the old head into a new folder does not create new experimental evidence. Likewise, rerunning evaluation repeatedly after changing thresholds turns the test into a development set unless a fresh final test is subsequently collected.

## D. Software validation records

The earlier audit’s targeted Python command covers CTC, vocabulary targets, dataset metrics, and LRW vocabulary and records 44 passing tests. Its live frontend test command records 14 passing and four failing tests, with the failures concentrated in word consensus. A prior full Python log records 499 passed and four skipped. These results have different source/time scopes and are reported as such. Test passes validate software contracts, not recognition accuracy.

The paper build validates exactly twenty PDF pages, required author and result text, source-derived figure inputs, and readable body content on every page. Editable Word pagination may differ by application and font settings. The PDF is the fixed twenty-page artifact; individual figures are also exported for reuse in a venue-specific template. No IEEE acceptance, publication status, DOI, copyright assignment, or institutional approval is asserted by this author-prepared manuscript.

## E. Source anchors for technical reproduction

The selected head-search implementation begins at `pretrained.py:208`; pooling begins at line 186 and evaluation at line 262. The representation adapter’s ROI preparation begins at `loader.py:118`, with frozen execution at line 149. The historical CNN–BiGRU definition begins at `legacy_word_model.py:11`. The alphabet forward function is at `models/alphabet_model.py:63`. These anchors refer to the audited source snapshot and should be accompanied by a revision identifier if the repository is released.

The CTC loss is implemented at `alphabet/losses.py:42`, and the greedy decoder class at `alphabet/decoding.py:41`. The live consensus defect is visible in `wordConsensus.ts:13`. The eight-word service’s current candidate-return behavior is at `pretrained_word_inference.py:47`. Listing these locations allows a reviewer to distinguish observed implementation details from methodological descriptions and to identify which files need reinspection after a future edit.

<!-- PAGE -->
# XX. Conclusion

The project demonstrates a reproducible limited-vocabulary visual word pipeline that executes on a Windows CPU. A frozen AV-HuBERT Base encoder, measured mouth alignment, temporal mean/std pooling, training-only scaling, and a validation-selected logistic head obtain 182/192 correct predictions on the specified eight-word MIRACL protocol. The preserved scratch model obtains 24/192 on the same source utterances. The resulting 82.29-percentage-point difference is a complete-system comparison, with changes in representation, preprocessing, and training-set size explicitly acknowledged.

The saved confusion matrix and per-class analysis identify previous as the weakest word, while per-speaker results expose variation hidden by the pooled score. Confidence rejection improves accepted accuracy at a substantial coverage cost and still falsely accepts eight of sixty diagnostic phrases. The current runtime threshold drift and confirmation regression reinforce the need to evaluate deployed state logic separately from offline classifier quality.

The repository’s other recognition modes remain distinct. The alphabet checkpoint predicts one of 26 isolated labels; the publisher LRW model has genuine 500-class weights but lacks a complete local authorized benchmark; GRID acquisition provides a validated corpus but no trained continuous decoder. Implemented CTC utilities, proposed language correction, and proposed speech synthesis must not be presented as completed recognition or generation results.

Future work should begin with a versioned repair of live confirmation, matched-data comparisons, and an untouched speaker cohort. Genuine continuous decoding, full LRW testing, calibrated open-set handling, and contextual correction/speech generation can then be evaluated under separate protocols. The present contribution is a measured transfer-learning result and a reproducible account of its boundaries, rather than a claim of general-purpose lip reading.

# Acknowledgment

Ravikumar Gupta and Omkar Gupta express their gratitude to Prof. Umang Patel for project supervision and guidance. The authors acknowledge the creators of AV-HuBERT, the publisher’s LRW model, and the MIRACL-VC1 and GRID resources used in this study. AI assistance was used for manuscript preparation; the authors are responsible for reviewing the technical content, results, and references.

# References

[1] Y. M. Assael, B. Shillingford, S. Whiteson, and N. de Freitas, “LipNet: End-to-End Sentence-level Lipreading,” arXiv:1611.01599, 2016. https://arxiv.org/abs/1611.01599

[2] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks,” in Proceedings of ICML, 2006. https://www.cs.toronto.edu/~graves/icml_2006.pdf

[3] B. Shi, W.-N. Hsu, K. Lakhotia, and A. Mohamed, “Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction,” ICLR, 2022, arXiv:2201.02184. https://arxiv.org/abs/2201.02184

[4] B. Martinez, P. Ma, S. Petridis, and M. Pantic, “Lipreading using Temporal Convolutional Networks,” arXiv:2001.08702, 2020. https://arxiv.org/abs/2001.08702

[5] P. Ma, Y. Wang, S. Petridis, J. Shen, and M. Pantic, “Training Strategies for Improved Lip-reading,” arXiv:2209.01383, 2022. Model repository: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

[6] A. Rekik, A. Ben-Hamadou, and W. Mahdi, “An adaptive approach for lip-reading using image and depth data,” Multimedia Tools and Applications, vol. 75, no. 14, pp. 8609–8636, 2016. Author dataset description and citation: https://abenhamadou.github.io/miraclvc1/

[7] University of Sheffield, “The GRID audiovisual sentence corpus,” official distribution and saved project acquisition index. https://spandh.dcs.shef.ac.uk/gridcorpus/ (the live index timed out during the manuscript check; the previously saved official index was inspected).

[8] Oxford Visual Geometry Group, “The Oxford-BBC Lip Reading in the Wild (LRW) Dataset,” official dataset specification and access conditions. https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

