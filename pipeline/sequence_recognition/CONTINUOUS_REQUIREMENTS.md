# Continuous silent-video recognition requirements

The 500-class ResNet18/DC-TCN pools a video into a single class distribution. The eight-word head also pools temporal representations. Neither generates token sequences. Concatenating guesses is not sentence recognition. The sentence track remains unavailable in `/api/models`; no transcript is manufactured.

The official AV-HuBERT encoder yields representations, not text. Continuous recognition needs:

- A legitimately acquired visual-only supervised sequence decoder checkpoint matching the encoder architecture/version. Audio-dependent models cannot silently be presented as video-only.
- Its exact dictionary/tokenizer, special symbols, text normalization and decoder configuration. The LRW class list is not a sentence tokenizer.
- Matching video alignment, native frame rates and full utterance lengths. Stretching every sentence to the word experiment's 29 frames destroys the task.
- Legitimate sentence videos/transcripts and an explicit evaluation protocol. GRID has constrained six-slot grammar; its letter-slot manifests cannot supervise sentence recognition. TCD/LRS3/MuAViC access and authentic local files must be established first.
- Real output validation: word/token error metrics on held-out speakers/content, actual greedy/beam decoding, CPU/RAM measurements and separation of language-model assistance from visual evidence.
- Hardware capacity for the exact model/utterance lengths. The i3/7.74-GiB CPU machine has limited free memory; no real-time sentence throughput is promised.

No compatible trained sentence decoder/tokenizer bundle is verified locally. This phase delivers documented requirements and an unavailable capability state. Existing CTC collapse/edit-distance helpers are software utilities, not model evidence. No guessed vocabulary, invented transcript or benchmark is introduced.
