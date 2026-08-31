"""Phase 5 tests: vocabulary, target encoding, phonetic/viseme supervision."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.targets import (
    PAD_TARGET_ID, TargetType, classify_target, collate_targets,
    encode_text, normalize_text,
)
from alphabet.phonetics import (
    LETTER_NAME_PHONES_VERSION, PhoneVocabulary, VisemeVocabulary,
    letter_to_phones, text_to_phone_sequence, text_to_viseme_sequence,
)
from alphabet.vocab import (
    CTC_BLANK_TOKEN, DEFAULT_VOCABULARY, LetterVocabulary, OutputState,
    VocabularyError,
)


class TestVocabulary:
    def test_canonical_size_27(self):
        v = DEFAULT_VOCABULARY
        assert v.size == 27
        assert v.blank_index == 0
        assert v.num_letters == 26
        assert v.letters_offset == 1

    def test_single_source_is_importable_and_consistent(self):
        from alphabet.heads import MultiTaskAlphabetModel  # noqa: F401
        from temporal.encoder import TemporalEncoderConfig
        model = MultiTaskAlphabetModel(TemporalEncoderConfig(), DEFAULT_VOCABULARY.size)
        assert model.letter_head.proj.out_features == 27

    def test_encode_decode_roundtrip(self):
        v = DEFAULT_VOCABULARY
        for text in ("A", "Z", "HELLO", "QWERTY"):
            ids = v.encode(text)
            assert v.decode_ids(ids) == text
            assert all(1 <= i <= 26 for i in ids)

    def test_blank_is_not_encodable(self):
        with pytest.raises(VocabularyError):
            DEFAULT_VOCABULARY.encode(CTC_BLANK_TOKEN)
        with pytest.raises(VocabularyError):
            DEFAULT_VOCABULARY.decode_ids([0])

    def test_rejects_non_letters(self):
        for bad in ("1", "AA!", "hello!", "", "Ä"):
            with pytest.raises(VocabularyError):
                DEFAULT_VOCABULARY.encode(bad)

    def test_save_load_roundtrip(self, tmp_path):
        p = tmp_path / "vocab.json"
        DEFAULT_VOCABULARY.save(p)
        loaded = LetterVocabulary.load(p)
        assert loaded == DEFAULT_VOCABULARY or loaded.to_dict() == DEFAULT_VOCABULARY.to_dict()

    def test_output_states_are_post_processing_only(self):
        assert OutputState.UNKNOWN.value == "unknown"
        assert OutputState.AMBIGUOUS.value == "ambiguous"
        # not part of the model vocabulary:
        assert "UNKNOWN" not in DEFAULT_VOCABULARY.letters
        assert "AMBIGUOUS" not in DEFAULT_VOCABULARY.letters


class TestTargetEncoding:
    def test_single_letter_target(self):
        assert encode_text(DEFAULT_VOCABULARY, "a") == [1]  # normalized + A=1
        assert classify_target("A") is TargetType.SINGLE_LETTER

    def test_multi_letter_target(self):
        ids = encode_text(DEFAULT_VOCABULARY, "HELLO")
        assert classify_target("HELLO") is TargetType.MULTI_LETTER
        assert ids == [8, 5, 12, 12, 15]

    def test_repeated_letters_encoded_literally(self):
        """LL encodes as two ids; CTC blanks make it decodable (see ctc tests)."""
        assert encode_text(DEFAULT_VOCABULARY, "LL") == [12, 12]
        assert encode_text(DEFAULT_VOCABULARY, "L") == [12]

    def test_collate_pads_and_reports_lengths(self):
        batch = collate_targets(DEFAULT_VOCABULARY, ["A", "HELLO"])
        assert batch.targets.shape == (2, 5)
        assert batch.target_lengths.tolist() == [1, 5]
        assert batch.targets[0].tolist() == [1, PAD_TARGET_ID, PAD_TARGET_ID,
                                             PAD_TARGET_ID, PAD_TARGET_ID]
        assert batch.targets[1].tolist() == [8, 5, 12, 12, 15]

    def test_collate_rejects_empty(self):
        with pytest.raises(VocabularyError):
            collate_targets(DEFAULT_VOCABULARY, [])


class TestPhoneticsSupervision:
    def test_versions_are_pinned(self):
        assert LETTER_NAME_PHONES_VERSION == "letter_names_arpabet_us_v1.0"
        assert PhoneVocabulary().version == LETTER_NAME_PHONES_VERSION
        assert VisemeVocabulary().version == "bear_harvey_initial_v1.0"

    def test_phone_vocab_covers_all_letter_names(self):
        pv = PhoneVocabulary()
        for letter, phones in [
            (L, letter_to_phones(L)) for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ]:
            ids = pv.encode(list(phones))
            assert all(1 <= i < pv.size for i in ids)
        assert pv.blank_index == 0
        assert SIL_OK(pv.size)

    def test_single_letter_phone_string(self):
        assert letter_to_phones("B") == ("B", "IY")
        assert letter_to_phones("Q") == ("K", "Y", "UW")

    def test_multi_letter_phone_sequence_has_sil_separators(self):
        seq = text_to_phone_sequence("LL")
        assert seq == ["EH", "L", "SIL", "EH", "L"]
        vis = text_to_viseme_sequence("LL")
        assert vis == ["V11", "V9", "SIL", "V11", "V9"]

    def test_phone_ctc_can_distinguish_ll_from_l(self):
        """With SIL between letters, LL and L have different target strings."""
        pv = PhoneVocabulary()
        assert pv.encode(text_to_phone_sequence("LL")) != \
            pv.encode(text_to_phone_sequence("L"))


def SIL_OK(size):
    return size > 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
