"""Adversarial modality/source checks for the optional historical covariate."""
from copy import deepcopy

import pytest

from v3.speaker_coverage import voice_diagnostic as v


@pytest.fixture
def vote_source(tmp_path, monkeypatch):
    (tmp_path / "processedResults").mkdir()
    source = {
        "README.md": "fixture, not official evidence",
        "tabulateVotesV2.r": "(100000*testResponses$queryType) + testResponses$clipNum",
        "summarizeVotes.r": "voiceTestResponses <- subset(testResponses,queryType==1)",
        "SentenceFilenames.csv": "fixture", "processedResults/tabulatedVotes.csv": "fixture",
        "processedResults/summaryTable.csv": "fixture",
    }
    for name, text in source.items():
        (tmp_path / name).write_text(text)
    names = [{"Stimulus_Number": str(i), "Filename": f"clip{i}"} for i in range(1, 7443)]
    summaries = [{"FileName": f"clip{i}", "VoiceVote": "A", "FaceVote": "H", "MultiModalVote": "F"} for i in range(1, 7443)]
    rows = [{"": str(q * 100000 + i), "fileName": f"clip{i}", "A": "4", "D": "1", "F": "0", "H": "0",
             "N": "0", "S": "0", "numResponses": "5", "agreement": "0.8", "emoVote": "A"}
            for q in range(1, 4) for i in range(1, 7443)]
    tables = {"SentenceFilenames.csv": names, "summaryTable.csv": summaries, "tabulatedVotes.csv": rows}
    monkeypatch.setattr(v, "read_csv", lambda path: deepcopy(tables[path.name]))
    return tmp_path, tables


def test_video_values_are_not_read_as_voice(vote_source):
    root, tables = vote_source
    for row in tables["tabulatedVotes.csv"][7442:]:
        row["emoVote"], row["numResponses"], row["A"] = "VIDEO_NOT_VOICE", "unread", "unread"
    voices, report = v.audit_votes(root)
    assert len(voices) == 7442 and voices["clip1"]["winners"] == ("A",)
    assert report["visual_vote_values_used"] is False


@pytest.mark.parametrize("field,value,message", [
    ("fileName", "clip2", "encoding/filename"),
    ("numResponses", "7", "voice counts"),
    ("agreement", "0.9", "agreement/count"),
    ("emoVote", "H", "voice majority"),
])
def test_inconsistent_source_rejected(vote_source, field, value, message):
    root, tables = vote_source
    tables["tabulatedVotes.csv"][0][field] = value
    with pytest.raises(ValueError, match=message):
        v.audit_votes(root)


def test_tied_modes_must_match_both_voice_tables(vote_source):
    root, tables = vote_source
    row = tables["tabulatedVotes.csv"][0]
    row.update(A="2", D="2", numResponses="4", agreement="0.5", emoVote="A:D")
    tables["summaryTable.csv"][0]["VoiceVote"] = "A:D"
    voices, report = v.audit_votes(root)
    assert voices["clip1"]["winners"] == ("A", "D") and report["voice_tied_pluralities"] == 1
    tables["summaryTable.csv"][0]["VoiceVote"] = "A"
    with pytest.raises(ValueError, match="voice majority"):
        v.audit_votes(root)


def test_ambiguous_query_type_source_fails_closed(vote_source):
    root, _ = vote_source
    (root / "summarizeVotes.r").write_text("voiceTestResponses <- subset(testResponses,queryType==2)")
    with pytest.raises(ValueError, match="does not identify voice-only"):
        v.audit_votes(root)


def test_rank_partial_undefined_when_control_exhausts_signal():
    assert v.correlation([1, 2, 3, 4], [4, 3, 1, 2], [[1, 2, 3, 4]]) is None
