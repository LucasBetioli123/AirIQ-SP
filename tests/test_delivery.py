"""Regressões da revisão: causalidade, configuração e preservação da base normativa."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.common.config import ROOT, load_env


def test_dotenv_respects_exported_values_and_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEEP", "externo")
    for key in ("TEST_NEW", "TEST_LITERAL", "TEST_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    p = tmp_path / ".env"
    p.write_text('TEST_KEEP=interno\nTEST_NEW="texto com espaço"\n'
                 'TEST_LITERAL=$(comando)\nTEST_EMPTY=\n', encoding="utf-8")
    load_env(p)
    assert os.environ["TEST_KEEP"] == "externo"
    assert os.environ["TEST_NEW"] == "texto com espaço"
    assert os.environ["TEST_LITERAL"] == "$(comando)"
    assert os.environ["TEST_EMPTY"] == ""


def test_redirected_output_is_utf8_without_make():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    out = subprocess.run([sys.executable, "-c", "import src.common.config; print('PM2.5 ≤ 15 µg/m³')"],
                         cwd=ROOT, env=env, capture_output=True, check=True).stdout
    assert out.decode("utf-8").strip() == "PM2.5 ≤ 15 µg/m³"


@pytest.mark.parametrize("forecast", [False, True])
def test_features_do_not_require_any_future_observation(forecast):
    from src.aq.features import load_context, point_features
    ctx = load_context(True)
    day = pd.Timestamp("2021-08-24")
    past = {key: value.sel(time=value.time <= day) if hasattr(value, "time") else value
            for key, value in ctx.items()}
    past["fires"] = {d: f for d, f in ctx["fires"].items() if d <= day}
    args = ([day], [-22.9], [-47.06])
    pd.testing.assert_frame_equal(point_features(ctx, *args, forecast=forecast),
                                  point_features(past, *args, forecast=forecast))
    assert not any(c.startswith("next_") and c not in ("next_dow", "next_weekend")
                   for c in point_features(past, *args, forecast=forecast))


def test_incomplete_ingest_preserves_existing_base_and_sample(tmp_path, monkeypatch):
    from src.rag import ingest
    paths = {key: tmp_path / key for key in ("processed", "sample")}
    for p in paths.values():
        p.mkdir()
    sample = paths["sample"] / "rag_chunks.jsonl"
    sample.write_text(json.dumps({"doc_id": "norma"}) + "\n", encoding="utf-8")
    out = paths["processed"] / "chunks.jsonl"
    out.write_text("base existente", encoding="utf-8")
    monkeypatch.setattr(ingest, "registry", lambda: [{"id": "norma"}])
    monkeypatch.setattr(ingest, "existing", lambda _: None)
    monkeypatch.setattr(ingest, "chunks_path", lambda: out)
    monkeypatch.setattr(ingest, "data_path", lambda key, *parts: paths[key].joinpath(*parts))
    before = sample.read_bytes()
    ingest.main()
    assert out.read_text(encoding="utf-8") == "base existente"
    assert sample.read_bytes() == before
    out.unlink()
    ingest.main()
    assert out.read_bytes() == before
    assert sample.read_bytes() == before


def test_retriever_uses_complete_sample_after_partial_ingestion(tmp_path, monkeypatch):
    from src.rag import download, retrieve
    processed, sample = tmp_path / "processed", tmp_path / "sample"
    (processed / "rag").mkdir(parents=True)
    sample.mkdir()
    (processed / "rag" / "chunks.jsonl").write_text('{"doc_id":"a"}\n', encoding="utf-8")
    (sample / "rag_chunks.jsonl").write_text('{"doc_id":"a"}\n{"doc_id":"b"}\n', encoding="utf-8")
    monkeypatch.setattr(download, "registry", lambda: [{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(retrieve, "data_path", lambda key, *parts: (processed if key == "processed" else sample).joinpath(*parts))
    assert {c["doc_id"] for c in retrieve.load_chunks()} == {"a", "b"}
