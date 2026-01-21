from __future__ import annotations

from pathlib import Path

from process_optimizer.storage.vector_store import VectorStore


def test_vector_store_tfidf_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "vs.joblib"
    vs = VectorStore(db, backend="tfidf")
    vs.add_documents(
        [
            ("temperature affects thickness", {"doc_id": "a"}),
            ("pressure affects sheet resistance", {"doc_id": "b"}),
        ]
    )

    vs2 = VectorStore.load(db)
    out = vs2.query("sheet resistance", top_k=1)
    assert out
    assert out[0].doc_id == "b"
