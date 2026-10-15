"""Offline regression tests for dataset sampling and random-state isolation."""
import json
import random

import pytest

from dataset.ethic import ETHICDataset
from dataset.financebench import FinanceBenchDataset
from dataset.hotpotqa import HotpotQADataset
from dataset.longbench import LongBenchDataset
from dataset.msmarco import MSMARCODataset
from dataset.qmsum import QMSumDataset


@pytest.fixture(autouse=True)
def isolate_random_state():
    state = random.getstate()
    random.seed(8675309)
    yield
    random.setstate(state)


@pytest.fixture(params=["hotpotqa", "ethic", "msmarco", "qmsum", "financebench", "longbench"])
def dataset(request, monkeypatch, tmp_path):
    name = request.param
    rows = []
    for i in range(10):
        if name == "hotpotqa":
            row = {"id": str(i), "question": "q", "answer": "a",
                   "context": {"title": ["title"], "sentences": [["context"]]}}
        elif name == "ethic":
            row = {"ID": str(i), "User_msg": "context", "Answer": ["a"],
                   "Domain": "domain", "Task": "task", "System_msg": "system"}
        elif name == "msmarco":
            row = {"query_id": i, "query": "q", "answers": ["a"],
                   "passages": {"passage_text": ["context"], "is_selected": [1]}}
        elif name == "qmsum":
            row = {"id": str(i), "meeting_transcripts": [{"speaker": "S", "content": "context"}],
                   "general_query_list": [{"query": "q", "answer": "a"}]}
        elif name == "financebench":
            row = {"financebench_id": str(i), "question": "q", "answer": "a",
                   "evidence": [{"evidence_text_full_page": "context"}]}
        else:
            row = {"context": "context", "input": "q", "answers": ["a"]}
        rows.append(row)

    if name == "longbench":
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "narrativeqa.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
        monkeypatch.setattr("dataset.longbench.LONGBENCH_CACHE_DIR", tmp_path)
        return LongBenchDataset()
    if name == "financebench":
        monkeypatch.setattr("dataset.financebench.hf_load_dataset", lambda *a, **kw: rows)
        return FinanceBenchDataset(pdf_dir=tmp_path / "missing-pdfs")

    monkeypatch.setattr(f"dataset.{name}.hf_datasets.load_dataset", lambda *a, **kw: rows)
    return {"hotpotqa": HotpotQADataset, "ethic": ETHICDataset,
            "msmarco": MSMARCODataset, "qmsum": QMSumDataset}[name]()


def test_sampling_preserves_global_random_state(dataset):
    state = random.getstate()
    assert len(list(dataset.load_examples(limit=3, seed=42))) == 3
    assert random.getstate() == state


def test_sampling_preserves_existing_seeded_selection(dataset):
    all_examples = list(dataset.load_examples())
    expected_indices = sorted(random.Random(42).sample(range(len(all_examples)), 3))
    expected_ids = [all_examples[i]["id"] for i in expected_indices]
    first = list(dataset.load_examples(limit=3, seed=42))
    second = list(dataset.load_examples(limit=3, seed=42))
    assert [example["id"] for example in first] == expected_ids
    assert second == first
    assert list(dataset.load_examples(limit=3, seed=7)) != first


@pytest.mark.parametrize("limit", [None, 0, 10, 20])
def test_sampling_limits(dataset, limit):
    state = random.getstate()
    examples = list(dataset.load_examples(limit=limit))
    assert len(examples) == (0 if limit == 0 else 10)
    assert random.getstate() == state
