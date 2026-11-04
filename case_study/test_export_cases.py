"""Tests use synthetic predictions; they are not paper case evidence."""
import importlib.util
import json
import pickle
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import export_cases

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_cases import (ROOT, TrajectoryObserver, case_behaviour, heldout_indices,
                          load_model, save_selection, select_cases, source_hashes, trace_forward)


def record(doc, gold, first, last, rel=None):
    return {"topic_id": "topic", "doc_id": doc, "run_id": "fold1", "held_out": True,
            "node_count": 4, "sentence_count": 3, "gold_causal_count": sum(g != 0 for g in gold),
            "gold_labels": gold, "rel_type": rel or [1] * len(gold),
            "rounds": [{"predicted_labels": first}, {"predicted_labels": last}]}


def test_mechanism_selection_and_deterministic_ties():
    a = record("a", [1, 0], [0, 0], [1, 0])
    b = record("b", [1, 0], [2, 1], [1, 0])
    c = record("c", [1, 0], [0, 0], [0, 1])
    result = select_cases([c, b, a])
    assert result["status"] == "complete"
    assert [r["doc_id"] for r in result["selected"]] == ["a", "b", "c"]
    assert [r["doc_id"] for r in select_cases([a, b, c])["selected"]] == ["a", "b", "c"]
    assert case_behaviour(b)["directions_corrected"] == [0]
    assert case_behaviour(b)["false_edges_removed"] == [1]


def test_no_fabricated_success_and_no_duplicate_documents():
    failed = record("a", [1], [0], [0])
    selection = select_cases([failed])
    assert selection["status"] == "insufficient_candidates"
    assert len(selection["selected"]) == 1
    with pytest.raises(ValueError, match="Repeated documents"):
        select_cases([failed, failed])
    failed["held_out"] = False
    assert not select_cases([failed])["selected"]


def test_regression_excludes_success_and_one_round_excluded():
    r = record("a", [1, 1, 0], [0, 0, 0], [1, 1, 1])
    assert not case_behaviour(r)["success"]
    assert case_behaviour(r)["failure"]
    r["rounds"] = r["rounds"][:1]
    assert not case_behaviour(r)["success"]
    assert not case_behaviour(r)["failure"]


def test_explicit_split_checks_document_overlap(tmp_path):
    (tmp_path / "train.json").write_text("[0]", encoding="utf-8")
    (tmp_path / "test.json").write_text("[1]", encoding="utf-8")
    fs = [types.SimpleNamespace(topic_id="t", doc_id="same") for _ in range(2)]
    split = {"kind": "explicit", "training_indices": "train.json", "evaluation_indices": "test.json"}
    with pytest.raises(ValueError, match="documents overlap"):
        heldout_indices(split, fs, tmp_path)
    fs[1].doc_id = "other"
    assert heldout_indices(split, fs, tmp_path)[0] == [1]


def test_kfold_matches_original_split(tmp_path):
    from sklearn.model_selection import KFold
    fs = [types.SimpleNamespace(topic_id="t", doc_id=str(i)) for i in range(13)]
    expected = list(KFold(n_splits=5, shuffle=True, random_state=209).split(fs))[2][1].tolist()
    spec = {"kind": "kfold", "fold": 3, "n_splits": 5, "seed": 209}
    assert heldout_indices(spec, fs, tmp_path)[0] == expected
    with pytest.raises(ValueError, match="one-based"):
        heldout_indices(dict(spec, fold=0), fs, tmp_path)


def test_encoder_only_checkpoint_is_rejected_before_graph_import(tmp_path):
    config = json.loads((ROOT / "case_study/run.config.example.json").read_text(encoding="utf-8"))
    weights = tmp_path / "encoder.pt"
    torch.save({"embeddings.word_embeddings.weight": torch.ones(2, 2)}, weights)
    with pytest.raises(ValueError, match="encoder-only"):
        load_model(weights, config, "cpu")


@pytest.mark.parametrize("changed", ["source", "cache"])
def test_training_manifest_rejects_changed_inputs(tmp_path, changed):
    manifest = {"runs": []}
    if changed == "source":
        manifest["expected_source_sha256"] = {"main.py": "wrong"}
    else:
        (tmp_path / "checkpoint.pt").write_bytes(b"synthetic")
        (tmp_path / "cache.pkl").write_bytes(b"synthetic")
        manifest["runs"] = [{"run_id": "synthetic", "training_provenance": "test only",
                             "checkpoint": "checkpoint.pt", "features": "cache.pkl",
                             "expected_features_sha256": "wrong"}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes differ|cache changed"):
        export_cases.export_manifest(tmp_path / "manifest.json", tmp_path / "failed_export", "cpu")
    assert (tmp_path / "failed_export/FAILED.json").is_file()


def test_selection_outputs_are_machine_readable(tmp_path):
    rs = [record("a", [1], [0], [1]), record("b", [1], [2], [1]),
          record("c", [1], [0], [0])]
    save_selection(rs, tmp_path)
    selected = json.loads((tmp_path / "selected_cases.json").read_text(encoding="utf-8"))
    assert selected["status"] == "complete"
    assert len(selected["cases"]) == 3
    assert (tmp_path / "candidates.csv").read_text(encoding="utf-8-sig").count("\n") == 4


def make_original_forward_model(monkeypatch, mode):
    """Execute actual model.py forward/construction; substitute encoder and CGE only.

    This verifies observer parity, not real BERT/GNN performance.
    """
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BertModel = object
    package = types.ModuleType("_case_test")
    package.__path__ = []
    fake_cge = types.ModuleType("_case_test.CGE")
    fake_cge.CGEConv = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "_case_test", package)
    monkeypatch.setitem(sys.modules, "_case_test.CGE", fake_cge)
    spec = importlib.util.spec_from_file_location("_case_test.model", ROOT / "model/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Encoder(nn.Module):
        def forward(self, ids, mask):
            return (torch.ones(*ids.shape, 768),)

    class CGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleDict({"event": nn.Identity()})

        def forward(self, nodes, edges):
            return {"event": nodes["event"] + 0.1}

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            stage = self.calls % 2
            self.calls += 1
            if mode == "empty":
                return torch.tensor([[9., 0., 0.], [9., 0., 0.]])
            if mode == "threshold":
                return torch.tensor([[0., 0.1, 0.], [0., 0., 0.1]])
            if stage == 0:
                return torch.tensor([[9., 0., 0.], [9., 0., 0.]])
            return torch.tensor([[0., 9., 0.], [0., 0., 9.]])

    model = module.bertCSRModel.__new__(module.bertCSRModel)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.pretrained_model = Encoder()
    model.CGE = CGE()
    model.mlp = MLP()
    model.proj_head = nn.Identity()
    model.ce_loss = nn.CrossEntropyLoss()
    model.loss_type = "ce"
    model.lambda_contrast = 0.1
    model.max_iteration = 4
    model.min_iteration = 2
    model.threshold = 99
    model.rate = 0.0
    model.w = 0.6
    model.eval()
    inputs = dict(enc_input_ids=torch.ones(3, 4, dtype=torch.long),
                  enc_mask_ids=torch.ones(3, 4, dtype=torch.long),
                  node_event=[[0, [1, 2]], [1, [1, 2]], [2, [1, 2]]],
                  t1_pos=[[0, 1]], t2_pos=[[1, 2]], target=[[1, 2]],
                  rel_type=[[1, 0]], event_pairs=[[0, 1], [1, 2]])
    return model, inputs


@pytest.mark.parametrize("mode", ["change", "empty", "threshold"])
def test_original_forward_parity_and_actual_edges(monkeypatch, mode):
    hashes = source_hashes()
    model, inputs = make_original_forward_model(monkeypatch, mode)
    rounds, loss = trace_forward(model, inputs, verify=True)
    assert len(rounds) == 2
    assert loss >= 0
    assert rounds[0]["used_for_next_prediction"]
    assert not rounds[-1]["used_for_next_prediction"]
    assert rounds[-1]["stop_difference"] is not None
    assert "get_graphedge_index" not in model.__dict__
    assert "Contrast_pre" not in model.__dict__
    assert not model.CGE._forward_pre_hooks
    if mode == "change":
        assert rounds[0]["constructed_edges"] == []
        edges = {tuple(e["edge_type"]): e["edge_index"] for e in rounds[-1]["constructed_edges"]}
        actual = {tuple(e["edge_type"]): e["edge_index"] for e in rounds[-1]["cge_input_edges"]}
        assert edges[("event", "inter", "event")] == [[0], [1]]
        assert edges[("event", "intra", "event")] == [[2], [1]]
        assert actual[("event", "intra", "event")] == [[2, 0, 1, 2], [1, 0, 1, 2]]
    else:
        assert all(not r["constructed_edges"] and not r["cge_input_edges"] for r in rounds)
        if mode == "threshold":
            assert rounds[0]["predicted_labels"] == [1, 2]
    assert hashes == source_hashes()


def test_observer_restores_after_exception(monkeypatch):
    model, _ = make_original_forward_model(monkeypatch, "empty")
    with pytest.raises(RuntimeError):
        with TrajectoryObserver(model):
            raise RuntimeError("test")
    assert "get_graphedge_index" not in model.__dict__
    assert not model.CGE._forward_pre_hooks


def test_export_pipeline_with_synthetic_model_and_real_forward(monkeypatch, tmp_path):
    model, inputs = make_original_forward_model(monkeypatch, "change")
    features = []
    for i in range(4):
        features.append(types.SimpleNamespace(
            topic_id="synthetic", doc_id=str(i), enc_text=["test only"],
            enc_tokens=[["test"] * 4] * 3, sentences=["test"] * 3,
            enc_input_ids=inputs["enc_input_ids"], enc_mask_ids=inputs["enc_mask_ids"],
            node_event=inputs["node_event"], t1_pos=inputs["t1_pos"][0],
            t2_pos=inputs["t2_pos"][0], target=[0, 0] if i == 2 else [1, 2],
            rel_type=inputs["rel_type"][0], event_pairs=inputs["event_pairs"]))
    with (tmp_path / "features.pkl").open("wb") as stream:
        pickle.dump(features, stream)
    (tmp_path / "train.json").write_text("[3]", encoding="utf-8")
    (tmp_path / "test.json").write_text("[0, 1, 2]", encoding="utf-8")
    config = json.loads((ROOT / "case_study/run.config.example.json").read_text(encoding="utf-8"))
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "synthetic.pt").write_bytes(b"test fixture; never real weights")
    manifest = {"runs": [{"run_id": "synthetic", "checkpoint": "synthetic.pt",
                          "config": "config.json", "features": "features.pkl",
                          "training_provenance": "Synthetic test only",
                          "split": {"kind": "explicit", "training_indices": "train.json",
                                    "evaluation_indices": "test.json"}}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(export_cases, "load_model", lambda *args: model)
    output = tmp_path / "exported"
    export_cases.export_manifest(tmp_path / "manifest.json", output, "cpu")
    selected = json.loads((output / "selected_cases.json").read_text(encoding="utf-8"))
    assert selected["status"] == "complete"
    assert len(selected["cases"]) == 3
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_sha256_before"] == provenance["source_sha256_after"]
    assert len((output / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert all(case["trajectory"]["observation_parity"] == "exact_logits_and_loss_passed"
               for case in selected["cases"])
    with pytest.raises(FileExistsError):
        export_cases.export_manifest(tmp_path / "manifest.json", output, "cpu")
