"""Protocol tests use tiny synthetic models/data, never paper performance evidence."""
import ast
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_cases import ROOT, source_hashes
from train_and_export import FoldSaver, instrument_cross_validation, parse_cli, validate_args
import train_and_export


SOURCE = (ROOT / "main.py").read_text(encoding="utf-8-sig")


def original_cv(namespace):
    tree = ast.parse(SOURCE)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cross_validation")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "original_cv", "exec"), namespace)
    return namespace["cross_validation"]


class TinyDataset(Dataset):
    def __init__(self, size):
        self.features = [SimpleNamespace(topic_id="toy", doc_id=str(i)) for i in range(size)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(batch):
        return batch


class Processor:
    def generate_dataloader(self, split):
        data = TinyDataset(10 if split == "train" else 2)
        return data.features, data


def args_for_test():
    return SimpleNamespace(k_fold=5, seed=209, device=torch.device("cpu"), batch_size=1,
                           weight_decay=0.01, t_lr=2e-5, lr=1e-4, adam_epsilon=1e-8,
                           epoch=3, warm_up_ratio=0.1, test_epoch=1, valid_epoch=1,
                           cache_path=str(ROOT / "data/database"), dataset_type="EventStoryLine",
                           model_type="bert", inter_or_intra="intra_and_inter", log="toy.log")


def toy_environment():
    models, orders, prints = [], [], []

    class Model(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.pretrained_model = nn.Linear(1, 1)
            self.mlp = nn.Linear(1, 1)
            self.CGE = nn.Linear(1, 1)
            self.proj_head = nn.Linear(1, 1)
            models.append(self)

    def train_epoch(args, model, loader, optimizer, scheduler, epoch):
        for batch in loader:
            orders.append(tuple(batch))
            # Exercise all three RNG streams and actual optimizer/scheduler updates.
            x = torch.tensor([[random.random() + np.random.random()]]) + torch.rand(1, 1)
            loss = model.CGE(model.mlp(model.pretrained_model(x))).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        return 1., .1, .2, .15, .9, {}, {}

    def metric(epoch):
        return {"p": epoch / 10, "r": epoch / 10, "f1": epoch / 10}

    def test_epoch(args, model, loader, epoch):
        f1 = {1: .1, 2: .2, 3: .9}[epoch]
        return [], [], [], .1, .2, .3, f1, .8, metric(epoch), metric(epoch)

    def valid_epoch(args, model, loader, epoch):
        return .1, .2, .3, {1: .4, 2: .4, 3: .3}[epoch], .8, metric(epoch), metric(epoch)

    namespace = dict(np=np, torch=torch, KFold=KFold, Subset=Subset, DataLoader=DataLoader,
                     RandomSampler=RandomSampler, SequentialSampler=SequentialSampler,
                     AdamW=torch.optim.AdamW, train_epoch=train_epoch, test_epoch=test_epoch,
                     valid_epoch=valid_epoch, printlog=prints.append,
                     get_linear_schedule_with_warmup=lambda optimizer, **kw:
                     torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.))
    return namespace, Model, models, orders, prints


def seed():
    random.seed(209)
    np.random.seed(209)
    torch.manual_seed(209)


def test_original_five_fold_training_and_rng_unchanged_with_saves(tmp_path):
    hashes = source_hashes()
    args = args_for_test()
    ns1, model1, models1, orders1, prints1 = toy_environment()
    seed()
    original_cv(ns1)(args, model1, Processor())
    after1 = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    ns2, model2, models2, orders2, prints2 = toy_environment()
    saver = FoldSaver(tmp_path, args, hashes, {"train_features_sha256": "synthetic"})
    seed()
    instrument_cross_validation(SOURCE, ns2, saver.start, saver.save, saver.end)(args, model2, Processor())
    after2 = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert orders1 == orders2
    assert prints1 == prints2
    assert len(models1) == len(models2) == len(saver.runs) == 5
    assert after1[0] == after2[0]
    assert np.array_equal(after1[1][1], after2[1][1])
    assert after1[1][2:] == after2[1][2:]
    assert torch.equal(after1[2], after2[2])
    for a, b in zip(models1, models2):
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    for fold in range(1, 6):
        path = tmp_path / ("fold{}".format(fold))
        best = json.loads((path / "best.json").read_text(encoding="utf-8"))
        assert best["epoch"] == 2  # >= tie policy; epoch 3 test score is higher but dev is lower.
        assert best["test_at_same_epoch"]["f1"] == .2
        state = torch.load(path / "best.pt", weights_only=True)
        assert state["epoch"] == 2
        assert set(state["state_dict"]) == set(models1[0].state_dict())
        train = json.loads((path / "train_indices.json").read_text(encoding="utf-8"))
        test = json.loads((path / "test_indices.json").read_text(encoding="utf-8"))
        expected = list(KFold(5, shuffle=True, random_state=209).split(range(10)))[fold - 1]
        assert train == expected[0].tolist() and test == expected[1].tolist()
    assert source_hashes() == hashes


def test_saver_preserves_checkpoint_at_best_epoch(tmp_path):
    args = args_for_test()
    saver = FoldSaver(tmp_path, args, source_hashes(), {"train_features_sha256": "synthetic"})
    saver.start(args, 0, [0], [1], TinyDataset(2).features)
    model = nn.Linear(1, 1)
    saved = copy.deepcopy(model.state_dict())
    saver.save(args, 0, 7, model, {"f1": .5}, {"f1": .4})
    with torch.no_grad():
        model.weight.add_(99)
    loaded = torch.load(tmp_path / "fold1/best.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(v, loaded[k]) for k, v in saved.items())


def test_instrumentation_rejects_different_selection_policy():
    with pytest.raises(ValueError, match="structure changed"):
        instrument_cross_validation(SOURCE.replace("valid_f1 >= best_valid_f1", "valid_f1 > best_valid_f1"),
                                    {}, None, None, None)


def test_original_argument_parser_is_reused():
    wrapper, args = parse_cli(["--output", "new-output", "--epoch", "7", "--seed", "19",
                              "--model_name", "some-local-model", "--beta_intra", "0.7"])
    assert wrapper.output == "new-output"
    assert args.epoch == 7 and args.seed == 19 and args.loss_type == "focal"


@pytest.mark.parametrize("export_fails", [False, True])
def test_entry_auto_exports_after_all_folds_and_keeps_checkpoints(monkeypatch, tmp_path, export_fails):
    ns, Model, models, _, _ = toy_environment()
    # Exercise entry orchestration on CPU without changing production code's GPU requirement.
    monkeypatch.setattr(Model, "to", lambda self, *a, **kw: self)
    fake_original = SimpleNamespace(**ns)
    fake_original.setup_seed = lambda value: seed()
    fake_original.MODEL_CLASSES = {"bert": (None, Model, SimpleNamespace(
        from_pretrained=lambda *a, **kw: object()))}
    fake_original.ESC_processor = lambda *a: Processor()
    monkeypatch.setitem(sys.modules, "main", fake_original)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    actual_find_spec = train_and_export.importlib.util.find_spec
    monkeypatch.setattr(train_and_export.importlib.util, "find_spec",
                        lambda name: object() if name in ("torch_geometric", "torch_sparse")
                        else actual_find_spec(name))
    monkeypatch.setattr(train_and_export, "audit_data", lambda args:
                        {"train_features_sha256": "synthetic", "dev_features_sha256": "synthetic"})
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        manifest = json.loads((tmp_path / "run/manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["runs"]) == len(models) == 5
        assert all((tmp_path / "run" / r["checkpoint"]).is_file() for r in manifest["runs"])
        if export_fails:
            raise RuntimeError("synthetic export failure")
        cases = tmp_path / "run/cases"
        cases.mkdir()
        (cases / "selection.json").write_text(json.dumps(
            {"status": "insufficient_candidates", "missing_roles": ["failure"]}), encoding="utf-8")

    monkeypatch.setattr(train_and_export.subprocess, "run", launch)
    args = args_for_test()
    # The synthetic orchestration test must not require private model weights
    # or the distributed corpus caches. The fake model/processor ignore inputs.
    encoder = tmp_path / "synthetic_encoder"
    encoder.mkdir()
    cache = tmp_path / "synthetic_cache"
    cache.mkdir()
    for split in ("train", "dev"):
        (cache / "EventStoryLine_bert_intra_and_inter_{}_features.pkl".format(split)).write_bytes(b"synthetic fixture")
    args.model_name = str(encoder)
    args.cache_path = str(cache)
    args.gradient_accumulation_steps = 1
    args.disable_cuda = False
    wrapper = SimpleNamespace(output=str(tmp_path / "run"), preflight=False)
    if export_fails:
        with pytest.raises(RuntimeError, match="synthetic export failure"):
            train_and_export.run(wrapper, args)
    else:
        train_and_export.run(wrapper, args)
    status = json.loads((tmp_path / "run/status.json").read_text(encoding="utf-8"))
    assert status["training_complete"]
    assert status["stage"] == ("trajectory_export_failed" if export_fails else "complete")
    assert len(calls) == 1
    assert (tmp_path / "run/fold5/best.pt").is_file()


@pytest.mark.parametrize("key,value", [("k_fold", 3), ("batch_size", 2), ("test_epoch", 2)])
def test_reject_ambiguous_protocol_overrides(key, value):
    args = args_for_test()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        validate_args(args)
