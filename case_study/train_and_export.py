"""Independent five-fold SAEF entry: original training + best-dev saves + export.

Only adds IO callbacks to an in-memory AST of the original cross_validation.
Original source files and optimizer/training/evaluation functions are untouched.
"""
import argparse
import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in (ROOT, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from export_cases import (CachedFeature, load_features, plain, read_json, sha256, source_hashes,
                          write_json)

# Legacy caches may refer to __main__.ESC_features when run as a script.
ESC_features = CachedFeature

CALLBACKS = ("__case_fold_start", "__case_save_best", "__case_fold_end")


def instrument_cross_validation(source, namespace, start, save, end):
    """Fail closed on unfamiliar function structure; prove callback-only edits."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "cross_validation"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one original cross_validation")
    function = functions[0]
    before = ast.dump(function, include_attributes=False)
    folds = [n for n in ast.walk(function) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple)
             and isinstance(n.target.elts[0], ast.Name) and n.target.elts[0].id == "fold"]
    best = [n for n in ast.walk(function) if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name) and n.test.left.id == "valid_f1"
            and len(n.test.ops) == 1 and isinstance(n.test.ops[0], ast.GtE)
            and isinstance(n.test.comparators[0], ast.Name)
            and n.test.comparators[0].id == "best_valid_f1"]
    if len(folds) != 1 or len(best) != 1 or best[0] not in list(ast.walk(folds[0])):
        raise ValueError("Original fold/selection structure changed; inspect main.py before adapting")
    expected = {"best_valid_f1", "best_valid_test_f1", "best_valid_test_r",
                "best_valid_test_p", "best_valid_test_acc", "best_valid_test_intra",
                "best_valid_test_cross"}
    assigned = {t.id for n in best[0].body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    if not expected <= assigned:
        raise ValueError("Original best-dev branch changed")

    def expr(text):
        return ast.parse(text).body[0]

    folds[0].body.insert(0, expr("__case_fold_start(args, fold, train_idx, test_idx, train_test_features)"))
    best[0].body.append(expr(
        "__case_save_best(args, fold, epoch, model, "
        "{'loss': valid_loss, 'p': valid_p, 'r': valid_r, 'f1': valid_f1, 'acc': valid_acc, "
        "'intra': valid_intra, 'cross': valid_cross}, "
        "{'loss': test_loss, 'p': test_p, 'r': test_r, 'f1': test_f1, 'acc': test_acc, "
        "'intra': test_intra, 'cross': test_cross})"))
    folds[0].body.append(expr("__case_fold_end(fold)"))

    class RemoveCallbacks(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in CALLBACKS):
                return None
            return self.generic_visit(node)

    # Verify stripping only the three added calls recovers the exact original AST.
    import copy
    restored = RemoveCallbacks().visit(copy.deepcopy(function))
    if ast.dump(restored, include_attributes=False) != before:
        raise AssertionError("Instrumentation changed more than save callbacks")
    globals_copy = dict(namespace)
    globals_copy.update(dict(zip(CALLBACKS, (start, save, end))))
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ROOT / "main.py") + " [save callbacks]", "exec"), globals_copy)
    return globals_copy["cross_validation"]


def cached_path(args, split):
    return Path(args.cache_path) / "{}_{}_{}_{}_features.pkl".format(
        args.dataset_type, args.model_type, args.inter_or_intra, split)


def validate_args(args):
    if args.k_fold != 5:
        raise ValueError("This entry preserves the original five-fold protocol: --k_fold must be 5")
    if args.batch_size != 1:
        raise ValueError("Original ESC collate uses one document; use --batch_size 1")
    if args.valid_epoch != 1 or args.test_epoch != 1:
        raise ValueError("Use --valid_epoch 1 --test_epoch 1 to save same-epoch dev-selected test results")
    if args.epoch < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("epoch and gradient_accumulation_steps must be positive")
    args.model_name = str(Path(args.model_name).resolve())
    args.cache_path = str(Path(args.cache_path).resolve())
    if not Path(args.model_name).is_dir():
        raise FileNotFoundError("Local BERT directory not found: " + args.model_name)
    for split in ("train", "dev"):
        if not cached_path(args, split).is_file():
            raise FileNotFoundError(cached_path(args, split))


def audit_data(args):
    train, dev = [load_features(cached_path(args, name)) for name in ("train", "dev")]
    identities = lambda fs: [(str(f.topic_id), str(f.doc_id)) for f in fs]
    train_ids, dev_ids = identities(train), identities(dev)
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("Duplicate documents in feature caches")
    if set(train_ids) & set(dev_ids):
        raise ValueError("Development documents overlap the five-fold train/test pool")
    if len(train) < 5:
        raise ValueError("Insufficient documents for five folds")
    return {"train_pool_documents": len(train), "development_documents": len(dev),
            "train_features_sha256": sha256(cached_path(args, "train")),
            "dev_features_sha256": sha256(cached_path(args, "dev")),
            "initial_encoder_files_sha256": {
                name: sha256(Path(args.model_name) / name)
                for name in ("config.json", "vocab.txt", "tokenizer_config.json",
                             "pytorch_model.bin", "model.safetensors")
                if (Path(args.model_name) / name).is_file()}}


class FoldSaver:
    def __init__(self, output, args, hashes, audit):
        self.output, self.args, self.hashes, self.audit = Path(output), args, hashes, audit
        self.runs = []
        self.summaries = []

    def start(self, args, fold, train_idx, test_idx, features):
        folder = self.output / ("fold{}".format(fold + 1))
        folder.mkdir()
        train, test = [int(i) for i in train_idx], [int(i) for i in test_idx]
        if set(train) & set(test):
            raise ValueError("Train/test index overlap")
        write_json(folder / "train_indices.json", train)
        write_json(folder / "test_indices.json", test)
        write_json(folder / "test_documents.json", [
            {"feature_index": i, "topic_id": plain(features[i].topic_id),
             "doc_id": plain(features[i].doc_id)} for i in test])
        write_json(folder / "run.config.json", plain(vars(args)))

    def save(self, args, fold, epoch, model, valid, test):
        import torch
        folder = self.output / ("fold{}".format(fold + 1))
        # CPU copying and serialization consume no training RNG. No optimizer changes.
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        temp = folder / "best.pt.tmp"
        torch.save({"state_dict": state, "epoch": epoch, "fold": fold + 1,
                    "selection": "valid_f1 >= best_valid_f1; latest tie wins"}, str(temp))
        os.replace(str(temp), str(folder / "best.pt"))
        metadata = {"fold": fold + 1, "epoch": epoch, "validation": plain(valid),
                    "test_at_same_epoch": plain(test), "checkpoint": "best.pt",
                    "source_sha256": self.hashes}
        write_json(folder / "best.json.tmp", metadata)
        os.replace(str(folder / "best.json.tmp"), str(folder / "best.json"))

    def end(self, fold):
        folder = self.output / ("fold{}".format(fold + 1))
        if not (folder / "best.pt").is_file():
            raise RuntimeError("No finite best-dev checkpoint saved for fold {}".format(fold + 1))
        best = read_json(folder / "best.json")
        best["checkpoint_sha256"] = sha256(folder / "best.pt")
        write_json(folder / "best.json", best)
        self.summaries.append(best)
        prefix = folder.name + "/"
        self.runs.append({
            "run_id": folder.name, "checkpoint": prefix + "best.pt",
            "config": prefix + "run.config.json", "features": str(cached_path(self.args, "train")),
            "expected_features_sha256": self.audit["train_features_sha256"],
            "split": {"kind": "explicit", "training_indices": prefix + "train_indices.json",
                      "evaluation_indices": prefix + "test_indices.json"},
            "training_provenance": "Generated by train_and_export.py; see " + prefix +
            "best.json and ../training.log; fixed dev, latest epoch wins dev-F1 ties"})
        write_json(self.output / "manifest.json", {"expected_source_sha256": self.hashes, "runs": self.runs})
        write_json(self.output / "fold_summary.json", self.summaries)


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True, help="New output directory; never overwrite")
    parser.add_argument("--preflight", action="store_true", help="Audit inputs/environment without training")
    wrapper, original_args = parser.parse_known_args(argv)
    from parameter import parse_args
    saved = sys.argv
    try:
        sys.argv = [str(ROOT / "main.py")] + original_args
        args = parse_args()
    finally:
        sys.argv = saved
    return wrapper, args


def run(wrapper, args):
    validate_args(args)
    hashes = source_hashes()
    # Check structure before any expensive training starts (also in preflight).
    instrument_cross_validation((ROOT / "main.py").read_text(encoding="utf-8-sig"),
                                {}, None, None, None)
    audit = audit_data(args)
    import torch
    environment = {"python": sys.version, "torch": torch.__version__, "command": sys.argv,
                   "cuda_available": torch.cuda.is_available(),
                   "graph_dependencies": {m: importlib.util.find_spec(m) is not None
                                          for m in ("torch_geometric", "torch_sparse")}}
    if wrapper.preflight:
        print(json.dumps({"data": audit, "environment": environment,
                          "parameters": plain(vars(args)), "source_sha256": hashes},
                         ensure_ascii=False, indent=2))
        return
    if args.disable_cuda or not environment["cuda_available"]:
        raise RuntimeError("Original CGE uses .cuda(); run in the existing SAEF GPU environment")
    if not all(environment["graph_dependencies"].values()):
        raise RuntimeError("Use the original SAEF environment with torch_geometric and torch_sparse")
    output = Path(wrapper.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = "training"
    try:
        args.device = torch.device("cuda")
        args.log = str(output / "training.log")
        write_json(output / "run.config.json", plain(vars(args)))
        write_json(output / "input_audit.json", audit)
        write_json(output / "environment.json", environment)
        write_json(output / "source_sha256.json", hashes)
        for name in hashes:
            target = output / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(ROOT / name), str(target))
        shutil.copyfile(str(Path(__file__)), str(output / "sources/train_and_export.py"))
        shutil.copyfile(str(HERE / "export_cases.py"), str(output / "sources/export_cases.py"))
        import main as original
        torch.cuda.empty_cache()
        for name in logging.root.manager.loggerDict:
            if "transformers" in name:
                logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.basicConfig(format="%(message)s", level=logging.INFO, filename=args.log, filemode="w")
        original.setup_seed(args.seed)  # Once, exactly as original main; never reset per fold.
        _, model_class, tokenizer_class = original.MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(args.model_name, add_special_tokens=True)
        processor = original.ESC_processor(args, tokenizer, original.printlog)
        saver = FoldSaver(output, args, hashes, audit)
        training = instrument_cross_validation((ROOT / "main.py").read_text(encoding="utf-8-sig"),
                                               vars(original), saver.start, saver.save, saver.end)
        original.printlog("Training/evaluation parameters :{}".format(args))
        training(args, model_class, processor)
        if len(saver.runs) != 5:
            raise RuntimeError("Expected five completed folds before automatic export")
        if source_hashes() != hashes:
            raise RuntimeError("Original sources changed during training; refusing export")
        stage = "trajectory_export"
        write_json(output / "status.json", {"stage": stage, "training_complete": True})
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, str(HERE / "export_cases.py"), "export",
                        "--manifest", str(output / "manifest.json"),
                        "--output", str(output / "cases"), "--device", "cuda:0"],
                       cwd=str(ROOT), check=True)
        selection = read_json(output / "cases/selection.json")
        write_json(output / "status.json", {"stage": "complete", "training_complete": True,
                   "trajectory_export_complete": True, "case_selection": selection["status"],
                   "missing_roles": selection["missing_roles"],
                   "source_files_unchanged": source_hashes() == hashes})
        print("Completed: {} (case selection: {})".format(output, selection["status"]), flush=True)
    except Exception as exc:
        write_json(output / "status.json", {"stage": stage + "_failed", "error": str(exc),
                   "training_complete": stage == "trajectory_export",
                   "source_files_unchanged": source_hashes() == hashes})
        raise


if __name__ == "__main__":
    run(*parse_cli())
