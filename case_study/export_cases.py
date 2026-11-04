"""Read-only SAEF trajectory observation and deterministic case selection.

No training entry point is imported. Run --help for standalone commands.
"""
import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_FILES = ("model/model.py", "model/CGE.py", "model/__init__.py",
                "main.py", "parameter.py", "utils.py", "data/process_esc_csr.py")
REQUIRED_CONFIG = ("model_name", "mlp_size", "mlp_drop", "n_last", "rate", "w",
                   "max_iteration", "min_iteration", "threshold", "in_channels",
                   "metadata", "num_heads", "beta_intra", "beta_inter", "gamma",
                   "no_of_classes", "loss_type", "seed")


def sha256(path):
    result = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


class CachedFeature:
    """Attribute container for legacy ESC_features pickle locations."""


class FeatureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ESC_features" and module in (
                "__main__", "main", "data.process_esc_csr", "process_esc_csr"):
            return CachedFeature
        return super().find_class(module, name)


def load_features(path):
    # As with the original processor, only use your own trusted pickle files.
    with open(str(path), "rb") as stream:
        features = FeatureUnpickler(stream).load()
    if not features:
        raise ValueError("Empty features cache")
    return features


def plain(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def graph_snapshot(graph):
    return [{"edge_type": list(key), "edge_index": plain(value)}
            for key, value in graph.items()]


class TrajectoryObserver:
    """Observe existing methods; never substitute a prediction or adjacency."""
    def __init__(self, model):
        self.model = model
        self.rounds = []
        self.handles = []

    def __enter__(self):
        if self.model.training:
            raise ValueError("Trajectory export requires model.eval()")
        model = self.model
        self.saved = {}
        for name in ("get_graphedge_index", "Contrast_pre"):
            self.saved[name] = (name in model.__dict__, model.__dict__.get(name))
        self.graph_method = model.get_graphedge_index
        self.compare_method = model.Contrast_pre

        def graph_method(prediction, pairs, rel_type):
            import torch
            graph = self.graph_method(prediction, pairs, rel_type)
            probabilities = torch.softmax(prediction.detach(), dim=-1)
            self.rounds.append({
                "round": len(self.rounds),
                "logits": plain(prediction),
                "probabilities": plain(probabilities),
                "predicted_labels": plain(prediction.argmax(dim=-1)),
                "constructed_edges": graph_snapshot(graph),
                "cge_input_edges": [],
                "update": "projection_only_no_causal_edges",
                "stop_difference": None,
            })
            return graph

        def compare_method(prediction, previous):
            difference = self.compare_method(prediction, previous)
            self.rounds[-1]["stop_difference"] = plain(difference)
            return difference

        def cge_pre_hook(module, inputs):
            self.rounds[-1]["cge_input_edges"] = graph_snapshot(inputs[1])
            self.rounds[-1]["update"] = "CGE"

        model.get_graphedge_index = graph_method
        model.Contrast_pre = compare_method
        self.handles.append(model.CGE.register_forward_pre_hook(cge_pre_hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for name, (existed, value) in self.saved.items():
            if existed:
                setattr(self.model, name, value)
            else:
                delattr(self.model, name)


def trace_forward(model, inputs, verify=False):
    import torch
    with torch.no_grad():
        baseline = model(**inputs) if verify else None
        with TrajectoryObserver(model) as observer:
            loss, logits = model(**inputs)
    if not observer.rounds:
        raise RuntimeError("Original forward produced no prediction rounds")
    final_logits = torch.tensor(observer.rounds[-1]["logits"], dtype=logits.dtype,
                                device=logits.device)
    if not torch.equal(final_logits, logits):
        raise RuntimeError("Observed final logits differ from returned logits")
    if baseline is not None:
        if not (torch.equal(baseline[1], logits) and torch.equal(baseline[0], loss)):
            raise RuntimeError("Observed and unobserved forward differ; refusing export")
    for i, row in enumerate(observer.rounds):
        row["used_for_next_prediction"] = i + 1 < len(observer.rounds)
    return observer.rounds, float(loss.detach().cpu())


def validate_feature(feature):
    pairs = plain(feature.event_pairs)
    gold, types = plain(feature.target), plain(feature.rel_type)
    nodes = plain(feature.node_event)
    if not pairs or not (len(pairs) == len(gold) == len(types)):
        raise ValueError("Event-pair/target/rel_type lengths differ or are empty")
    if any(g not in (0, 1, 2) for g in gold) or any(t not in (0, 1) for t in types):
        raise ValueError("Expected labels 0/1/2 and relation types 0/1")
    if any(len(p) != 2 or any(v < 0 or v >= len(nodes) for v in p) for p in pairs):
        raise ValueError("Invalid event index")
    return pairs, gold, types, nodes


def feature_record(feature, index):
    pairs, gold, types, nodes = validate_feature(feature)
    return {
        "feature_index": index, "topic_id": plain(feature.topic_id),
        "doc_id": plain(feature.doc_id), "sentences": plain(feature.sentences),
        "enc_text": plain(feature.enc_text), "enc_tokens": plain(feature.enc_tokens),
        "nodes": [{"event_index": i, "sentence_index": int(n[0]),
                   "token_span": n[1]} for i, n in enumerate(nodes)],
        "event_pairs": pairs, "gold_labels": gold, "rel_type": types,
        "node_count": len(nodes),
        "sentence_count": len(set(int(n[0]) for n in nodes)),
        "gold_causal_count": sum(g != 0 for g in gold),
    }


def case_behaviour(record):
    gold, types = record["gold_labels"], record["rel_type"]
    first = record["rounds"][0]["predicted_labels"]
    last = record["rounds"][-1]["predicted_labels"]
    if not (len(gold) == len(types) == len(first) == len(last)):
        raise ValueError("Misaligned prediction arrays")
    cross, removed, direction, regress, errors = [], [], [], [], []
    corrected = []
    for i, (g, a, b, t) in enumerate(zip(gold, first, last, types)):
        if a != g and b == g:
            corrected.append(i)
            if a == 0 and g != 0 and t == 1:
                cross.append(i)
            if a != 0 and g == 0:
                removed.append(i)
            if a != 0 and g != 0:
                direction.append(i)
        if a == g and b != g:
            regress.append(i)
        if b != g:
            errors.append(i)
    initial_errors = sum(a != g for a, g in zip(first, gold))
    gain = initial_errors - len(errors)
    # A success must improve exact labels overall and introduce no new errors.
    success = len(record["rounds"]) >= 2 and gain > 0 and not regress
    return {"cross_sentence_recovered": cross, "false_edges_removed": removed,
            "directions_corrected": direction, "corrected_pairs": corrected,
            "regressed_pairs": regress, "final_error_pairs": errors,
            "initial_errors": initial_errors, "final_errors": len(errors),
            "net_error_reduction": gain, "success": success,
            "failure": len(record["rounds"]) >= 2 and bool(errors)}


def select_cases(records):
    if not records:
        raise ValueError("No trajectories")
    identities = [(str(r["topic_id"]), str(r["doc_id"])) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Repeated documents across runs; use one held-out prediction per document")
    medians = {key: statistics.median(r[key] for r in records)
               for key in ("node_count", "sentence_count", "gold_causal_count")}
    behaviours = [case_behaviour(r) for r in records]
    eligible = [i for i, r in enumerate(records) if r.get("held_out", False)]

    def rank(i):
        r = records[i]
        distance = sum(abs(r[key] - value) / max(value, 1)
                       for key, value in medians.items())
        return (distance, str(r["topic_id"]), str(r["doc_id"]))

    used, selections, missing = set(), [], []

    def choose(role, predicate, fallback=None):
        candidates = [i for i in eligible if i not in used and predicate(behaviours[i])]
        actual_role = role
        if not candidates and fallback:
            actual_role, predicate = fallback
            candidates = [i for i in eligible if i not in used and predicate(behaviours[i])]
        if not candidates:
            missing.append(role)
            return
        i = min(candidates, key=rank)
        used.add(i)
        selections.append({"role": actual_role, "record_index": i,
                           "run_id": records[i]["run_id"],
                           "doc_id": records[i]["doc_id"],
                           "topic_id": records[i]["topic_id"],
                           "behaviour": behaviours[i]})

    choose("success_cross_sentence", lambda b: b["success"] and b["cross_sentence_recovered"],
           ("success_other_correction", lambda b: b["success"]))
    choose("success_remove_or_reverse", lambda b: b["success"] and
           (b["false_edges_removed"] or b["directions_corrected"]),
           ("success_other_correction", lambda b: b["success"]))
    choose("failure", lambda b: b["failure"] and not b["success"])
    counts = {key: sum(bool(b[key]) for b in behaviours) for key in
              ("success", "failure", "cross_sentence_recovered", "false_edges_removed",
               "directions_corrected", "regressed_pairs")}
    return {"status": "complete" if not missing else "insufficient_candidates",
            "requested": {"success": 2, "failure": 1}, "missing_roles": missing,
            "selection_rule": "mechanism strata; nearest full-pool medians; topic/doc tie break",
            "population_medians": medians, "population_count": len(records),
            "eligible_count": len(eligible), "behaviour_counts_overlapping": counts,
            "selected": selections}


def heldout_indices(spec, features, base):
    kind = spec["kind"]
    if kind == "kfold":
        import numpy as np
        from sklearn.model_selection import KFold
        fold = spec["fold"]
        if not 1 <= fold <= spec["n_splits"]:
            raise ValueError("fold must be one-based and within n_splits")
        splits = KFold(n_splits=spec["n_splits"], shuffle=True, random_state=spec["seed"])
        train, test = list(splits.split(np.arange(len(features))))[fold - 1]
        train_docs = {(str(features[i].topic_id), str(features[i].doc_id)) for i in train}
        test_docs = {(str(features[i].topic_id), str(features[i].doc_id)) for i in test}
        if train_docs & test_docs:
            raise ValueError("KFold training/evaluation documents overlap")
        return test.tolist(), {"kind": kind, "spec": spec,
                              "training_index_count": len(train),
                              "mapping_evidence": "reconstructed; checkpoint-to-fold mapping user supplied"}
    if kind == "explicit":
        test_path = resolve(base, spec["evaluation_indices"])
        train_path = resolve(base, spec["training_indices"])
        test, train = read_json(test_path), read_json(train_path)
        for items in (test, train):
            if not isinstance(items, list) or any(type(i) is not int or i < 0 or i >= len(features) for i in items):
                raise ValueError("Indices must be integer arrays within the feature cache")
            if len(items) != len(set(items)):
                raise ValueError("Duplicate indices")
        if not test or not train or set(test) & set(train):
            raise ValueError("Empty or overlapping training/evaluation indices")
        train_docs = {(str(features[i].topic_id), str(features[i].doc_id)) for i in train}
        test_docs = {(str(features[i].topic_id), str(features[i].doc_id)) for i in test}
        if train_docs & test_docs:
            raise ValueError("Training/evaluation documents overlap")
        return test, {"kind": kind, "training_indices_sha256": sha256(train_path),
                      "evaluation_indices_sha256": sha256(test_path),
                      "mapping_evidence": "disjoint supplied indices; training history user supplied"}
    raise ValueError("split.kind must be kfold or explicit")


def load_model(checkpoint, config, device):
    import torch
    missing = sorted(set(REQUIRED_CONFIG) - set(config))
    if missing:
        raise ValueError("Missing original run config: " + ", ".join(missing))
    if config["min_iteration"] < 1 or config["max_iteration"] < 0:
        raise ValueError("Invalid iteration settings")
    if config["no_of_classes"] != 3:
        raise ValueError("Only the original three-label model is supported")
    config = dict(config)
    config["metadata"] = (config["metadata"][0], [tuple(t) for t in config["metadata"][1]])
    config["device"] = torch.device(device)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    state = torch.load(str(checkpoint), **kwargs)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected full SAEF state_dict")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if not all(any(k.startswith(prefix) for k in state) for prefix in
               ("pretrained_model.", "mlp.", "CGE.", "proj_head.")):
        raise ValueError("Not a full checkpoint for the current SAEF model; encoder-only weights rejected")
    from model import bertCSRModel
    model = bertCSRModel(SimpleNamespace(**config))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def export_manifest(manifest_path, output, device):
    import torch
    from utils import setup_seed
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    # Refuse to overwrite any prior evidence.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    records, runs = [], []
    seen_runs = set()
    try:
        if manifest.get("expected_source_sha256") and manifest["expected_source_sha256"] != before:
            raise ValueError("Source hashes differ from the training manifest")
        for run in manifest["runs"]:
            run_id = run["run_id"]
            if run_id in seen_runs:
                raise ValueError("Repeated run_id")
            seen_runs.add(run_id)
            if not run.get("training_provenance") or str(run["training_provenance"]).startswith("REPLACE"):
                raise ValueError("Supply training_provenance explaining checkpoint and held-out split")
            checkpoint = resolve(base, run["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError("Full trained SAEF checkpoint not found: " + str(checkpoint))
            cache = resolve(base, run["features"])
            if run.get("expected_features_sha256") and sha256(cache) != run["expected_features_sha256"]:
                raise ValueError("Feature cache changed since training")
            config_path = resolve(base, run["config"])
            config = read_json(config_path)
            config["model_name"] = str(resolve(config_path.parent, config["model_name"]))
            features = load_features(cache)
            indices, split_evidence = heldout_indices(run["split"], features, base)
            if run["split"]["kind"] == "kfold" and config["seed"] != run["split"]["seed"]:
                raise ValueError("Config seed differs from original KFold seed")
            setup_seed(config["seed"])
            model = load_model(checkpoint, config, device)
            run_meta = {"run_id": run_id, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "features": str(cache), "features_sha256": sha256(cache),
                        "config_sha256": sha256(config_path), "resolved_config": config,
                        "split_evidence": split_evidence, "evaluation_indices": indices,
                        "training_provenance": run["training_provenance"]}
            runs.append(run_meta)
            for position, index in enumerate(indices):
                f = features[index]
                record = feature_record(f, index)
                inputs = {"enc_input_ids": f.enc_input_ids.to(device),
                          "enc_mask_ids": f.enc_mask_ids.to(device), "node_event": f.node_event,
                          "t1_pos": [f.t1_pos], "t2_pos": [f.t2_pos], "target": [f.target],
                          "rel_type": [f.rel_type], "event_pairs": f.event_pairs}
                rounds, loss = trace_forward(model, inputs, verify=True)
                cap = min(config["max_iteration"] + 1, record["sentence_count"])
                record.update({"run_id": run_id, "held_out": True, "rounds": rounds,
                               "loss": loss, "effective_round_cap": cap,
                               "stopping_reason": "round_cap" if len(rounds) >= cap else "original_difference_rule",
                               "observation_parity": "exact_logits_and_loss_passed"})
                records.append(record)
                print("{}: {}/{} documents".format(run_id, position + 1, len(indices)), flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not records:
            raise ValueError("No evaluation documents")
        with (output / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        selection = save_selection(records, output)
        write_json(output / "provenance.json", {
            "status": "exported", "manifest_sha256": sha256(manifest_path),
            "exporter_sha256": sha256(Path(__file__)), "python_version": sys.version,
            "runs": runs, "source_sha256_before": before,
            "source_sha256_after": source_hashes(),
            "torch_version": torch.__version__, "device": device,
            "notes": ["Gold is used for original loss and post-hoc selection, not graph construction.",
                      "Last round CGE update executes but is not used for another prediction.",
                      "Stop difference is observed from original Contrast_pre, not recomputed."]})
        print(json.dumps({"status": selection["status"], "output": str(output)}, ensure_ascii=False))
    except Exception as exc:
        write_json(output / "FAILED.json", {"error": str(exc), "completed_documents": len(records),
                   "source_sha256_before": before, "source_sha256_after": source_hashes()})
        raise
    finally:
        if before != source_hashes():
            raise RuntimeError("Original source files changed during export")


def save_selection(records, output):
    selection = select_cases(records)
    write_json(output / "selection.json", selection)
    write_json(output / "selected_cases.json", {
        "status": selection["status"],
        "cases": [{"role": row["role"], "behaviour": row["behaviour"],
                   "trajectory": records[row["record_index"]]} for row in selection["selected"]]})
    with (output / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["run_id", "topic_id", "doc_id", "held_out", "node_count", "sentence_count",
                  "rounds", "initial_errors", "final_errors", "net_error_reduction", "success",
                  "failure", "cross_sentence_recovered", "false_edges_removed", "directions_corrected",
                  "regressed_pairs"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            behaviour = case_behaviour(record)
            row = {k: record[k] for k in fields[:6]}
            row["rounds"] = len(record["rounds"])
            for key in fields[7:]:
                value = behaviour[key]
                row[key] = len(value) if isinstance(value, list) else value
            writer.writerow(row)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("inspect", help="Inspect trusted cached features without loading SAEF")
    audit.add_argument("--features", required=True)
    export = sub.add_parser("export", help="Export held-out traces and select 2 successes + 1 failure")
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--device", default="cuda:0")
    select = sub.add_parser("select", help="Re-select from exported real trajectories")
    select.add_argument("--trajectories", required=True)
    select.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        features = load_features(args.features)
        records = [feature_record(f, i) for i, f in enumerate(features)]
        print(json.dumps({"documents": len(records), "features_sha256": sha256(args.features),
                          "duplicate_document_count": len(records) - len(set(
                              (str(r["topic_id"]), str(r["doc_id"])) for r in records)),
                          "median_events": statistics.median(r["node_count"] for r in records),
                          "median_sentences": statistics.median(r["sentence_count"] for r in records),
                          "labels": dict(Counter(g for r in records for g in r["gold_labels"])),
                          "source_sha256": source_hashes()}, ensure_ascii=False, indent=2))
    elif args.command == "export":
        export_manifest(args.manifest, args.output, args.device)
    else:
        with open(args.trajectories, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(save_selection(records, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
