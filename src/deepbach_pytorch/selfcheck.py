"""
selfcheck.py -- assert the invariants of this installation / checkpoint / cache.

Division of labour with `analysis.py`: `analysis` only measures (returns numbers),
`selfcheck` draws conclusions (`{'name', 'ok', 'detail'}`). Conventions: zero I/O on
import, all loading happens inside the functions that are called; data paths are
always arguments (defaulting to `db.`); every check returns its result instead of
passing it through printing (`log=` records it separately). `check_cache(ds)` can be
run interactively on its own, the rest are called in order.

Typical usage:

    log = db.selfcheck.CheckLog()
    before = db.selfcheck.cache_mtimes()
    ds = db.build_dataset()
    db.selfcheck.import_shadow_guard(log=log)
    db.selfcheck.cache_dir_guard(log=log)
    db.selfcheck.check_cache(ds, mtimes_before=before, log=log)
    db.selfcheck.check_weights_dir(models_dir, arch=arch, dataset=ds, log=log)
    db.selfcheck.check_shared_trunk(shared, base_model=base, dataset=ds, log=log)
    db.selfcheck.check_preprocessing_equivalence(base, shared, tc, tm, log=log)
    log.failed
"""

import os
import sys
import contextlib

import numpy as np
import torch
from torch import nn

import deepbach_pytorch as db
from .analysis import (CENTER, DENS_BACH, melody_preservation, pitch_index_sets,
                       score_stats, weight_dir_signature)


# ---------------------------------------------------------------------------
# Results and logging
# ---------------------------------------------------------------------------

def _result(name, ok, detail=""):
    """One check result; the three keys are fixed, the caller can dump it to json."""
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def _record(result, log):
    if log is not None:
        log.check(result["name"], result["ok"], result.get("detail", ""))
    return result


def _tail_ok(log, start):
    """Whether every check this call recorded passed; `log` may be shared, so only its own slice is looked at."""
    return all(entry["ok"] for entry in log.results[start:])


def _size(path):
    """File size in bytes; -1 when it does not exist."""
    return os.path.getsize(path) if os.path.exists(path) else -1


class CheckLog:
    """
    The ledger for one group of checks: pass/fail counts, and which ones failed.

    Uses its own seeded `np.random.default_rng(rng_seed)`, not a module global, so
    two `check_cache` calls on the same `CheckLog` draw the same windows.
    """

    def __init__(self, rng_seed=0, verbose=True):
        self.rng = np.random.default_rng(rng_seed)
        self.results = []
        self.verbose = verbose

    def check(self, name, ok, detail=""):
        entry = _result(name, ok, detail)
        self.results.append(entry)
        if self.verbose:
            print("  [{}] {}".format("OK  " if entry["ok"] else "FAIL", name)
                  + ("  --  {}".format(detail) if detail else ""), flush=True)
        return entry["ok"]

    def note(self, message):
        """Information that belongs to no check: printed only, not recorded."""
        if self.verbose:
            print("        {}".format(message), flush=True)

    @property
    def failed(self):
        return [entry for entry in self.results if not entry["ok"]]

    def summary(self):
        bad = self.failed
        return ("  {} checks, {} passed, {} failed".format(
            len(self.results), len(self.results) - len(bad), len(bad)))

    def to_json(self, path):
        import json
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, ensure_ascii=False, indent=2)
        return path


# ---------------------------------------------------------------------------
# Do the imports land inside the source tree
# ---------------------------------------------------------------------------

def import_shadow_guard(src_pkg=None, log=None):
    """Which tree is actually running: `deepbach_pytorch`'s, `DatasetManager`'s and
    `DeepBach`'s `__file__` must all be under `src_pkg` (mixed = new code, old conduct)."""
    src_pkg = os.path.abspath(src_pkg or db.PACKAGE_ROOT)
    got = os.path.abspath(db.__file__)
    problems = []
    if not got.startswith(src_pkg):
        problems.append("deepbach_pytorch -> {}".format(got))
    for module_name in ("DatasetManager", "DeepBach"):
        path = getattr(sys.modules.get(module_name), "__file__", None)
        if path is None:
            problems.append("{} not imported".format(module_name))
        elif not os.path.abspath(path).startswith(src_pkg):
            problems.append("{} -> {}".format(module_name, os.path.abspath(path)))

    return _record(_result(
        "imports resolve to the source tree",
        not problems,
        "package={}  src={}".format(got, src_pkg) if not problems
        else "; ".join(problems)), log)


def cache_dir_guard(src_pkg=None, log=None):
    """`DatasetManager`'s cache directory must be inside the source tree. What is
    checked is the path it computes itself, not `__file__`, so a mix is caught."""
    from DatasetManager.dataset_manager import DatasetManager

    src_pkg = os.path.abspath(src_pkg or db.PACKAGE_ROOT)
    want = os.path.abspath(os.path.join(src_pkg, "data", "dataset_cache"))
    got = os.path.abspath(DatasetManager().cache_dir)
    return _record(_result(
        "DatasetManager.cache_dir is inside the source tree",
        got == want,
        "{}".format(got) if got == want
        else "got {}  want {}".format(got, want)), log)


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def check_state_dict(model_object, path, label="", log=None):
    """
    Check whether the checkpoint and the architecture match **strictly**: same key
    set, same shape for every key.

    :param model_object: `nn.Module` (baseline's `voice_models[v]`, shared trunk's `shared.model`)
    :return: dict with `ok` / `detail` / `missing` / `extra` / `shape_bad` / `keys`
    """
    label = label or os.path.basename(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    current = model_object.state_dict()

    missing = [key for key in current if key not in state]
    extra = [key for key in state if key not in current]
    shape_bad = [key for key in state
                 if key in current
                 and tuple(state[key].shape) != tuple(current[key].shape)]
    n_keys = len(state)
    del state

    ok = not (missing or extra or shape_bad)
    result = _result(
        "{}: checkpoint matches the architecture".format(label), ok,
        "keys={} missing={} extra={} shape_bad={}".format(
            n_keys, len(missing), len(extra), len(shape_bad)))
    result.update({"missing": missing, "extra": extra,
                   "shape_bad": shape_bad, "keys": n_keys})
    if log is not None and shape_bad:
        log.note("shape mismatch: {}".format(shape_bad[:4]))
    return _record(result, log)


def check_weights_dir(models_dir, arch='baseline', dataset=None,
                      expect_files=None, role_conditioned=False,
                      lstm_hidden_size=None, log=None):
    """
    Whether the weight directory is usable: names match, exactly N files, all key
    shapes equal. Three layers:

    1. `db.check_pretrained_weights` -- are the files found by **name**;
    2. `analysis.weight_dir_signature` -- file count and md5 (the loader picks files
       by `os.listdir` order with `endswith`, so one extra file is a real failure);
    3. `check_state_dict` -- given a `dataset`, build the model and compare key by key.

    :param expect_files: 1 by default for the shared trunk, 4 for baseline
    :return: dict with `ok` / `weights` (layer 1) / `signature` (layer 2)
    """
    log = log or CheckLog()
    start = len(log.results)
    if expect_files is None:
        expect_files = 1 if arch == 'shared_trunk' else 4

    weights = db.check_pretrained_weights(
        models_dir=models_dir, arch=arch, role_conditioned=role_conditioned,
        lstm_hidden_size=lstm_hidden_size)
    # "No directory" and "directory present but files wrong" are two different
    # things: with no directory missing_files is empty and the other layers are moot.
    dir_exists = os.path.isdir(models_dir)
    log.check("{} weights present by name ({} required)".format(arch, expect_files),
              weights["complete"],
              "found {}/{} in {}".format(weights["found_weights"],
                                         weights["required_weights"],
                                         weights["models_dir"])
              if weights["complete"]
              else ("{} -- nothing to check here; Step 2's "
                    "check_pretrained_weights() reports the same fact as "
                    "complete=False".format(weights.get("error") or models_dir)
                    if not dir_exists
                    # print the suffix, not file: the latter is None exactly when missing.
                    else "missing tails: {}".format(
                        [e["suffix"] for e in weights["expected"]
                         if not e["found"]][:2])))
    if not weights["complete"]:
        log.note("present: {}".format(weights["present_files"]))
        log.note("dirs   : {}".format(weights["present_dirs"]))

    # Layer 2 is always computed (empty when there is no directory) but only asserted if it exists.
    signature = weight_dir_signature(weights["models_dir"],
                                     expect_files=expect_files)
    if dir_exists:
        log.check("exactly {} file(s) in the weight directory".format(expect_files),
                  signature["as_expected"],
                  "{}, dirs={}".format([f["name"] for f in signature["files"]],
                                       signature["dirs"]))
        for entry in signature["files"]:
            log.note("{}  {:>12,} B  md5 {}  {}".format(
                entry["name"], entry["bytes"], entry["md5"], entry["mtime"]))
    else:
        log.note("file count not checked: the directory does not exist")

    if dataset is not None and weights["complete"]:
        model = db.create_model(dataset=dataset, models_dir=weights["models_dir"],
                                arch=arch, role_conditioned=role_conditioned,
                                lstm_hidden_size=lstm_hidden_size)
        if arch == 'shared_trunk':
            check_state_dict(model.model,
                             os.path.join(weights["models_dir"],
                                          weights["expected"][0]["file"]),
                             label="shared trunk", log=log)
        else:
            for entry in weights["expected"]:
                check_state_dict(model.voice_models[entry["voice"]],
                                 os.path.join(weights["models_dir"],
                                              entry["file"]),
                                 label="baseline voice {}".format(entry["voice"]),
                                 log=log)
    elif dataset is None:
        log.note("no dataset given -> the key/shape diff was NOT verified")

    result = _result("weights directory usable", _tail_ok(log, start),
                     "{}: {}/{} files, complete={}".format(
                         weights["models_dir"], signature["n_files"],
                         expect_files, weights["complete"]))
    result.update({"weights": weights, "signature": signature})
    return result


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_paths(dataset=None, cache_dir=None):
    """
    The two absolute paths of the Bach cache: (dataset objects file, tensors file).

    The file name is `ChoraleDataset.__repr__()`, with no extension. You can pass a
    built `dataset`, or take the repr just for a shell key (`corpus_it_gen=None`: no
    corpus read, no tensors built).

    Goes through the `dataset.filepath` / `dataset.tensor_dataset_filepath`
    attributes rather than assembling paths here. Side effect: they make directories
    (idempotent).

    :return: `(objects_path, tensors_path)`
    """
    if dataset is None:
        dataset = db.ChoraleDataset(
            corpus_it_gen=None, name='bach_chorales', voice_ids=[0, 1, 2, 3],
            metadatas=[db.FermataMetadata(), db.TickMetadata(subdivision=4),
                       db.KeyMetadata()],
            sequences_size=8, subdivision=4,
            cache_dir=os.path.abspath(cache_dir or db.DATASET_CACHE_DIR))
    return dataset.filepath, dataset.tensor_dataset_filepath


def cache_mtimes(dataset=None, cache_dir=None):
    """The mtimes of the two cache files, the "loaded, not rebuilt" baseline. Take
    them **before** building the dataset, then pass them to
    `check_cache(mtimes_before=...)`."""
    return {path: (os.path.getmtime(path) if os.path.exists(path) else None)
            for path in cache_paths(dataset=dataset, cache_dir=cache_dir)}


def cache_file_sizes(dataset=None, cache_dir=None):
    """Byte size of the two cache files (-1 when absent). Used for printing only."""
    return {path: _size(path)
            for path in cache_paths(dataset=dataset, cache_dir=cache_dir)}


def _midi_of(token):
    """Vocabulary token string -> MIDI pitch; None when it is not a pitch."""
    import music21
    try:
        return int(music21.note.Note(token).pitch.midi)
    except Exception:
        return None


def check_cache(dataset, cache_dir=None, mtimes_before=None, sample=2000,
                log=None):
    """
    Whether what the cache decodes to is real music.

    :param dataset: an already loaded `ChoraleDataset`
    :param mtimes_before: `cache_mtimes(...)` taken before the dataset was built.
        When given, asserts the files were not rewritten; **when not given it says
        so**, it does not skip silently.
    :param sample: how many windows to check
    :return: dict with `ok` / `density` (per-voice note density %) / `n_windows`

    What is asserted: both cache files exist, `window_counts` sums to the tensor row
    count, the movement count lines up; the sampled indices are all inside their own
    voice's vocabulary; the metadata semantics (**boundary windows fill notes with
    START/END and metadata with 0, so a "real tick" is `key >= 1`**; density / phase
    / voice_id are computed on real ticks only); the decoded pitches fall inside the
    trained range and every voice has enough distinct pitches; density falls in
    15%..40%; one unpadded window round-trips through `tensor_to_score`.

    Sampling uses `log.rng`, so two calls on the same `CheckLog` draw the same windows.

    """
    log = log or CheckLog()
    start = len(log.results)
    rng = log.rng

    objects_path, tensors_path = cache_paths(dataset=dataset, cache_dir=cache_dir)
    log.note("cache key : {}".format(dataset.__repr__()))
    for path in (objects_path, tensors_path):
        log.note("{:>16}  {:>14,} B".format(
            os.path.basename(os.path.dirname(path)), _size(path)))
        log.note("                  {}".format(path))

    log.check("both cache files exist (datasets/ and tensor_datasets/)",
              _size(objects_path) > 0 and _size(tensors_path) > 0)

    if mtimes_before is not None:
        after = cache_mtimes(dataset=dataset, cache_dir=cache_dir)
        changed = [p for p, t in mtimes_before.items() if after.get(p) != t]
        log.check("loaded from cache, not rebuilt (mtimes unchanged)",
                  not changed, "changed: {}".format(changed) if changed else "")
    else:
        log.note("mtimes_before not given -> 'not rebuilt' was NOT verified")

    log.check("dataset repr matches the cache file name",
              dataset.__repr__() == os.path.basename(objects_path),
              dataset.__repr__())
    log.note("num_voices={} subdivision={} sequences_size={}".format(
        dataset.num_voices, dataset.subdivision, dataset.sequences_size))
    log.note("metadatas={}".format([m.name for m in dataset.metadatas]))
    log.note("voice_ranges={}".format(dataset.voice_ranges))
    log.note("vocab sizes={}".format([len(d) for d in dataset.note2index_dicts]))

    tensors = dataset.tensor_dataset
    n_windows = len(tensors)
    log.check("tensor dataset loaded", n_windows > 0,
              "{:,} windows".format(n_windows))
    log.note("tensor shapes: notes={} metas={} dtype={}".format(
        tuple(tensors.tensors[0].shape), tuple(tensors.tensors[1].shape),
        tensors.tensors[0].dtype))

    counts = list(dataset.window_counts or [])
    layout = list(dataset.corpus_layout or [])
    log.check("window_counts sums to the tensor row count",
              sum(counts) == n_windows,
              "sum={:,} rows={:,}".format(sum(counts), n_windows))
    log.check("movement count matches window_counts length",
              len(layout) == len(counts) and len(layout) > 0,
              "{} movements (= Bach chorale corpus)".format(len(layout)))
    lo_i, hi_i = int(0.85 * n_windows), int(0.95 * n_windows)
    log.note("split 85/10/5 -> train={:,} val={:,} eval={:,}".format(
        lo_i, hi_i - lo_i, n_windows - hi_i))

    # --- content: does it decode to real music -------------------------------
    picked = np.sort(rng.choice(n_windows, size=min(sample, n_windows),
                                replace=False))
    notes = tensors.tensors[0][torch.from_numpy(picked)].long()
    metas = tensors.tensors[1][torch.from_numpy(picked)].long()

    log.check("every index is inside its voice's vocabulary",
              all(int(notes[:, v].max()) < len(dataset.note2index_dicts[v])
                  for v in range(4)),
              "max={}".format([int(notes[:, v].max()) for v in range(4)]))

    # The four metadata channels. Boundary windows fill notes with START/END and
    # metadata with 0, so the key channel singles out the real ticks.
    fermata, tick, key, voice_id = (metas[..., 0], metas[..., 1],
                                    metas[..., 2], metas[..., 3])
    real = key >= 1
    n_pad = int((~real).sum())
    log.note("zero-padded ticks (window out of bounds): {:,} / {:,} = {:.2f}%"
             .format(n_pad, real.numel(), 100.0 * n_pad / real.numel()))

    log.check("fermata channel is 0/1 only",
              bool(((fermata == 0) | (fermata == 1)).all()),
              "fraction of 1s on real ticks = {:.4f}".format(
                  float(fermata[real].float().mean())))
    log.check("key channel stays within 0..15",
              bool((key >= 0).all() and (key <= 15).all()),
              "distinct non-zero values = {}".format(
                  sorted(set(key[real].flatten().tolist()))[:8]))

    # The real ticks are one contiguous run, over which the tick channel cycles 0..3
    # (the phase comes from the window start).
    real_np, tick_np = real.numpy(), tick.numpy()
    violations = []
    for window in range(real_np.shape[0]):
        for voice in range(real_np.shape[1]):
            positions = np.nonzero(real_np[window, voice])[0]
            if positions.size == 0:
                continue
            if positions.size != positions[-1] - positions[0] + 1:
                violations.append(("gap", window, voice))
                continue
            expected = (np.arange(positions.size)
                        + tick_np[window, voice, positions[0]]) % 4
            if not np.array_equal(tick_np[window, voice, positions], expected):
                violations.append(("phase", window, voice))
    log.check("tick channel cycles 0..3 over a contiguous run of real ticks",
              not violations,
              "{} violations".format(len(violations))
              + ("  e.g. {}".format(violations[:3]) if violations else ""))
    log.check("voice_id channel equals the voice index on real ticks",
              bool(all((voice_id[:, v][real[:, v]] == v).all()
                       for v in range(4))))

    indexes = pitch_index_sets(dataset)
    is_note = torch.zeros_like(notes, dtype=torch.bool)
    for voice in range(4):
        mask = torch.zeros(len(dataset.note2index_dicts[voice]), dtype=torch.bool)
        mask[list(indexes[voice])] = True
        is_note[:, voice] = mask[notes[:, voice]]

    in_range = True
    range_detail = []
    density = []
    for voice in range(4):
        low, high = dataset.voice_ranges[voice]
        reverse = {i: token for i, token in dataset.index2note_dicts[voice].items()}
        midis = [_midi_of(reverse[i])
                 for i in torch.unique(notes[:, voice][is_note[:, voice]]).tolist()]
        bad = [m for m in midis if m is None or m < low or m > high]
        in_range = in_range and not bad
        range_detail.append("v{} {}..{}/{} pitches".format(
            voice, min(midis), max(midis), len(midis)))
        # density = fraction of the real ticks that are notes.
        density.append(float(is_note[:, voice][real[:, voice]].float().mean()) * 100)

    log.check("decoded pitches all fall inside the voice's trained range",
              in_range, "; ".join(range_detail))
    log.check("every voice has many distinct pitches (not a constant)",
              all(len(indexes[v]) > 10 for v in range(4)),
              "distinct pitch tokens={}".format([len(p) for p in indexes]))

    dens = [round(d, 1) for d in density]
    log.note("real Bach note density (onset ticks, %): {}".format(dens))
    log.check("real Bach density is in a sane band (15%..40%)",
              all(15 <= d <= 40 for d in density))

    # Round-trip one unpadded window.
    unpadded = np.nonzero((key.numpy() >= 1).all(axis=(1, 2)))[0]
    if unpadded.size:
        w = int(unpadded[len(unpadded) // 2])
        log.note("decoding sampled window #{} (no padding)".format(w))
        score = dataset.tensor_to_score(tensor_score=notes[w],
                                        fermata_tensor=metas[w, :, :, 0])
        soprano = [n.nameWithOctave for n in score.parts[0].flatten().notes][:12]
        log.check("a window round-trips through tensor_to_score",
                  len(list(score.parts[0].flatten().notes)) > 0,
                  "soprano: " + " ".join(soprano))
    else:
        log.check("a window round-trips through tensor_to_score", False,
                  "no unpadded window among the {} sampled".format(len(picked)))

    result = _result("cache is self-consistent and decodes to real music",
                     _tail_ok(log, start), "density={}".format(dens))
    result.update({"density": dens, "n_windows": int(n_windows),
                   "sampled": int(len(picked))})
    return result


# ---------------------------------------------------------------------------
# Shared trunk (option A)
# ---------------------------------------------------------------------------

def _nn_modules(model):
    """The real `nn.Module`s inside a wrapper object (the trunk / the voice models)."""
    if model is None:
        return
    inner = getattr(model, 'shared_model', None)
    if inner is not None:
        if isinstance(inner, nn.Module):
            yield inner
        return
    for voice_model in (getattr(model, 'voice_models', None) or []):
        if isinstance(voice_model, nn.Module):
            yield voice_model


@contextlib.contextmanager
def _eval_mode(*models):
    """
    Switch the models to eval for the duration, restore on exit.

    This is the premise of the equivalence assertions below: in training mode dropout
    redraws its mask on every forward pass, so the equivalence must fail. The restore
    uses `module.train(was_training)`, not an unconditional `train()`.
    """
    saved = []
    for model in models:
        for module in _nn_modules(model):
            saved.append((module, module.training))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in saved:
            module.train(was_training)


def _forward_all_vs_role(model, tensor_chorale, tensor_metadata, center, role):
    """
    Run hand-made tensors through `forward_all`, taking one role only -- what
    `analysis.eval_shared` does for that role. The metadata uses `tm[:, 0]` (voice
    0's slice), a shortcut `check_preprocessing_equivalence` asserts.
    """
    from .DeepBach.data_utils import reverse_tensor

    trunk_metas = model.model.trunk_metas
    left = tensor_chorale[:, :, :center]
    right = reverse_tensor(tensor_chorale[:, :, center + 1:], dim=2)
    left_metas = tensor_metadata[:, 0, :center, :trunk_metas]
    right_metas = reverse_tensor(
        tensor_metadata[:, 0, center + 1:, :trunk_metas], dim=1)
    center_metas = tensor_metadata[:, 0, center, :trunk_metas]
    with _eval_mode(model), torch.no_grad():
        return model.model.forward_all(
            left, tensor_chorale[:, :, center], right,
            (left_metas, center_metas, right_metas))[role]


def check_shared_trunk(model, base_model=None, dataset=None, batch=64,
                       center=CENTER, device=None, log=None):
    """
    Whether the shared trunk really is "one trunk + four heads". Four things can
    each break on its own:

    * the parameters really are shared: the four `RoleView`s' `_model` is the same
      object, and the union of their `parameters()` equals the trunk's `{id(p)}` set;
    * one trunk forward == four per-role forwards (elementwise);
    * the four heads' output dimensions == the four voice vocab sizes in the cache;
    * with `base_model` given, print both parameter counts (recurrent part listed apart).

    :param model: `SharedTrunkDeepBach`
    :param base_model: optional `DeepBach`, used only for the parameter count compare
    :return: dict with `ok` / `n_params` / `n_params_baseline` /
        `forward_all_vs_role_max_diff`
    """
    log = log or CheckLog()
    start = len(log.results)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    shared_parameters = list(model.model.parameters())
    n_params = sum(p.numel() for p in shared_parameters)
    ids = {id(p) for view in model.voice_models for p in view.parameters()}
    log.check("the four RoleViews share one set of parameters",
              all(view._model is model.model for view in model.voice_models)
              and ids == {id(p) for p in shared_parameters},
              "{} tensors / {:.2f} M params, shared by four roles".format(
                  len(shared_parameters), n_params / 1e6))

    n_params_base = None
    if base_model is not None:
        n_params_base = sum(p.numel() for m in base_model.voice_models
                            for p in m.parameters())
        recurrent_shared = sum(p.numel() for k, p in model.model.named_parameters()
                               if k.startswith("lstm_"))
        recurrent_base = sum(p.numel() for m in base_model.voice_models
                             for k, p in m.named_parameters()
                             if k.startswith("lstm_"))
        log.note("params: shared {:.2f} M (recurrent {:.2f} M)  vs  baseline "
                 "{:.2f} M (recurrent {:.2f} M)".format(
                     n_params / 1e6, recurrent_shared / 1e6,
                     n_params_base / 1e6, recurrent_base / 1e6))

    max_diff = None
    if dataset is not None:
        head_vocab = [head[-1].out_features
                      for head in model.model.mlp_predictions]
        want = [len(d) for d in dataset.note2index_dicts]
        log.check("the four heads match the four voice vocabularies",
                  head_vocab == want, "{} vs {}".format(head_vocab, want))

        tensors = dataset.tensor_dataset
        n = min(batch, len(tensors))
        tensor_chorale = tensors.tensors[0][:n].long().to(device)
        tensor_metadata = tensors.tensors[1][:n].long().to(device)
        max_diff = _forward_all_vs_role_max_diff(
            model, tensor_chorale, tensor_metadata, center)
        log.check("one trunk pass == four per-role passes (elementwise)",
                  max_diff == 0.0, "max|diff| = {}".format(max_diff))
    else:
        log.note("no dataset given -> head vocabularies and the forward "
                 "equivalence were NOT verified")

    result = _result("shared trunk is one trunk with four roles",
                     _tail_ok(log, start), "n_params={}".format(n_params))
    result.update({"n_params": int(n_params),
                   "n_params_baseline": (int(n_params_base)
                                         if n_params_base is not None else None),
                   "forward_all_vs_role_max_diff": max_diff})
    return result


def _forward_all_vs_role_max_diff(model, tensor_chorale, tensor_metadata, center):
    """The largest elementwise difference between `forward_all(...)[r]` and `forward_role(r, ...)`."""
    from .DeepBach.data_utils import mask_entry, reverse_tensor

    left = tensor_chorale[:, :, :center]
    right = reverse_tensor(tensor_chorale[:, :, center + 1:], dim=2)
    current = tensor_chorale[:, :, center]
    trunk_metas = model.model.trunk_metas
    left_metas = tensor_metadata[:, 0, :center, :trunk_metas]
    right_metas = reverse_tensor(
        tensor_metadata[:, 0, center + 1:, :trunk_metas], dim=1)
    center_metas = tensor_metadata[:, 0, center, :trunk_metas]

    with _eval_mode(model), torch.no_grad():
        joint = model.model.forward_all(left, current, right,
                                        (left_metas, center_metas, right_metas))
        per_role = [model.model.forward_role(
            r, (left, mask_entry(current, r, dim=1), right),
            (left_metas, center_metas, right_metas))
            for r in range(len(model.voice_models))]
    return max(float((joint[r] - per_role[r]).abs().max())
               for r in range(len(per_role)))


def check_preprocessing_equivalence(base_model, shared_model, tensor_chorale,
                                    tensor_metadata, center=CENTER, log=None):
    """
    Whether the two preprocessing paths agree on **the same batch of real inputs**.
    `analysis.eval_shared` runs hand-made left/right tensors through `forward_all`,
    bypassing `RoleView`; this asserts the three things needed to license that:

    * **the conditioning metadata channels are identical across voices**: only the
      first `len(metadatas)` channels are compared (fermata / tick / key), as
      `voice_id` differing per voice is by design. With `role_conditioned=True` the
      trunk reads `voice_id`, so that shortcut no longer holds and this says so
      instead of pretending equivalence.
    * **`RoleView.forward` == `forward_all(...)[r]`** elementwise, stronger than
      `check_shared_trunk`'s version: the inputs come from `RoleView`'s own `preprocess_*`.
    * baseline's `preprocess_*` == `RoleView`'s, i.e. the "same inputs" premise.

    :return: dict with `ok` / `notes_max_diff` / `metas_max_diff` /
        `forward_max_diff` / `trunk_metas`
    """
    log = log or CheckLog()
    start = len(log.results)
    n_voices = len(shared_model.voice_models)
    trunk_metas = shared_model.model.trunk_metas
    role_conditioned = bool(shared_model.model.role_conditioned)

    # --- Premise A: the **conditioning** metadata channels agree across voices ---
    # Only the voice-independent channels are compared (fermata / tick / key), not
    # the whole `:trunk_metas`: voice_id is the last channel and differing per voice
    # is by design. This identity is the premise of `analysis.eval_shared`'s
    # `tm[:, 0]` shortcut.
    n_conditioning = trunk_metas - int(role_conditioned)
    reference = tensor_metadata[:, 0, :, :n_conditioning]
    mismatched = [r for r in range(1, n_voices)
                  if not torch.equal(reference,
                                     tensor_metadata[:, r, :, :n_conditioning])]
    log.check("conditioning metadata channels are identical across voices",
              not mismatched,
              "{} voice-independent channel(s) compared".format(n_conditioning)
              if not mismatched else "differ for voices {}".format(mismatched))

    if role_conditioned:
        # Compared on real ticks only: boundary windows fill metadata with 0, and a
        # padding slot's voice_id is 0 rather than r. The key channel does the
        # masking (`KeyMetadata.get_index == sharps + 8 >= 1`, padding is always 0).
        # `dataset.metadatas` holds objects not names, each carrying its own `.name`.
        names = [getattr(m, 'name', None)
                 for m in (getattr(shared_model.model.dataset, 'metadatas', []) or [])]
        if 'key' in names:
            real = tensor_metadata[:, :, :, names.index('key')] >= 1
            mask_detail = "masked by the 'key' channel (key >= 1)"
        else:
            # Fallback mask: a padding tick has all four channels at 0 and a real
            # tick has key >= 1, so "not all-zero" is equivalent. A weaker mask plus
            # a note beats skipping silently.
            real = (tensor_metadata != 0).any(dim=-1)
            mask_detail = ("masked by 'not all-zero' (no named 'key' channel in "
                           "{}); the assertion still ran".format(names))
        per_voice = [
            bool((tensor_metadata[:, r, :, n_conditioning][real[:, r]] == r).all())
            for r in range(n_voices)]
        log.check("voice_id channel equals the voice index on real ticks "
                  "(role_conditioned)", all(per_voice),
                  "{} of {} voice(s) match, {}; voice_id is voice-dependent by "
                  "design -> the eval_shared shortcut (tm[:, 0] for all roles) "
                  "is licensed only when role_conditioned=False"
                  .format(sum(per_voice), n_voices, mask_detail))
    else:
        log.note("role_conditioned=False: voice_id is channel {} and is not "
                 "consumed by the trunk".format(n_conditioning))

    # --- preprocessing: VoiceModel vs RoleView --------------------------------
    # The whole block runs under eval: `preprocess_*` has no dropout, but the
    # `RoleView.forward` / `forward_all` below do, and the two do not share a mask.
    notes_diff = 0.0
    metas_diff = 0.0
    forward_diff = 0.0
    with _eval_mode(base_model, shared_model):
        for r in range(n_voices):
            voice_model = base_model.voice_models[r]
            role_view = shared_model.voice_models[r]

            base_notes, base_label = voice_model.preprocess_notes(
                tensor_chorale=tensor_chorale, time_index_ticks=center)
            role_notes, role_label = role_view.preprocess_notes(
                tensor_chorale=tensor_chorale, time_index_ticks=center)
            for a, b in zip(base_notes, role_notes):
                if a is None or b is None:
                    if a is not b:
                        notes_diff = float('inf')
                else:
                    notes_diff = max(notes_diff, float((a - b).abs().max()))
            if not torch.equal(base_label, role_label):
                notes_diff = float('inf')

            base_metas = voice_model.preprocess_metas(
                tensor_metadata=tensor_metadata, time_index_ticks=center)
            role_metas = role_view.preprocess_metas(
                tensor_metadata=tensor_metadata, time_index_ticks=center)
            for a, b in zip(base_metas, role_metas):
                width = min(a.shape[-1], b.shape[-1])
                metas_diff = max(metas_diff,
                                 float((a[..., :width] - b[..., :width]).abs().max()))

        log.check("VoiceModel.preprocess_* == RoleView.preprocess_*",
                  notes_diff == 0.0 and metas_diff == 0.0,
                  "notes max|diff|={}  metas max|diff|={}".format(notes_diff,
                                                                  metas_diff))

        # --- RoleView.forward vs the hand-made forward_all ---------------------
        for r in range(n_voices):
            role_view = shared_model.voice_models[r]
            notes, _ = role_view.preprocess_notes(tensor_chorale=tensor_chorale,
                                                  time_index_ticks=center)
            metas = role_view.preprocess_metas(tensor_metadata=tensor_metadata,
                                               time_index_ticks=center)
            with torch.no_grad():
                via_view = role_view.forward(notes, metas)
                via_joint = _forward_all_vs_role(shared_model, tensor_chorale,
                                                 tensor_metadata, center, r)
            forward_diff = max(forward_diff,
                               float((via_view - via_joint).abs().max()))

        log.check("RoleView.forward == forward_all(...)[r] on real inputs",
                  forward_diff == 0.0, "max|diff| = {}".format(forward_diff))

    result = _result("both preprocessing paths agree on real windows",
                     _tail_ok(log, start),
                     "role_conditioned={} notes={} metas={} forward={}".format(
                         role_conditioned, notes_diff, metas_diff, forward_diff))
    result.update({"role_conditioned": role_conditioned,
                   "notes_max_diff": notes_diff, "metas_max_diff": metas_diff,
                   "forward_max_diff": forward_diff,
                   "trunk_metas": int(trunk_metas)})
    return result


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------

def check_generation(output, melody_path=None, dataset=None, dens_bach=None,
                     log=None):
    """
    Whether a generated score has the properties it should.

    :param output: path of the generated musicxml
    :param melody_path: the input melody. When given, asserts voice 0 matches
        **pitch by pitch + position by position**.
    :param dataset: required; `score_stats` needs its `voice_ranges` and `subdivision`
    :param dens_bach: real Bach density reference, defaults to `analysis.DENS_BACH`
    :return: dict with `ok` / `stats` / `melody`
    """
    log = log or CheckLog()
    start = len(log.results)
    if dataset is None:
        raise ValueError("check_generation needs the dataset (voice ranges and "
                         "subdivision come from it)")

    score, stats = score_stats(output, dataset)
    generated = stats[1:] if len(stats) > 1 else stats

    log.check("the generated voices are neither empty nor constant",
              all(s["notes"] > 20 and s["distinct"] > 3 for s in generated),
              "notes={} distinct={}".format([s["notes"] for s in generated],
                                            [s["distinct"] for s in generated]))
    log.check("generated pitches stay inside the trained ranges",
              all(s["in_trained_range_pct"] >= 90 for s in generated),
              "{}%".format([s["in_trained_range_pct"] for s in generated]))

    melody = None
    if melody_path is not None:
        import music21
        melody = melody_preservation(score, music21.converter.parse(melody_path))
        log.check("output voice 0 keeps the input melody (pitch + offset)",
                  melody["exact"],
                  "{} of {} notes matched (input has {})".format(
                      melody["n_matched"], melody["n_output"], melody["n_input"]))

    reference = list(dens_bach) if dens_bach is not None else list(DENS_BACH)
    density = [s["density_pct"] for s in stats]
    log.note("density % {}   real Bach {}".format(density, reference))
    log.note("deviation from real Bach (positive = denser): {}".format(
        [round(density[v] - reference[v], 1)
         for v in range(1, min(len(density), len(reference)))]))

    result = _result("generated score passes the musical sanity checks",
                     _tail_ok(log, start),
                     "notes={}".format([s["notes"] for s in stats]))
    result.update({"stats": stats, "melody": melody, "density": density})
    return result
