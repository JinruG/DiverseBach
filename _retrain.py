"""
_retrain.py — rebuild one tree's Bach corpus cache, then train its arms.

Two trees, each with its own batch size:

    data256   batch 256      data128   batch 128

Each tree is a complete, independent install of the data layer: its own cache, its own
21 snapshot directories per arm, sharing nothing.
The batch size is **derived from the tree name**, not a separate parameter.

Cache building and the training routes are split into stages, **one process per stage**:
on this machine cuDNN's LSTM still crashes at process exit after a successful run and
destroys the exit code, and two GPU processes corrupt each other's device, so the stages
must be **serial**, never overlapping.

Usage:

    python _retrain.py --tree data256 --stage cache        # rebuild this tree's cache
    python _retrain.py --tree data256 --stage baseline     # four voices
    python _retrain.py --tree data256 --stage shared       # shared trunk, rc=False
    python _retrain.py --tree data256 --stage shared_rc    # shared trunk, rc=True
    python _retrain.py --tree data256 --stage voice0       # single-voice parity
    python _retrain.py --tree data256 --stage gru          # baseline, nn.GRU

Every training stage writes **21 checkpoint sets** (one per pass + best), in sibling
directories under the tree (see DatasetManager.helpers):

    <tree>/models_epoch01/<arch>/   end of pass 1
    ...
    <tree>/models_epoch20/<arch>/   end of pass 20 (the set loaded by default)
    <tree>/models_best/<arch>/      refreshed when the validation loss improves

Keeping every pass rather than sampling a few points is because the true minimum falls in
the middle.

`--stage voice0` uses the custom directory `<tree>/models_voice0`, so its sibling
directories carry a suffix (`models_voice0__best` / `models_voice0__epoch01` / ...,
`models_voice0` itself is the default snapshot) — see `helpers.sibling_snapshot_dir`.
Its cache is single-voice, with far more windows than four voices, so the step counts are
not comparable.

`shared_rc` uses `<tree>/models_rc/shared_trunk`, matching infer.py's hardcoded
convention.  It must never share a directory with `shared`: the two checkpoint sets differ
only in the last field `,True)` / `,False)`, and `_load_pretrained_weights` picks files by
`endswith` + listdir order.

`gru` is the `baseline` architecture with `nn.LSTM` swapped for `nn.GRU`, everything else
unchanged, directory `<tree>/models_<snapshot>/gru/` — a leaf **inside** the standard
21-set ladder, so it has best/epoch01..20 as well, and `sibling_snapshot_dir` can find
`models_best/gru` by segment replacement.  Its checkpoint filenames carry a trailing
`,gru` (`...256,gru)` vs `...256)`), which is the first line of defence; the separate leaf
directory is the second.

Default behaviour: an existing directory is always **renamed to *_old to make way, never
deleted**.  Each stage writes the elapsed time, the window count, the checkpoint sizes and
mtimes into `_retrain/<tree>_<stage>.json`, and stashes a second copy of the per-pass loss
curves from `<models_dir>/_history/` — `_stash()` renames the snapshot directory
wholesale, so a rerun may have moved the curves away.
"""

import argparse
import json
import os
import sys
import time

REPO    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.path.join(REPO, "src")
SRC_PKG = os.path.join(SRC, "deepbach_pytorch")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Explicitly require UTF-8 output: with stdout redirected to a file Python takes the
# encoding from the locale (cp936 here) and the log becomes unreadable bytes;
# errors='replace' keeps a stray character from blowing up the log.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# tree name -> batch size.  The tree name is the only definition of the batch size.
TREES = {"data256": 256, "data128": 128}

# The budget is counted in steps.  Both None means the reference budget
# (DEFAULT_TOTAL_PASSES full passes); the true step count is computed by
# resolve_training_budget from the current cache's row count and printed.  Never hardcode
# a concrete step count.
STEPS_PER_EPOCH = None
MAX_STEPS       = None
OUT_DIR         = os.path.join(REPO, "_retrain")

# Measured window count of the current corpus (361 rows, 2026-09-21).  **A hint, not an
# assertion**: the number follows the data layer, so a rebuild that disagrees means the
# corpus changed.  The batch size does not affect windowing, so both trees should give the
# same number.  It has changed with the filter rules many times, so it is a dated
# measurement rather than a constant.
PREV_WINDOWS = 1_102_258


def _rep(path):
    try:
        return os.path.relpath(os.path.abspath(path), REPO).replace("\\", "/")
    except ValueError:
        # No relative path across drives; show the whole thing.
        return os.path.abspath(path).replace("\\", "/")


def _stash(path):
    """Rename an existing directory out of the way (same-volume rename; 2.6 GB is seconds)."""
    if not os.path.isdir(path):
        return None
    dst = path + "_old"
    if os.path.exists(dst):
        dst = "{}_{}".format(path, time.strftime("old-%Y%m%d-%H%M%S"))
    os.rename(path, dst)
    print("  [让位] {}  ->  {}".format(_rep(path), _rep(dst)), flush=True)
    return dst


def _import_pkg():
    import deepbach_pytorch as db
    got = os.path.abspath(db.__file__)
    want = os.path.abspath(SRC_PKG)
    # Assert this is the source tree, not the site-packages copy with no arch parameter.
    assert got.startswith(want), (got, want)
    print("  package: {}".format(_rep(got)), flush=True)
    return db


def _install_tree(db, tree):
    """
    Point this process at one tree: `SRC_PKG/<tree>/dataset_cache`.

    Both names must be rebound: `db.DatasetManager` (the built-in Bach corpus path, which
    is what `_get_default_dataset` calls) and `db.DATASET_CACHE_DIR` (the module global
    that the same function's custom-corpus branch reads).

    Not `db.MODELS_DIR` and friends: those constants are computed at **import** from
    `helpers.DATA_DIR`, so `db.snapshot_dir` still answers `data/` and this file's
    `models_dir` is always assembled explicitly from the tree root.  The 21 sibling
    directories then follow from that path by pure arithmetic.

    Side effect: the base `__init__` runs first and creates an empty
    `<pkg>/data/dataset_cache/`, which is harmless.

    :return: (tree_root, cache_dir)
    """
    root = os.path.join(SRC_PKG, tree)
    cache_dir = os.path.join(root, "dataset_cache")

    base = db.DatasetManager

    class _TreeDatasetManager(base):
        def __init__(self):
            super().__init__()
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)

    db.DatasetManager = _TreeDatasetManager
    db.DATASET_CACHE_DIR = cache_dir
    return root, cache_dir


def _check_cache_dir(db, tree, cache_dir):
    """
    Assert from both directions that the bound cache directory belongs to this tree.

    Instantiate through **`db.DatasetManager`**, not
    `from DatasetManager.dataset_manager import DatasetManager` — the latter gets the
    original class from before the rebinding and reports the old path whatever
    `_install_tree` did.  The second assertion checks the path's shape, so a mistyped tree
    name is not taken as a successful build.

    :return: the actual cache_dir absolute path
    """
    got = os.path.abspath(db.DatasetManager().cache_dir)
    want = os.path.abspath(cache_dir)
    assert got == want, "DatasetManager.cache_dir = {}\n  expected {}".format(got, want)
    assert os.path.abspath(cache_dir) == os.path.join(
        os.path.abspath(SRC_PKG), tree, "dataset_cache"), cache_dir
    return got


def _models_dir(tree, stage):
    """
    One arm's weight directory, as an explicit path under the tree.

    Does not go through `db.snapshot_dir` (it is not per-tree, it always answers `data/`;
    see `_install_tree`).  The leaf name still comes from the package's own constants, so
    the naming cannot drift from the loader.

    Indexed by **stage**, not by arch: `voice0` is a baseline model and `shared_rc` is a
    shared_trunk model, each needing its own directory, and neither name exists in
    `db.ARCHS` (passing it as `arch=` raises ValueError outright).  The stage decides the
    directory, the arch is a property of the model built in it.
    """
    db = sys.modules["deepbach_pytorch"]
    if stage == "voice0":
        return os.path.join(SRC_PKG, tree, "models_voice0")
    if stage == "shared_rc":
        return os.path.join(SRC_PKG, tree, "models_rc", "shared_trunk")
    if stage == "gru":
        # A sibling of `baseline` inside the standard ladder, not a new ladder: the leaf
        # name comes from the stage, the snapshot segment from the package's own
        # `models_<DEFAULT_SNAPSHOT>`, so `sibling_snapshot_dir` finds `models_best/gru`
        # by segment replacement.  The separate leaf directory is required: the GRU and
        # LSTM reprs differ only in a trailing `,gru`, and a shared directory would let
        # the loader's `endswith` select by `os.listdir` order.
        return os.path.join(SRC_PKG, tree,
                            "models_{}".format(db.DEFAULT_SNAPSHOT), "gru")
    arch = STAGES[stage][0]
    return os.path.join(SRC_PKG, tree,
                        "models_{}".format(db.DEFAULT_SNAPSHOT), arch)


def _read_histories(models_dir):
    """
    Read back the per-pass loss curves written under `<models_dir>/_history/`.

    The curves exist only there, inside the snapshot directory that `_stash()` renames
    wholesale; copying them into this stage's JSON is what keeps the earlier stages'
    numbers in the report when one stage is rerun.

    Indexed by model repr (that is, by filename), so the four baseline voices cannot be
    confused.  A failed read is recorded rather than raised, so one missing curve does not
    ruin an otherwise sound report.
    """
    import torch
    hist_dir = os.path.join(models_dir, "_history")
    out = {}
    if not os.path.isdir(hist_dir):
        return out
    for name in sorted(os.listdir(hist_dir)):
        path = os.path.join(hist_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            out[name] = {"error": "{}: {}".format(type(exc).__name__, exc)}
            continue
        out[name] = {
            "batch_size": payload.get("batch_size"),
            "steps_per_epoch": payload.get("steps_per_epoch"),
            "max_steps": payload.get("max_steps"),
            "global_step": payload.get("global_step"),
            "best_val_loss": payload.get("best_val_loss"),
            "n_passes": len(payload.get("history") or []),
            "history": payload.get("history") or [],
        }
    return out


# ---------------------------------------------------------------------------
#  stages
# ---------------------------------------------------------------------------

def stage_cache(tree):
    db = _import_pkg()
    root, cache_dir = _install_tree(db, tree)
    _check_cache_dir(db, tree, cache_dir)
    print("  树: {}   缓存目录: {}".format(tree, _rep(cache_dir)), flush=True)

    _stash(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    print("\n  >>> 重建中（Bach 众赞歌，逐 tick 逐移调切窗）...", flush=True)
    t0 = time.time()
    ds = db.build_dataset()
    dt = time.time() - t0

    n = len(ds.tensor_dataset)
    files = []
    for sub in ("datasets", "tensor_datasets"):
        d = os.path.join(cache_dir, sub)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            files.append({"path": _rep(p), "bytes": os.path.getsize(p)})

    print("\n  窗口数: {:,}  (上一版 {:,})".format(n, PREV_WINDOWS), flush=True)
    for f in files:
        print("  {:<110} {:>13,} B".format(f["path"], f["bytes"]), flush=True)
    print("  重建耗时: {:.1f} min".format(dt / 60), flush=True)
    print("  窗口数一致: {}".format("是" if n == PREV_WINDOWS else "否 —— 请检查"),
          flush=True)

    return {"stage": "cache", "tree": tree, "seconds": round(dt, 1), "windows": n,
            "prev_windows": PREV_WINDOWS, "same_as_prev": n == PREV_WINDOWS,
            "cache_dir": _rep(cache_dir), "files": files}


def _snapshot_report(db, models_dir):
    """Collect the directories and files of every snapshot set into the result dict."""
    out = {}
    for name in db.SNAPSHOT_NAMES:
        d = db.sibling_snapshot_dir(models_dir, name)
        files = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    files.append({
                        "name": f, "bytes": os.path.getsize(p),
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(os.path.getmtime(p)))})
        out[name] = {"dir": _rep(d), "files": files}
    return out


def stage_train(tree, stage):
    # `role_conditioned` is an explicit field, not inferred from a rule like "the arch
    # ends in _rc" — rc shows up only in the checkpoint filename's last field
    # (`,True)` vs `,False)`).
    arch, voice_ids, role_conditioned = STAGES[stage]
    rnn_type = STAGE_RNN.get(stage, "lstm")
    db = _import_pkg()
    root, cache_dir = _install_tree(db, tree)
    _check_cache_dir(db, tree, cache_dir)

    # Check against the package's own list: several stage names in this file are not arch
    # names, and an `arch=` outside `db.ARCHS` raises before the run starts.
    assert arch in db.ARCHS, (arch, db.ARCHS)
    # Same for the cell name: a typo would train a second LSTM into the `gru` directory
    # and report it as success.
    assert rnn_type in db.RNN_TYPES, (rnn_type, db.RNN_TYPES)

    batch_size = TREES[tree]
    models_dir = _models_dir(tree, stage)
    dataset = db.build_dataset(voice_ids=voice_ids) if voice_ids else None

    print("  树: {}   batch={}   路线: {}   cell={}   voice_ids={}   "
          "role_conditioned={}   steps_per_epoch={}  max_steps={}".format(
              tree, batch_size, arch, rnn_type, voice_ids, role_conditioned,
              STEPS_PER_EPOCH, MAX_STEPS), flush=True)
    print("  缓存: {}  权重目录: {}   (默认快照 = {})".format(
        _rep(cache_dir), _rep(models_dir), db.DEFAULT_SNAPSHOT), flush=True)

    # Every snapshot set is moved aside, not just models_dir: mixing them in one directory
    # would let the loader pick between the two sets by listdir order.  `_history/` moves
    # with models_dir, hence the curves have to be stashed into the JSON.
    for name in db.SNAPSHOT_NAMES:
        _stash(db.sibling_snapshot_dir(models_dir, name))
    os.makedirs(models_dir, exist_ok=True)

    t0 = time.time()
    model = db.train_from_scratch(
        steps_per_epoch=STEPS_PER_EPOCH,
        max_steps=MAX_STEPS,
        batch_size=batch_size,
        models_dir=models_dir,
        arch=arch,
        role_conditioned=role_conditioned,
        dataset=dataset,
        rnn_type=rnn_type,
    )
    dt = time.time() - t0

    snapshots = _snapshot_report(db, models_dir)
    histories = _read_histories(models_dir)

    print("\n  用时: {:.1f} min".format(dt / 60), flush=True)
    for name in db.SNAPSHOT_NAMES:
        entry = snapshots[name]
        mark = "  <- 默认加载" if name == db.DEFAULT_SNAPSHOT else ""
        print("  [{}] {}{}".format(name, entry["dir"], mark), flush=True)
        for c in entry["files"]:
            print("      {}  {:>12,} B  {}".format(c["mtime"], c["bytes"],
                                                   c["name"]), flush=True)

    print("\n  曲线 (_history/): {} 个".format(len(histories)), flush=True)
    for name, h in histories.items():
        if "error" in h:
            print("      [读取失败] {}  {}".format(name, h["error"]), flush=True)
            continue
        print("      {} pass, best_val_loss={}, -> {}".format(
            h["n_passes"], h["best_val_loss"], name), flush=True)

    return {"stage": arch, "tree": tree, "batch_size": batch_size,
            "seconds": round(dt, 1),
            "steps_per_epoch": STEPS_PER_EPOCH, "max_steps": MAX_STEPS,
            "voice_ids": voice_ids, "role_conditioned": role_conditioned,
            "rnn_type": rnn_type,
            "models_dir": _rep(models_dir), "cache_dir": _rep(cache_dir),
            "snapshots": snapshots, "histories": histories}


# stage -> (arch, voice_ids, role_conditioned).  `models_dir` is derived by `_models_dir`
# from the tree and the stage, and is not in this table.
#
# **Always three fields**: callers such as `_check_artifacts.py` unpack three, so adding a
# field would ripple into them.  The recurrent cell (a property of the model, not a second
# architecture) lives in `STAGE_RNN` below.
STAGES = {
    "cache":     None,
    # Single-voice parity: one baseline VoiceModel on a voice_ids=[0] dataset, strictly
    # aligned with the 2018 code.  `voice0` is a directory name, not a model name.
    "voice0":    ("baseline",     [0],    False),
    "shared":    ("shared_trunk", None,  False),   # rc=False arm
    # rc=True arm: the shared trunk with the role flag on, its own directory.
    "shared_rc": ("shared_trunk", None,  True),
    "baseline":  ("baseline",     None,  False),
    # GRU arm: the baseline architecture with `nn.LSTM` swapped for `nn.GRU`, everything
    # else (dataset, four voices, 20-pass ladder, batch size) unchanged.  `arch` is still
    # called baseline — a different arch name would send every
    # `arch == 'baseline' / else` test in the package down the wrong branch.
    "gru":       ("baseline",     None,  False),
}


# stage -> recurrent cell, listing only deviations from the default.  A dict rather than a
# fourth field of `STAGES`: see above.  Absent means 'lstm'.
STAGE_RNN = {
    "gru": "gru",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, choices=sorted(TREES))
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = "{}_{}".format(args.tree, args.stage)
    print("=" * 78)
    print("  tree: {}   stage: {}      {}".format(
        args.tree, args.stage, time.strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 78, flush=True)

    t0 = time.time()
    spec = STAGES[args.stage]
    result = stage_train(args.tree, args.stage) if spec else stage_cache(args.tree)
    result["stage"] = args.stage   # voice0 and baseline share an arch, so record by stage
    result["wall_seconds"] = round(time.time() - t0, 1)
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(OUT_DIR, stem + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("\n  [完成] tree {} stage {} 写入 _retrain/{}.json".format(
        args.tree, args.stage, stem), flush=True)


if __name__ == "__main__":
    main()
