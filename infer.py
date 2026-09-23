#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer.py — the local entry point for one inference run: python infer.py

This file answers exactly one question: that musicxml just now — which weights, which
parameters, which seed produced it. So it asserts nothing, it only records facts — the
resolved weights dir, the file the loader will actually read (name + bytes + mtime +
md5), the seed, which switches are on. Printed before the run, written to a same-named
`.json` after it.

Four axes decide which weights a run uses, all overridable on the command line (unset
means the CONFIG value, so `python infer.py` equals editing this file):

    data set   data256 (the batch-256 round) | data128 (the batch-128 round). One axis,
               not two: picking the tree fixes the dataset cache and the weights root.
    arch       baseline | shared_trunk
    rc         only meaningful for shared_trunk
    rnn_type   only for baseline: lstm (default) | gru. Selects the weights leaf
               (`models_<snapshot>/gru`), because it changes the checkpoint filename;
               not a third architecture, arch is still baseline.
    snapshot   one constant per model family (SNAPSHOT_BASELINE / _GRU / _SHARED /
               _SHARED_RC).

    python infer.py                                    # whatever CONFIG says
    python infer.py --arch baseline --snapshot best     # no shared trunk, best snapshot
    python infer.py --rnn-type gru                      # the GRU arm, SNAPSHOT_GRU
    python infer.py --rc --snapshot epoch10             # role_conditioned, pass 10
    python infer.py --data-set data128                  # the batch-128 tree throughout
    python infer.py --help

The output name is built from the **resolved weights dir** (the axes are the request,
the dir is the fact, `--models-dir` separates the two), so changing snapshot changes
the name, combinations never overwrite each other, and a non-default tree enters the
name too. The score's title instead describes this run's mode (axes + all sampling
parameters, see `_score_title`) — it travels with the score, the manifest does not.

The configuration is all in the CONFIG block below. After a run look in two places:
the end of stdout, and `infer_out/<tag>.json`.
"""

import argparse
import json
import os
import random
import sys
import time

# =============================================================================
#  CONFIG — edit here, no need to read on
# =============================================================================

# --- what this run does ---------------------------------------------------
# 'harmonize' adds three voices to a melody (needs MELODY_FILE); 'scratch' makes four
# voices from nothing.
MODE = "harmonize"

# Melody file, used only under MODE='harmonize'. Empty = pop up a file dialog, and that
# dialog is tkinter-modal: under redirection / headless it either errors or hangs forever.
MELODY_FILE = r"C:\Users\Grud\Documents\总谱\melody\belaa-koska.musicxml"
#MELODY_FILE = r"C:\Users\Grud\Documents\总谱\melody\o-haupt-244.54-soprano.musicxml"
#MELODY_FILE = r"C:\Users\Grud\Documents\总谱\melody\Дороги.mxl"
#MELODY_FILE = r"C:\Users\Grud\Documents\总谱\melody\Ode to joy.musicxml"

# --- paths ------------------------------------------------------------------
# Must be that 1.5.0 source tree: the package's own old data/ holds neither weights nor
# cache, and every path below points into one of the two round trees.
#
# Derived from __file__ rather than hardcoded: this file is at the repo root, beside
# src/. Hardcoding it cost us once — it pointed at **another tree** and the log was silent.
_HERE   = os.path.dirname(os.path.abspath(__file__))
SRC_PKG = os.path.join(_HERE, "src", "deepbach_pytorch")

# Custom corpus dir (needed only if training used one, so vocab and weights match); empty
# = the built-in Bach chorales.
CUSTOM_MIDI_DIR = ""

# --- data tree: cache + weights, one choice ------------------------
# The package dir holds two parallel trees, each with a `dataset_cache/` and a full set
# of `models_<snapshot>/`:
#
#     data256/   the batch-256 round (batch 256, 20 passes)
#     data128/   the batch-128 round (batch 128, 20 passes)
#
# **One axis, not two**: the trees are matched products of two training rounds, so
# picking the tree fixes the cache and the weights at once.
#
# Nothing importable names either tree: `helpers.DATA_DIR` and `DatasetManager.__init__`
# both derive `data/` from their own `__file__`, and that old tree is no longer an
# option, so the package can name neither. This constant is the only choice point,
# carried into the run by `_install_tree_cache`.
#
# The tree's three paths (root, cache dir, rc round's models dir) are all computed from
# this name by `_set_data_set()` below; `--data-set` re-runs that one function.
DATA_SETS = ("data256", "data128")
DATA_SET  = "data256"

# --- weights: the four axes ------------------------------------------------
# The axes decide which weights a run uses, all overridable on the command line
# (`python infer.py --help`), and unset means the value here. Axes -> dir is only four
# cases, implemented in `_weights_dir()` below, where `<tree>` is DATA_SET:
#
#     baseline, lstm             <pkg>/<tree>/models_<snapshot>/baseline/
#     baseline, gru              <pkg>/<tree>/models_<snapshot>/gru/
#     shared_trunk, rc=False     <pkg>/<tree>/models_<snapshot>/shared_trunk/
#     shared_trunk, rc=True      <pkg>/<tree>/models_rc/shared_trunk[__<snapshot>]/
#
# 'baseline'     = **no shared trunk**: four independent VoiceModels (four weight files,
#                  trained separately).
# 'shared_trunk' = one shared LSTM trunk + four per-role heads (one joint training).
ARCH = "baseline"

# The baseline's recurrent cell: 'lstm' (default) or 'gru'. **Not a third arch** —
# `ARCH` is still 'baseline', only the cell changes, so every `arch == 'baseline'`
# branch in the package is unaffected.
#
# It changes the weights directory because it changes the **checkpoint filename**: a GRU
# voice's repr ends in `,256,gru)`, an LSTM's in `,256)`. The GRU arm lives at
# `<tree>/models_<snapshot>/gru/`, so this axis picks the leaf, and `_weights_label`
# reads `gru` back out of the path.
#
# Only meaningful for `ARCH='baseline'`: the shared trunk has no cell switch, so
# `--rnn-type gru --arch shared_trunk` is an error right here.
RNN_TYPE = "gru"

# Only meaningful for shared_trunk. baseline has no such switch and the package
# **silently ignores** it — so rc=True with baseline is an error right here, never a
# parameter that is accepted and does nothing.
#
# rc=True has its own weights, under `<tree>/models_rc/` (see RC_MODELS_DIR). They are
# not in the standard snapshot layout, so sibling snapshots carry a `__<snapshot>`
# suffix.
ROLE_CONDITIONED = False

# --- snapshot: one setting per model family --------------------------
# 'best' | 'epoch01' .. 'epoch20' (= db.SNAPSHOT_NAMES). **Every pass is on disk**, one
# directory per pass, so changing snapshot switches directory rather than retraining.
# epoch20 agrees with the package default (helpers.DEFAULT_SNAPSHOT).
#
# **One constant per training round, not per architecture**: these families are separate
# trainings, and val loss bottoms out at different pass counts (measured around epoch5-8),
# so a single SNAPSHOT constant would quietly carry one family's choice into the next.
# Each family gets its own value, and `--snapshot` overrides the one being run.
#
# The GRU arm is the fourth such training (same architecture, different cell, with its
# own 21 snapshots under `models_<snapshot>/gru/`), so it has its own constant rather
# than inheriting the LSTM baseline's.
SNAPSHOT_BASELINE  = "epoch20"
SNAPSHOT_GRU       = "epoch20"
SNAPSHOT_SHARED    = "epoch20"
SNAPSHOT_SHARED_RC = "epoch20"

# Empty = each family uses its own constant above; the command-line `--snapshot` still
# outranks it. Non-empty = this run uses that snapshot whichever family it is (the
# CONFIG version of `--snapshot`).
SNAPSHOT_OVERRIDE = ""

# The rc=True round's directory, inside the selected tree. It is **not** a standard
# snapshot path — training was handed this custom models_dir, so sibling snapshots carry
# a `__<snapshot>` suffix (same convention as `<tree>/models_voice0`, see the custom-dir
# branch of helpers.sibling_snapshot_dir): `shared_trunk` is itself the default snapshot.
#
# Computed from DATA_SET by `_set_data_set()` below: it is that tree's model, and
# hardcoding it would make a second copy.

# Empty = resolved from the axes above. To run an off-table directory write an absolute
# path here — deliberately no "use it if exists" rule, since that rule is exactly the
# failure where half-trained weights quietly become the experiment; `--models-dir`
# outranks it, and both outrank the axes. Moves the **weights** only; the cache is DATA_SET.
WEIGHT_DIR_OVERRIDE = ""

# None = use the architecture default. Both architectures are 256 now (the 2018
# reference), so a wrong value is no longer possible; it stays None so the table is the
# only definition and the literal is not another copy that can drift.
LSTM_HIDDEN_SIZE = None

# --- harmonize() parameters (MODE='harmonize') ----------------------------
HARMONIZE_ITERATIONS      = 1700   # Gibbs iterations; more is slower and better
HARMONIZE_TEMPERATURE     = 0.3    # sampling temperature: 1.0 standard, <1 conservative
HARMONIZE_BATCH_PER_VOICE = 64     # parallel proposals per step
HARMONIZE_MELODY_VOICE    = 0      # which voice is the melody: 0=S,1=A,2=T,3=B
HARMONIZE_KEEP_MELODY     = True   # True rebuilds the others only; False all four

# Input melodies usually carry no fermata, and an all-zero channel means the model sees no
# phrase ending at all, so the harmony has no cadential structure. derive_fermatas fills
# them in per the training convention (last beat of every two bars + final note). Setting
# False changes the output — it is not "draw one fewer mark".
DERIVE_FERMATAS = True

# Note durations run +9% under the fermata channel; the melody itself is fixed, so the
# accompaniment audibly drags at every two-bar boundary. 'none' = do not draw those marks
# in the output (the model still receives the channel).
FERMATA_MARKS = "none"

# --- generate_from_scratch() parameters (MODE='scratch') -------------------
SCRATCH_SEQUENCE_TICKS    = 64     # sixteenth-note tick length, 64 = 4 bars
SCRATCH_ITERATIONS        = 1700
SCRATCH_TEMPERATURE       = 0.3
SCRATCH_BATCH_PER_VOICE   = 64

# --- reproducibility --------------------------------------------------------
# What the sampler is seeded with. Three modes, none of which needs editing this file —
# use `--seed`:
#   "random"  (default) draws a fresh seed per run and records it — printed, in the
#             `s<seed>` segment of the output name, and in the manifest. To reproduce,
#             pass that integer back: `--seed <int>`.
#   <int>     pinned: the same seed gives byte-identical files across processes.
#   None      no seeding at all; the sampler reads global RNG state as it stands.
#             `--seed none`.
# The seed goes to torch / numpy / random / cuda (see `_seed_all`).
SEED = "random"

# Turn on deterministic algorithms. cuDNN's LSTM backward is atomic, so even with a seed
# runs differ by ~1e-3; suppressing that needs cudnn off + use_deterministic_algorithms on.
# Slower.
# model_manager.py sets cudnn.benchmark = True at import, so this must run after the
# import — which is the order the script is in.
DETERMINISTIC = True

# --- output -----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "infer_out")

# Empty = automatic: <melody>_<weights label>_s<seed> / scratch<len>_<weights label>_s<seed>.
# The weights label is read from the **resolved weights dir** (see `_weights_label`), so
# changing snapshot changes the name and combinations do not overwrite each other. The
# command-line `--run-tag` overrides it.
# When writing one by hand, mind what it overwrites — this script has no timestamp fallback.
RUN_TAG = ""

# Reference density printed after the run: onset-tick density (%) of the four voices in
# real Bach chorales. Only meaningful under harmonize, and **not the same measure**: the
# reference is four-voice chorales while your melody may be a folk tune of any length, so
# read it as "are the generated voices balanced against each other", not as an absolute fit.
DENS_BACH = [23.4, 27.5, 28.1, 29.0]

# =============================================================================
#  the infrastructure below — leave alone
# =============================================================================

OUTPUT_DIR              = os.path.abspath(OUTPUT_DIR)
CUSTOM_MIDI_DIR         = os.path.abspath(CUSTOM_MIDI_DIR) if CUSTOM_MIDI_DIR else None
WEIGHT_DIR_OVERRIDE     = os.path.abspath(WEIGHT_DIR_OVERRIDE) if WEIGHT_DIR_OVERRIDE else ""


def _set_data_set(data_set):
    """
    Recompute the tree's three paths and rebind them: TREE_DIR / TREE_CACHE_DIR /
    RC_MODELS_DIR.

    One formula, called twice: at import per CONFIG.DATA_SET, and in `main()` again after
    the command-line override.
    """
    global TREE_DIR, TREE_CACHE_DIR, RC_MODELS_DIR
    TREE_DIR       = os.path.abspath(os.path.join(SRC_PKG, data_set))
    TREE_CACHE_DIR = os.path.join(TREE_DIR, "dataset_cache")
    RC_MODELS_DIR  = os.path.join(TREE_DIR, "models_rc", "shared_trunk")


_set_data_set(DATA_SET)

# When stdout is a pipe / file, Python encodes with the local code page (cp936 here) and
# the ✓ / ✗ / ⚠ below are not in GBK, so it raises UnicodeEncodeError mid-run. A real
# console goes through WriteConsoleW and is unaffected, so it only bites under redirection.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# sys.path needs the package's **parent** dir (…\src), not the package dir itself: a copy
# installed in site-packages would win over any bare `import deepbach_pytorch`.
_SRC_DIR = os.path.dirname(SRC_PKG)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

import deepbach_pytorch as db                                         # noqa: E402

OK = "✓"
BAD = "✗"
WARN = "⚠"


def section(title):
    print(f"\n{'═' * 72}\n  {title}\n{'═' * 72}", flush=True)


def fail(msg, hint=""):
    print(f"\n  {BAD} {msg}", flush=True)
    if hint:
        print(f"    {hint}", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# The four axes -> weights dir. The table is in CONFIG; this is its implementation.
# ---------------------------------------------------------------------------

def _snapshot_for(arch, role_conditioned, rnn_type="lstm"):
    """
    The snapshot name for this family of (arch, role_conditioned, rnn_type).

    Each round has its own constant, so a single SNAPSHOT would carry one family's choice
    into the next. `SNAPSHOT_OVERRIDE` (and `--snapshot` above it) folds them to one value.
    """
    if SNAPSHOT_OVERRIDE:
        return SNAPSHOT_OVERRIDE
    if arch == "baseline":
        return SNAPSHOT_GRU if rnn_type != "lstm" else SNAPSHOT_BASELINE
    return SNAPSHOT_SHARED_RC if role_conditioned else SNAPSHOT_SHARED


def _weights_dir(arch, role_conditioned, snapshot, rnn_type="lstm"):
    """
    (arch, role_conditioned, snapshot, rnn_type) -> absolute weights dir, in the DATA_SET
    tree.

    This deliberately rewrites the "config side" formula instead of using the package's
    `snapshot_dir`: section 1 cross-checks the two, so package layout drift (DATA_DIR
    moving, a snapshot renamed, DEFAULT_SNAPSHOT changing) fails loudly.

    The tree root is the one part the package cannot supply: `db.snapshot_dir` only points
    into `data/`, so section 1 lands the package's answer back on TREE_DIR before comparing.

    `rnn_type` moves the **leaf** only, and only for baseline: the GRU arm is the same
    architecture trained into `<tree>/models_<snapshot>/gru/`, not another snapshot tree. It
    is not in section 1's drift table; the leaf is checked by section 2's
    `check_pretrained_weights` (it expects filenames carrying `,256,gru)`).
    """
    if arch == "baseline":
        leaf = "baseline" if rnn_type == "lstm" else rnn_type
        return os.path.join(TREE_DIR, "models_" + snapshot, leaf)
    if not role_conditioned:
        return os.path.join(TREE_DIR, "models_" + snapshot, "shared_trunk")
    # rc=True had one training round, so its dir is not a standard snapshot path:
    # `shared_trunk` is itself default, siblings add `__<snapshot>`. Constants, not literals.
    return (RC_MODELS_DIR if snapshot == db.DEFAULT_SNAPSHOT
            else RC_MODELS_DIR + "__" + snapshot)


def _pkg_snapshot_dir(snapshot, arch):
    """
    `db.snapshot_dir`, with the root moved from the package's `data/` to the selected tree.

    The package knows one tree only (`helpers.DATA_DIR`, hardcoded from its own `__file__`),
    so for `data128` its answer is right apart from the root. Take its path **relative** to
    `db.DATA_DIR`, and section 1's drift check survives: rename a segment and it fails.
    """
    return os.path.join(TREE_DIR,
                        os.path.relpath(db.snapshot_dir(snapshot, arch),
                                        db.DATA_DIR))


def _tree_leaf(weights_dir):
    """
    Which tree weights_dir belongs to — `data256`, `data128`, or None if unrecognised.

    The tree is the level **above** the snapshot segment: `<pkg>/<tree>/models_<snap>/<arch>`
    and `<pkg>/<tree>/models_rc/<arch>[__<snap>]`. Read from the path, not from DATA_SET, as
    in `_weights_label`: with `--models-dir` the two differ, and the dir decides the files.

    A path outside a tree returns None, not a leaf name — a temp dir never enters the name.
    """
    leaf = os.path.basename(os.path.dirname(os.path.dirname(weights_dir)))
    return leaf if leaf in DATA_SETS else None


def _weights_label(weights_dir, arch, role_conditioned):
    """
    The "which weights" segment of the output name (`<melody>_<segment>_s<seed>`).

    **Read the dir, and fall back to the axes only if that fails** — the dir is the fact,
    the axes are the request, and `--models-dir` / WEIGHT_DIR_OVERRIDE separate the two.

        <pkg>/data256/models_best/shared_trunk      -> shared_trunk_best
        <pkg>/data256/models_rc/shared_trunk__epoch10  -> shared_trunk_rc_epoch10
        <pkg>/data256/models_epoch20/baseline       -> baseline
        <pkg>/data256/models_epoch20/gru            -> gru
        <pkg>/data128/models_rc/shared_trunk__epoch10
                                                    -> data128_shared_trunk_rc_epoch10

    The GRU arm needs no rule of its own: its leaf is `gru`, and the standard-layout
    branch reads it straight back.

    The default snapshot is not written into the name (`models_rc/shared_trunk` is itself
    the default snapshot, same convention). rc can only come from `role_conditioned`: the
    dir name does not show it, it appears only in the checkpoint filename suffix.

    **The tree appears only when it is not `data256`**, per the snapshot rule above:
    otherwise two trees' runs at the same (arch, rc, snapshot) collide on one filename.
    """
    leaf   = os.path.basename(weights_dir)
    parent = os.path.basename(os.path.dirname(weights_dir))
    snapshot, arch_name = None, arch

    if parent.startswith("models_") and parent[len("models_"):] in db.SNAPSHOT_NAMES:
        # standard layout `models_<snapshot>/<arch>`: the middle segment is the snapshot,
        # the leaf the arch name.
        snapshot, arch_name = parent[len("models_"):], leaf
    else:
# custom dir (models_rc, models_voice0…): sibling snapshots carry a suffix.
        for name in db.SNAPSHOT_NAMES:
            if name != db.DEFAULT_SNAPSHOT and leaf.endswith("__" + name):
                snapshot = name
                arch_name = leaf[: -len("__" + name)]
                break
        if arch_name not in db.ARCHS:
            # Unrecognised arch name (models_voice0, say). A deliberate fallback: the name
            # degrades to the one the axes imply, and the path still enters the manifest.
            arch_name = arch
    if snapshot is None:
        snapshot = db.DEFAULT_SNAPSHOT

    label = arch_name + ("_rc" if role_conditioned else "")
    if snapshot != db.DEFAULT_SNAPSHOT:
        label = f"{label}_{snapshot}"
    tree = _tree_leaf(weights_dir)
    if tree and tree != DATA_SETS[0]:
        label = f"{tree}_{label}"
    return label


# ---------------------------------------------------------------------------
# the selected tree's dataset cache
# ---------------------------------------------------------------------------
#
# The tree's cache is `<tree>/dataset_cache`, a name the package cannot produce
# (`helpers.DATA_DIR` and `DatasetManager.__init__` both hardcode `data/` from their own
# `__file__`). Both places on the inference path that read the cache are module globals, so
# rebinding these two names is what carries DATA_SET into the run:
#
#   built-in Bach corpus   `deepbach_pytorch.DatasetManager`   (see `_get_default_dataset`)
#   custom corpus          `deepbach_pytorch.DATASET_CACHE_DIR`
#
# Nothing else on this path reads them, and neither `harmonize()` nor
# `generate_from_scratch()` takes a `dataset` argument, so there is no other way in.

def _install_tree_cache(cache_dir):
    """
    Point this process's dataset cache at `cache_dir`.

    Rebinding is **not** optional: the package only points at its old `data/`, which is
    wrong for any round tree. The binding also makes the cache dir a **stated** fact about
    the run, so section 2 can print the real path and the manifest can record it.

    `cache_dir` is closed over rather than read from TREE_CACHE_DIR, so the printed path
    and the path actually read cannot disagree.

    Returns the old binding, which the caller may put back.
    """
    saved = (db.DatasetManager, db.DATASET_CACHE_DIR)
    base = db.DatasetManager

    class _TreeDatasetManager(base):
        def __init__(self):
            super().__init__()
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)

    db.DatasetManager = _TreeDatasetManager
    # `_get_default_dataset`'s custom-corpus branch reads this module global rather than a
    # DatasetManager instance, so the two must move together.
    db.DATASET_CACHE_DIR = cache_dir
    return saved


def _bach_cache_paths():
    """
    Which two cache files the built-in Bach corpus reads from, absolute paths.

    Goes through the package's own `selfcheck.cache_paths`: it builds an empty
    `ChoraleDataset` with `corpus_it_gen=None`, reading no corpus and building no tensors,
    so it is pure path arithmetic. The filenames **are** `ChoraleDataset.__repr__()` — long
    and extensionless — and copying them here would drift from the loader.

    Valid for the built-in corpus only: a custom corpus's dataset is called
    `custom_<basename>`, so a caller must not use these paths when CUSTOM_MIDI_DIR is set.
    """
    return db.selfcheck.cache_paths(cache_dir=TREE_CACHE_DIR)


# ---------------------------------------------------------------------------
# score title: writing "which mode made this xml" into the score itself
# ---------------------------------------------------------------------------

# Snapshot name -> title segment: `best` becomes `bestEpoch`, `epoch20` becomes `20Epoch`
# (`models_epoch20` reads like a path, not a mode). The labels are generated from
# SNAPSHOT_NAMES rather than listed by hand: a hand-kept list that misses a pass falls back
# to the raw name (`epoch07`).
_EPOCH_LABELS = dict(
    {"best": "bestEpoch"},
    **{name: "{}Epoch".format(int(name[len("epoch"):]))
       for name in db.SNAPSHOT_NAMES if name.startswith("epoch")},
)


def _score_title(subject, snapshot, arch, role_conditioned, seed,
                 iterations, temperature, batch_size, keep_melody, tree=None,
                 rnn_type="lstm"):
    """
    <melody>_[<tree>_]<epoch mode>_<arch>[_<cell>]_rc<0|1>_s<seed>_i<iterations>_T<temp>_b<batch>[_km<0|1>]

    e.g. belaa-koska_15Epoch_shared_trunk_rc0_s0_i1700_T0.3_b64_km1
         belaa-koska_15Epoch_baseline_gru_rc0_s0_i1700_T0.3_b64_km1

    The title travels with the score, the manifest does not (the manifest is **the file
    beside it**, and the two part company the moment the xml is taken away), so "which
    weights, which parameters" has to be written in the title.

    Division of labour with the output name (`tag`): tag = **the weights** (read from the
    resolved dir, see `_weights_label`), title = **this run** (axes + all sampling
    parameters). Both are decided by configuration alone — the title carries **no
    timestamp**, or two runs of the same configuration would have different titles and the
    canonical md5 below could not answer "are these two generations the same music".

    `tree` (which tree the weights came from) and `cell` appear only when they are not the
    default, per the snapshot rule below: LSTM is the default, so existing titles stay
    byte-identical and only a GRU run writes it out. `km` (keep_melody) exists only under
    harmonize; scratch passes None and the segment is dropped.
    """
    epoch = _EPOCH_LABELS.get(snapshot, snapshot)
    parts = [subject]
    if tree and tree != DATA_SETS[0]:
        parts.append(tree)
    parts += [
        epoch,
        arch,
    ]
    if rnn_type != "lstm":
        parts.append(rnn_type)
    parts += [
        f"rc{1 if role_conditioned else 0}",
        f"s{seed}" if seed is not None else "sfree",
        f"i{iterations}",
        f"T{temperature:g}",
        f"b{batch_size}",
    ]
    if keep_melody is not None:
        parts.append(f"km{1 if keep_melody else 0}")
    return "_".join(parts)


def _set_score_title(score, title):
    """
    Write `title` into the score's metadata. For a Score, music21 writes `Metadata.title`
    to both `<work><work-title>` and `<movement-title>` — which is what notation software
    displays.

    Metadata only, nothing on disk: the caller writes after this, see the single write
    point in `main()`.
    """
    import music21.metadata
    # A Score built by tensor_to_score() has metadata None (not an empty Metadata), so
    # `.title = ...` on it would be an AttributeError.
    if score.metadata is None:
        score.metadata = music21.metadata.Metadata()
    score.metadata.title = title


# ---------------------------------------------------------------------------
# command line: axes + weights dir + output name
# ---------------------------------------------------------------------------

def _str2bool(text):
    """The type for `--rc`: true/false, and 1/0, yes/no, on/off as well."""
    lowered = text.strip().lower()
    if lowered in ("true", "1", "yes", "y", "on"):
        return True
    if lowered in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean (true/false), got {text!r}")


def _parse_args(argv=None):
    """
    Override a few axes (+ weights dir + output name). **Unset means the CONFIG value.**

    Only these switches are exposed: iterations / temperature stay in CONFIG — this script
    asks "which weights does this run use", not "sweep the parameters", and every extra
    switch is another chance for a parameter that is accepted and does nothing.

    `choices` come from `db.ARCHS` / `db.SNAPSHOT_NAMES` rather than a local copy (a copy
    drifts). `--data-set` is the exception: `data128` is this repo's own tree, the package
    has no name for it, so it comes from the local DATA_SETS.
    """
    ap = argparse.ArgumentParser(
        description="Single inference run: which weights, which parameters and which "
                    "seed produced this musicxml — printed before the run and written "
                    "to a sibling .json afterwards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python infer.py                                     "
               "run whatever CONFIG says\n"
               "  python infer.py --arch baseline --snapshot best      "
               "no shared trunk, best snapshot\n"
               "  python infer.py --rnn-type gru                       "
               "the GRU arm, <tree>/models_<snapshot>/gru\n"
               "  python infer.py --rc                                 "
               "role_conditioned, at CONFIG.SNAPSHOT_SHARED_RC\n"
               "  python infer.py --data-set data128                   "
               "the batch-128 tree (cache + weights)\n"
               "  python infer.py --models-dir D:\\path\\to\\weights      "
               "name the dir directly (axes no longer pick it)\n"
               "  python infer.py --seed 12345                         "
               "reproduce a past run (default: a fresh random seed)\n")
    ap.add_argument("--arch", choices=list(db.ARCHS), default=None,
                    help="override CONFIG.ARCH. baseline = four independent "
                         "VoiceModels (no shared trunk); shared_trunk = plan A")
    ap.add_argument("--rnn-type", choices=list(db.RNN_TYPES), default=None,
                    help="override CONFIG.RNN_TYPE: the baseline's recurrent cell. "
                         "'gru' selects <tree>/models_<snapshot>/gru (a separate arm "
                         "with its own 21 snapshots); it is not a third architecture, "
                         "so --arch stays baseline. Only meaningful for baseline")
    ap.add_argument("--rc", nargs="?", const=True, default=None, type=_str2bool,
                    metavar="BOOL",
                    help="override CONFIG.ROLE_CONDITIONED, only meaningful for "
                         "shared_trunk (bare --rc means --rc true)")
    ap.add_argument("--snapshot", choices=list(db.SNAPSHOT_NAMES), default=None,
                    help="override the snapshot for this run only. Without it each "
                         "model family uses its own CONFIG constant "
                         "(SNAPSHOT_BASELINE / SNAPSHOT_GRU / SNAPSHOT_SHARED / "
                         "SNAPSHOT_SHARED_RC). "
                         "Every pass has a snapshot on disk; this switches directory, "
                         "not retraining")
    ap.add_argument("--data-set", choices=list(DATA_SETS), default=None,
                    help="override CONFIG.DATA_SET: which tree supplies the dataset "
                         "cache **and** the weights root (data256 = batch 256, "
                         "data128 = batch 128). Not two independent choices — the "
                         "trees are matched products of two training rounds")
    ap.add_argument("--models-dir", default=None, metavar="DIR",
                    help="name the weights dir directly, taking priority over the "
                         "axes (= CONFIG.WEIGHT_DIR_OVERRIDE). Moves the weights "
                         "only: the cache still comes from --data-set")
    ap.add_argument("--run-tag", default=None, metavar="TAG",
                    help="override CONFIG.RUN_TAG (output name, no extension)")
    ap.add_argument("--seed", default=_SEED_UNSET, type=_str2seed, metavar="SEED",
                    help="override CONFIG.SEED (default 'random'): an integer pins the "
                         "run, 'random' draws a fresh one, 'none' does not seed at all. "
                         "The default already draws a new seed per run — use an integer "
                         "only to reproduce a run whose seed the manifest recorded")
    return ap.parse_args(argv)


# The "not given" marker for `--seed`. `default=None` cannot tell "not given" from
# "`--seed none`" (= no seeding), and the `args.x or CONFIG.x` the other axes use would
# quietly turn an explicit none back into the CONFIG value — this is the one place where
# "given, and given as None" must survive the merge.
_SEED_UNSET = object()


def _str2seed(text):
    """The argparse type for `--seed`: an integer, `random`, or `none` (returns None)."""
    low = text.strip().lower()
    if low in ("random", "rand", "auto"):
        return "random"
    if low in ("none", "no", "off", "free"):
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, 'random' or 'none', got {text!r}")


def _resolve_seed(seed):
    """
    Turn the CONFIG/CLI value into the integer actually used for seeding, or None for "no
    seeding".

    "random" comes from `SystemRandom`, deliberately not from `random` itself: the seeding
    below resets `random`, so drawing there would make the seed depend on RNG state this
    script has not set yet.
    """
    if seed == "random":
        return random.SystemRandom().randrange(2 ** 31)
    return seed


def _seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The sampler actually uses numpy (np.random.randint / np.random.multinomial in
    # model_manager.parallel_gibbs, random_score_tensor in chorale_dataset); the torch
    # seeds do not reach it.
    np.random.seed(seed)
    random.seed(seed)


def _determinism_on():
    saved = (torch.backends.cudnn.enabled,
             torch.backends.cudnn.benchmark,
             torch.are_deterministic_algorithms_enabled())
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return saved


def _determinism_restore(saved):
    torch.backends.cudnn.enabled = saved[0]
    torch.backends.cudnn.benchmark = saved[1]
    torch.use_deterministic_algorithms(saved[2])


def main():
    # -------------------------------------------------------------------------
    #  Effective per-axis values (command line > CONFIG). Validation precedes section 1,
    #  which hands the snapshot to db.snapshot_dir — that raises on an unknown one rather
    #  than returning. argparse's choices guard the CLI; this guards CONFIG typos.
    # -------------------------------------------------------------------------
    args     = _parse_args()

    # Seed: reduced to an integer first, before any printing or naming, so everything
    # downstream (`_score_title`, the `s<seed>` name, the seed line, `_seed_all`, the
    # manifest) sees the number actually used, not the mode string. Same shape as the
    # other axes, but "not given" and "given none" differ — it tests `is _SEED_UNSET`.
    seed_raw = SEED if args.seed is _SEED_UNSET else args.seed
    seed_was_random = (seed_raw == "random")
    seed = _resolve_seed(seed_raw)

    arch     = args.arch or ARCH
    rc       = ROLE_CONDITIONED if args.rc is None else args.rc
    rnn_type = args.rnn_type or RNN_TYPE
    data_set = args.data_set or DATA_SET
    # snapshot is resolved after the checks below, because its source — the per-family
    # constants — depends on arch and rc.
    snapshot = args.snapshot or SNAPSHOT_OVERRIDE
    from_cli = [name for name, given in (("arch", args.arch),
                                         ("role_conditioned", args.rc),
                                         ("rnn_type", args.rnn_type),
                                         ("snapshot", args.snapshot),
                                         ("data_set", args.data_set),
                                         ("seed", None if args.seed is _SEED_UNSET
                                          else args.seed))
                if given is not None]

    if arch not in db.ARCHS:
        fail(f"ARCH must be one of {list(db.ARCHS)}, got {arch!r}")
    if rnn_type not in db.RNN_TYPES:
        fail(f"RNN_TYPE must be one of {list(db.RNN_TYPES)}, got {rnn_type!r}")
    if data_set not in DATA_SETS:
        fail(f"DATA_SET must be one of {list(DATA_SETS)}, got {data_set!r}")
    if not snapshot:
        # Neither --snapshot nor CONFIG.SNAPSHOT_OVERRIDE: the family decides. Resolved
        # here rather than as a CONFIG default so the family constants stay the definition.
        snapshot = _snapshot_for(arch, rc, rnn_type)
    if snapshot not in db.SNAPSHOT_NAMES:
        fail(f"snapshot must be one of {list(db.SNAPSHOT_NAMES)}, got {snapshot!r}",
             "it came from "
             + ("--snapshot" if args.snapshot else
                "CONFIG.SNAPSHOT_OVERRIDE" if SNAPSHOT_OVERRIDE else
                "CONFIG.SNAPSHOT_" + ("GRU" if (arch == "baseline"
                                                and rnn_type != "lstm")
                                      else "BASELINE" if arch == "baseline"
                                      else "SHARED_RC" if rc else "SHARED")))
    if rc and arch == "baseline":
        fail("role_conditioned=True with arch='baseline': baseline has no such "
             "switch and the package silently ignores it, so this is an error "
             "rather than a parameter that is accepted and does nothing",
             "the rc weights are shared-trunk — use --arch shared_trunk; "
             "or drop role conditioning (--rc false).")
    if rnn_type != "lstm" and arch != "baseline":
        # Same reason as the rc check: the shared trunk has no cell switch, so this would
        # be accepted and ignored by model construction (which `_get_default_model` also
        # refuses, but only after the weights dir is resolved).
        fail(f"RNN_TYPE={rnn_type!r} with arch={arch!r}: only the baseline has a "
             f"recurrent-cell switch, and its GRU arm is arch='baseline' with "
             f"RNN_TYPE='gru'",
             "the GRU weights live at <tree>/models_<snapshot>/gru and are a "
             "baseline model — use --arch baseline, or drop --rnn-type.")

    # TREE_DIR / TREE_CACHE_DIR / RC_MODELS_DIR are module constants computed at import from
    # CONFIG.DATA_SET, so an override re-runs that one formula, not one of the three.
    _set_data_set(data_set)

    # Dir and model construction are two things: the dir decides which file is read, the
    # axes decide how the model is built. With --models-dir the former wins, and a mismatch
    # shows up in section 2 as "incomplete weights".
    override      = (os.path.abspath(args.models_dir) if args.models_dir
                     else WEIGHT_DIR_OVERRIDE)
    weights_dir   = override or _weights_dir(arch, rc, snapshot, rnn_type)
    weights_label = _weights_label(weights_dir, arch, rc)
    tree_leaf     = _tree_leaf(weights_dir)

    # =========================================================================
    # 1 — which package this is
    # =========================================================================
    section("1 — 包来源")

    db_file = os.path.abspath(db.__file__)
    if not db_file.startswith(os.path.abspath(SRC_PKG)):
        fail(f"import resolved to {db_file}, not the configured source tree",
             f"sys.path needs the package's parent dir ({_SRC_DIR}), not the "
             f"package dir itself — the latter silently falls back to site-packages.")
    print(f"  {OK} {os.path.relpath(db_file, os.path.dirname(SRC_PKG))}", flush=True)

    # CONFIG's path formula vs the layout the package derives from its own __file__.
    # **Every family is checked, each with its own snapshot** — not just this run's — so a
    # wrong SNAPSHOT_SHARED is caught by a baseline run too.
    #
    # The family constants are checked at their own values (not via `_snapshot_for`),
    # since they are the definition; the effective snapshot is validated in `main()`.
    #
    # `_rnn` keeps the "this run" marker below off the LSTM baseline while running GRU.
    # The GRU leaf is deliberately **not** a fourth row: this table's other column is
    # `db.snapshot_dir`, and the package has no name for a `gru` snapshot; the GRU dir is
    # checked by section 2.
    families = (("baseline", "baseline", False, SNAPSHOT_BASELINE, "lstm"),
                ("shared_trunk", "shared_trunk", False, SNAPSHOT_SHARED, "lstm"),
                ("shared_trunk rc=True", "shared_trunk", True, SNAPSHOT_SHARED_RC,
                 "lstm"))
    for _label, _arch, _rc, _snap, _rnn in families:
        if _rc:
            _cfg = _weights_dir(_arch, True, _snap)
            _pkg = db.sibling_snapshot_dir(RC_MODELS_DIR, _snap)
        else:
            _cfg = _weights_dir(_arch, False, _snap, _rnn)
            _pkg = _pkg_snapshot_dir(_snap, _arch)
        if os.path.normcase(_cfg) != os.path.normcase(_pkg):
            fail(f"weights dir for {_label}: CONFIG says {_cfg} but "
                 f"the package layout says {_pkg}",
                 f"CONFIG's three paths are computed from DATA_SET (={data_set}) and "
                 f"the family snapshot constants; the package derives its own from "
                 f"helpers.DATA_DIR and helpers.DEFAULT_SNAPSHOT. They must agree, "
                 f"or the run reads weights from a directory that was never built."
                 + (f"\n    the rc round additionally uses RC_MODELS_DIR = "
                    f"{RC_MODELS_DIR}, whose sibling-snapshot naming must match "
                    f"helpers.sibling_snapshot_dir." if _rc else ""))

    # Which family reads which tree + which snapshot; all three printed for eyeballing.
    print(f"\n  树 {data_set}: {TREE_DIR}", flush=True)
    for _label, _arch, _rc, _snap, _rnn in families:
        _eff = _snapshot_for(_arch, _rc, _rnn)
        _mark = ("  <- 本次运行" if (_arch == arch and _rc == rc and _rnn == rnn_type)
                 else "")
        _note = "" if _eff == _snap else f"（配 SNAPSHOT_OVERRIDE，覆盖 {_snap}）"
        print(f"    {_label:<20} 快照 {_eff}{_note}{_mark}", flush=True)
    if rnn_type != "lstm":
        # The GRU arm is not in the three rows above (see the note at `families`), so state
        # outright which dir this run actually reads.
        print(f"    {'baseline cell=' + rnn_type:<20} 快照 {snapshot}"
              f"   目录 {weights_dir}  <- 本次运行", flush=True)

    # =========================================================================
    # 2 — cache and weights preflight (before the run, not after)
    # =========================================================================
    section("2 — 缓存与权重")

    # --------------------------------------------------------------- cache
    # The dataset cache feeds `voice_ranges` and the metadata encoder, and it is what tells
    # the model where the vocabulary is. Checked **before** the run: without it
    # `DatasetManager` treats "no cache object" as "build one", deletes the existing tensor
    # file and re-reads the whole corpus — tens of minutes and a few GB of RAM.
    #
    # Rebinding first: everything below reads the cache through the package, so the dir
    # must be installed before any path is computed, or the printed path describes the
    # default tree while another one runs.
    _install_tree_cache(TREE_CACHE_DIR)
    print(f"  缓存目录: {TREE_CACHE_DIR}   (DATA_SET={data_set})", flush=True)
    if CUSTOM_MIDI_DIR:
        # `_get_default_dataset`'s custom branch calls the dataset `custom_<basename>`, so
        # the bach_chorales filenames `selfcheck.cache_paths` gives are not the ones to
        # check. The dir still follows the tree (it reads db.DATASET_CACHE_DIR, which
        # `_install_tree_cache` bound), so report the dir and skip the file check.
        print(f"  {WARN} 自定义语料 {CUSTOM_MIDI_DIR}：缓存文件名为 "
              f"custom_{os.path.basename(CUSTOM_MIDI_DIR)}(<vocab>)，"
              f"不按 Bach 众赞歌的文件名预检", flush=True)
    else:
        objs_path, tensors_path = _bach_cache_paths()
        if not os.path.exists(objs_path):
            fail(f"数据集缓存不存在：{objs_path}",
                 f"树 {data_set} 没有已建好的缓存。缺这个文件时 `DatasetManager` 会"
                 f"当成「需要重建」，删掉 {tensors_path} 并重新读整个语料——"
                 f"那是几十分钟和几 GB 内存，不是这次运行要问的问题。\n"
                 f"      要么改用已有的树（--data-set {'/'.join(DATA_SETS)}），"
                 f"要么先建缓存（仓库根的 _retrain.py 缓存 stage）。")
        print(f"  {OK} {os.path.basename(objs_path)}", flush=True)
        print(f"      {os.path.getsize(objs_path):,} B", flush=True)
        print(f"      张量缓存 {'在' if os.path.exists(tensors_path) else '不在'}"
              f"：{tensors_path}", flush=True)
        # Generation never reads the tensor cache (only `data_loaders` and
        # `analysis.build_eval_batches` do), so a missing one is fine; said out loud so it
        # is not mistaken for a problem.
        print(f"      （生成不读张量缓存，只读上面那个对象文件；"
              f"张量只用于训练与评测）", flush=True)

    # ------------------------------------------------------------- weights
    expect_files = 1 if arch == "shared_trunk" else 4

    status = db.check_pretrained_weights(
        models_dir=weights_dir, arch=arch,
        role_conditioned=rc,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        rnn_type=rnn_type)

    source = ("--models-dir" if args.models_dir else
              "CONFIG.WEIGHT_DIR_OVERRIDE" if WEIGHT_DIR_OVERRIDE else
              "轴解析")
    print(f"  目录   : {weights_dir}   ({source})", flush=True)
    print(f"  架构   : {arch}"
          f"{'  role_conditioned=True' if rc else ''}"
          f"{'  cell=' + rnn_type if rnn_type != 'lstm' else ''}"
          f"   快照 {snapshot}"
          f"   需要 {status['required_weights']} 个文件", flush=True)
    if override and from_cli:
        # Dir and axes are two things: the dir decides which file is read, the axes decide
        # how the model is built. A mismatch shows up below as "incomplete weights" (the
        # expected filenames carry arch + role_conditioned).
        print(f"  {WARN} 目录由 {source} 决定，命令行给的轴"
              f"（{', '.join(from_cli)}）只决定模型怎么构造", flush=True)
    if tree_leaf and tree_leaf != data_set:
        # `--models-dir` moved the weights but not the cache (the cache follows
        # --data-set). This is usually a deliberate "these weights against that cache"
        # comparison, so it is called out rather than left implicit.
        print(f"  {WARN} 权重在 {tree_leaf} 树里，缓存仍取 {data_set} 树——"
              f"两侧来自不同训练轮次。做这个对比时请确认是有意的。", flush=True)

    if not status["complete"]:
        print(f"  目录里 : files={status.get('present_files', [])} "
              f"dirs={status.get('present_dirs', [])}", flush=True)
        if not os.path.isdir(weights_dir):
            # check_pretrained_weights returns early on a missing dir, and that dict has
            # found_weights = 0 with an empty missing_files — two numbers that contradict
            # each other, so copying them out reads as "all the files are there". There is
            # only one real cause, so state that one.
            head = f"weights directory does not exist: {weights_dir}"
        else:
            head = (f"incomplete weights: found {status['found_weights']}"
                    f"/{status['required_weights']}; "
                    f"missing {status.get('missing_files', [])}")
        # The hint does not hardcode `max_steps`: any literal is this cache's step count and
        # goes stale with it. Omitting both budget parameters gives the reference budget.
        if arch != "shared_trunk":
            hint = ("baseline needs 4 VoiceModel files — download the pretrained "
                    "weights, or train one with "
                    "train_from_scratch(arch='baseline', ...)")
        elif rc:
            hint = ("rc=True has one training round only and is not a _retrain.py "
                    "stage — point models_dir at this directory when training:\n"
                    "      python -c \"import deepbach_pytorch as db;"
                    " db.train_from_scratch(arch='shared_trunk',"
                    " role_conditioned=True, models_dir=r'" + RC_MODELS_DIR + "',"
                    " batch_size=512)\"")
        else:
            hint = ("the shared trunk has 1 checkpoint — train one:\n"
                    "      python -c \"import deepbach_pytorch as db;"
                    " db.train_from_scratch(arch='shared_trunk',"
                    " batch_size=512)\"")
        fail(head, hint)

    # Which file the loader will actually read. `expected[i]['file']` is the **matched
    # name**, not the suffix being looked for — printing the suffix gives a long useless repr.
    loader_pick = [e["file"] for e in status["expected"]]
    signature = db.analysis.weight_dir_signature(weights_dir,
                                                 expect_files=expect_files)
    md5s = signature["md5s"]
    for name in loader_pick:
        entry = next((f for f in signature["files"] if f["name"] == name), None)
        if entry:
            print(f"  {OK} {name}", flush=True)
            print(f"      {entry['bytes']:,} B   mtime {entry['mtime']}", flush=True)
            print(f"      md5 {entry['md5']}", flush=True)

    if status.get("present_dirs"):
        print(f"  （子目录，不是权重：{status['present_dirs']}）", flush=True)

    # Extra files are not an error, but the loader takes the first endswith() match in
    # os.listdir order, so one extra matching name hands model selection to the filesystem.
    if not signature["as_expected"]:
        print(f"  {WARN} 目录里有 {signature['n_files']} 个文件，期望 "
              f"{expect_files} 个；加载器取 os.listdir 顺序里的第一个 endswith() "
              f"匹配。", flush=True)

    # =========================================================================
    # 3 — what this run will do
    # =========================================================================
    section("3 — 运行计划")

    melody_path = None
    input_notes = None
    if MODE == "harmonize":
        if not MELODY_FILE:
            fail("MODE='harmonize' but MELODY_FILE is empty")
        melody_path = os.path.abspath(MELODY_FILE)
        if not os.path.exists(melody_path):
            fail(f"melody file does not exist: {melody_path}")
        import music21
        input_notes = list(music21.converter.parse(melody_path).parts[0]
                           .flatten().notes)
        print(f"  模式   : harmonize  旋律 {os.path.basename(melody_path)}"
              f"  ({len(input_notes)} 音)", flush=True)
        print(f"  md5    : {db.analysis.md5_file(melody_path)}", flush=True)
        print(f"  参数   : melody_voice={HARMONIZE_MELODY_VOICE}"
              f"  keep_melody={HARMONIZE_KEEP_MELODY}"
              f"  iterations={HARMONIZE_ITERATIONS}"
              f"  T={HARMONIZE_TEMPERATURE}"
              f"  batch={HARMONIZE_BATCH_PER_VOICE}"
              f"  derive_fermatas={DERIVE_FERMATAS}"
              f"  fermata_marks={FERMATA_MARKS!r}", flush=True)
        # The title and the output name share this melody name; both come from here.
        subject = os.path.splitext(os.path.basename(melody_path))[0]
        run_iterations, run_temperature, run_batch, run_keep_melody = (
            HARMONIZE_ITERATIONS, HARMONIZE_TEMPERATURE,
            HARMONIZE_BATCH_PER_VOICE, HARMONIZE_KEEP_MELODY)
    elif MODE == "scratch":
        print(f"  模式   : scratch  长度 {SCRATCH_SEQUENCE_TICKS} tick", flush=True)
        print(f"  参数   : iterations={SCRATCH_ITERATIONS}"
              f"  T={SCRATCH_TEMPERATURE}"
              f"  batch={SCRATCH_BATCH_PER_VOICE}", flush=True)
        subject = f"scratch{SCRATCH_SEQUENCE_TICKS}"
        # keep_melody **does not exist** under scratch (generate_from_scratch does not take
        # it), so pass None rather than False — the segment is dropped, not read as km0.
        run_iterations, run_temperature, run_batch, run_keep_melody = (
            SCRATCH_ITERATIONS, SCRATCH_TEMPERATURE,
            SCRATCH_BATCH_PER_VOICE, None)
    else:
        fail(f"MODE must be 'harmonize' or 'scratch', got {MODE!r}")

    default_tag = f"{subject}_{weights_label}"
    # The title uses the **effective** values (after the command-line override), not the
    # ones written in CONFIG — reproducing this xml later needs what was effective. The
    # weights dir itself is covered by the manifest's models_dir + md5.
    #
    # `tree` comes from the resolved dir (`_tree_leaf`) rather than `data_set`, so the title
    # says which tree the weights really came from; with `--models-dir` the two can differ.
    run_title = _score_title(subject, snapshot, arch, rc, seed,
                             run_iterations, run_temperature, run_batch,
                             run_keep_melody, tree=tree_leaf,
                             rnn_type=rnn_type)

    seed_label = f"s{seed}" if seed is not None else "sfree"
    tag = args.run_tag or RUN_TAG or f"{default_tag}_{seed_label}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{tag}.musicxml")

    if seed is None:
        print(f"  {WARN} seed=None：不播种，这次的结果下次跑不出来", flush=True)
    else:
        # A drawn seed is only worth anything if it is said out loud: this line makes the
        # run reproducible (and it is the number the `s<seed>` output name and manifest also
        # carry).
        drawn = f"   本次随机抽取，--seed {seed} 复现这一份" if seed_was_random else ""
        print(f"  种子   : {seed}"
              f"{'   确定性算法已开' if DETERMINISTIC else '   （确定性算法关）'}"
              f"{drawn}",
              flush=True)
    # The name is built from the weights dir (not the axes), so both are laid out here:
    # combinations never collide, and "does this name match the weights" is answered by the
    # manifest's md5. After RUN_TAG / --run-tag the name has nothing to do with the weights.
    tag_source = ("--run-tag" if args.run_tag else
                  "CONFIG.RUN_TAG" if RUN_TAG else "自动（旋律名+权重标签+种子）")
    print(f"  权重标签: {weights_label}"
          f"   选项来自 {'命令行 ' + ', '.join(from_cli) if from_cli else 'CONFIG'}"
          f"   输出名来自 {tag_source}", flush=True)
    print(f"  输出   : {out_path}", flush=True)

    # =========================================================================
    # 4 — generation
    # =========================================================================
    section("4 — 生成")

    saved = _determinism_on() if DETERMINISTIC else None
    t0 = time.time()
    try:
        if seed is not None:
            _seed_all(seed)
        if MODE == "harmonize":
            score = db.harmonize(
                input_file           = melody_path,
                output_path          = out_path,
                num_iterations       = HARMONIZE_ITERATIONS,
                temperature          = HARMONIZE_TEMPERATURE,
                batch_size_per_voice = HARMONIZE_BATCH_PER_VOICE,
                melody_voice         = HARMONIZE_MELODY_VOICE,
                keep_melody          = HARMONIZE_KEEP_MELODY,
                models_dir           = weights_dir,
                custom_midi_dir      = CUSTOM_MIDI_DIR,
                derive_fermatas      = DERIVE_FERMATAS,
                fermata_marks        = FERMATA_MARKS,
                arch                 = arch,
                role_conditioned     = rc,
                lstm_hidden_size     = LSTM_HIDDEN_SIZE,
                rnn_type             = rnn_type,
            )
        else:
            score = db.generate_from_scratch(
                sequence_length_ticks = SCRATCH_SEQUENCE_TICKS,
                num_iterations        = SCRATCH_ITERATIONS,
                temperature           = SCRATCH_TEMPERATURE,
                batch_size_per_voice  = SCRATCH_BATCH_PER_VOICE,
                models_dir            = weights_dir,
                custom_midi_dir       = CUSTOM_MIDI_DIR,
                arch                  = arch,
                role_conditioned      = rc,
                lstm_hidden_size      = LSTM_HIDDEN_SIZE,
                rnn_type              = rnn_type,
            )
    finally:
        if saved is not None:
            _determinism_restore(saved)
    elapsed = time.time() - t0

    # The single write point — after the two modes converge.
    # harmonize() writes once internally (at which point the title is not built yet, and its
    # signature does not take a title), and the scratch branch's write moved here too, so
    # both modes share one "set metadata + write file". The cost is that harmonize writes
    # twice; the gain is that "is there a title" no longer depends on which branch ran.
    _set_score_title(score, run_title)
    score.write("musicxml", fp=out_path)
    print(f"  标题   : {run_title}", flush=True)

    # =========================================================================
    # 5 — what actually came out
    # =========================================================================
    section("5 — 结果")

    # Stats come from the returned score, without re-parsing the xml.
    dataset = db.build_dataset(custom_midi_dir=CUSTOM_MIDI_DIR)
    _, stats = db.analysis.score_stats(score, dataset)

    # Two md5s answer two questions. raw is "this file"; canonical strips the random part id
    # music21 regenerates on every write, and is "this music" — comparing two generations
    # can only use the latter, since raw always says "different".
    raw_md5 = db.analysis.md5_file(out_path)
    can_md5 = db.analysis.canonical_md5(out_path)
    print(f"  {OK} {out_path}", flush=True)
    print(f"      {os.path.getsize(out_path):,} B   {elapsed:.1f}s", flush=True)
    print(f"      md5（文件）        {raw_md5}", flush=True)
    print(f"      md5（canonical）   {can_md5}   ← 比两次生成是否一致用这个",
          flush=True)

    print(f"\n  逐声部（密度 = 音符数 / tick 数，%）", flush=True)
    print(f"  {'声部':<6}{'音数':>7}{'密度%':>9}{'音域':>12}{'不同音高':>10}"
          f"{'训练音域内%':>13}", flush=True)
    for s in stats:
        lo_hi = "{}–{}".format(*s["range"]) if s["range"] else "—"
        print(f"  v{s['voice']:<5}{s['notes']:>7}{s['density_pct']:>9.1f}"
              f"{lo_hi:>12}{s['distinct']:>10}"
              f"{s['in_trained_range_pct']:>13.1f}", flush=True)

    print(f"\n  真 Bach 众赞歌参照       v0 {DENS_BACH[0]}  v1 {DENS_BACH[1]}  "
          f"v2 {DENS_BACH[2]}  v3 {DENS_BACH[3]}", flush=True)
    deltas = [round(stats[v]["density_pct"] - DENS_BACH[v], 1)
              for v in range(min(4, len(stats)))]
    print(f"  偏差（正 = 更密）        {deltas}", flush=True)
    print("  （参照是四声部众赞歌，口径未必同你的曲子；看的是生成声部之间"
          "是否失衡）", flush=True)

    # Melody preservation is meaningful under harmonize only, and only when the melody
    # really is the voice compared. melody_preservation compares "the notes passed in", so
    # the melody_voice part must be taken explicitly — the whole score would compare parts[0].
    preservation = None
    if MODE == "harmonize":
        sung = list(score.parts[HARMONIZE_MELODY_VOICE].flatten().notes)
        preservation = db.analysis.melody_preservation(sung, input_notes)
        print(f"\n  旋律逐音保留: {preservation['exact']}"
              f"   （{preservation['n_matched']}/{preservation['n_input']} 对上"
              f"，少 {len(preservation['lost'])}，多 {len(preservation['extra'])}）",
              flush=True)
        if preservation["lost"][:4]:
            print(f"    丢失: {preservation['lost'][:4]}", flush=True)
        if preservation["extra"][:4]:
            print(f"    多出: {preservation['extra'][:4]}", flush=True)

    # =========================================================================
    # 6 — manifest: writing "who generated this xml" into a file
    # =========================================================================
    manifest_path = os.path.join(OUTPUT_DIR, f"{tag}.json")
    manifest = {
        "mode": MODE,
        # The **effective** values of each axis (after the command-line override); what is
        # written in CONFIG is only the default. `axes_from_cli` records which came from
        # the command line.
        "arch": arch,
        "role_conditioned": rc,
        # The baseline's recurrent cell. Part of the model's identity (the checkpoint
        # filename carries a trailing `,gru`), and without it the manifest cannot say which
        # of two same-shaped baselines produced this xml.
        "rnn_type": rnn_type,
        "snapshot": snapshot,
        # The tree axis. `data_set` is the one asked for; `weights_tree` is what the
        # resolved dir actually says (`_tree_leaf`), and the two differ only under
        # `--models-dir`.
        "data_set": data_set,
        "weights_tree": tree_leaf,
        "tree_dir": TREE_DIR,
        # The title written into the score itself. Two runs of the same configuration are
        # character-identical (no timestamp).
        "score_title": run_title,
        "axes_from_cli": from_cli,
        "weights_label": weights_label,
        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
        "models_dir": weights_dir,
        "models_dir_source": source,
        "models_dir_from_override": bool(override),
        "weights_loaded": [
            {"name": f["name"], "bytes": f["bytes"], "mtime": f["mtime"],
             "md5": f["md5"]}
            for f in signature["files"] if f["name"] in loader_pick],
        "weights_present_files": status.get("present_files", []),
        "weights_present_dirs": status.get("present_dirs", []),
        "melody": ({"path": melody_path, "md5": db.analysis.md5_file(melody_path),
                    "notes": len(input_notes)} if melody_path else None),
        "output": {"path": out_path, "bytes": os.path.getsize(out_path),
                   "md5": raw_md5, "md5_canonical": can_md5},
        "params": ({
            "num_iterations": HARMONIZE_ITERATIONS,
            "temperature": HARMONIZE_TEMPERATURE,
            "batch_size_per_voice": HARMONIZE_BATCH_PER_VOICE,
            "melody_voice": HARMONIZE_MELODY_VOICE,
            "keep_melody": HARMONIZE_KEEP_MELODY,
            "derive_fermatas": DERIVE_FERMATAS,
            "fermata_marks": FERMATA_MARKS,
        } if MODE == "harmonize" else {
            "sequence_length_ticks": SCRATCH_SEQUENCE_TICKS,
            "num_iterations": SCRATCH_ITERATIONS,
            "temperature": SCRATCH_TEMPERATURE,
            "batch_size_per_voice": SCRATCH_BATCH_PER_VOICE,
        }),
        "seed": seed,
        "seed_drawn_per_run": bool(seed_was_random),
        "deterministic": bool(DETERMINISTIC),
        "custom_midi_dir": CUSTOM_MIDI_DIR,
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # The tree's cache dir written directly, not read back from `db.DATASET_CACHE_DIR`:
        # that global equals this path only because `_install_tree_cache` bound it.
        "dataset_cache_dir": TREE_CACHE_DIR,
        "voice_stats": stats,
        "dens_bach": DENS_BACH,
        "density_delta_vs_bach": deltas,
        "melody_preservation": preservation,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\n  清单: {manifest_path}", flush=True)
    print(f"  （权重 md5 + 全部参数 + 种子都在里面，"
          f"日后问「这份 xml 是谁生成的」看这个文件）", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
