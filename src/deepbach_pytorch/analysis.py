"""
analysis.py - measurement code: tensors / paths in, dicts out.  Read-only: no printing, no
writing to disk.

Division of labour:

    analysis.py     measurement - teacher-forcing metrics, generation stats, weight dirs
    selfcheck.py    assertions - invariants of this install / checkpoints / caches

Every `print` stays with the caller.

`db.analysis` / `db.selfcheck` are lazy attributes (PEP 562, see `__init__.__getattr__`),
so after `import deepbach_pytorch as db` you can call `db.analysis.flat_metrics(...)`
directly.
"""

import contextlib
import hashlib
import io
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import deepbach_pytorch as db
from .DeepBach.data_utils import reverse_tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Vocab tokens that are not real notes: rest / tie placeholder / window padding / OOR.
#: Excluded when measuring onset accuracy, otherwise that number mostly measures the
#: model's ability to guess "continuation".
SPECIAL = frozenset({"rest", "__", "START", "END", "OOR", "XX"})

#: Number of ECE bins.
N_BINS_DEFAULT = 15

#: Default number of sampled windows for holdout evaluation (drawn at random from the
#: validation span).
EVAL_WINDOWS = 40000

#: Forward batch size for evaluation.  Unlike training this is forward only, so the
#: bottleneck is speed, not VRAM.
EVAL_BATCH = 256

#: Index of the central tick of a 32-tick window (sequences_size=8, subdivision=4, so the
#: centre is tick 16).
CENTER = 16

#: Onset density (%) of the inner voices in real Bach chorales; the basis for "generated
#: too dense / too sparse".
DENS_BACH = [23.4, 27.5, 28.1, 29.0]


# ---------------------------------------------------------------------------
# Vocab / masks
# ---------------------------------------------------------------------------

def pitch_index_sets(dataset):
    """
    Per voice: the vocab indices that decode to real notes (rests / tie tokens excluded).

    Returns 4 sets, in the same order as `dataset.index2note_dicts`.  Can be handed
    straight to `is_pitch_masks`, or `list(...)`ed to build masks yourself.
    """
    return [{i for i, token in mapping.items() if token not in SPECIAL}
            for mapping in dataset.index2note_dicts]


def is_pitch_masks(dataset, device='cpu'):
    """
    Per-voice bool tensor: `mask[token_id]` is True when that token is a real note.

    Used by `flat_metrics` / `eval_*` / `teacher_forced_accuracy`.  The device is the
    caller's choice — those functions move the mask to the labels' device internally, so
    either works.
    """
    masks = []
    for voice, indexes in enumerate(pitch_index_sets(dataset)):
        mask = torch.zeros(len(dataset.note2index_dicts[voice]), dtype=torch.bool)
        mask[list(indexes)] = True
        masks.append(mask.to(device))
    return masks


# ---------------------------------------------------------------------------
# Per-batch metrics and accumulators
# ---------------------------------------------------------------------------

def flat_metrics(logits, labels, is_pitch, n_bins=N_BINS_DEFAULT):
    """
    All metrics for one forward batch and one voice, returned as **unnormalised sums**.

    :param logits: (N, vocab) output before softmax
    :param labels: (N,) true token ids
    :param is_pitch: (vocab,) bool tensor, see `is_pitch_masks`; its device need not match
        `labels`, it is aligned internally
    :param n_bins: number of ECE bins
    :return: dict; accumulated values end in `_sum`, extremes start with `max_`

    Sums rather than means: `merge` has to accumulate any number of batches, so the
    division happens once, in `finalize`.
    """
    logits = logits.float()
    # Device alignment happens once, here, so callers need not remember it.
    is_pitch = is_pitch.to(labels.device)

    nll = float(F.cross_entropy(logits, labels, reduction="sum"))
    prob = F.softmax(logits, dim=1)
    conf, pred = prob.max(1)
    correct = (pred == labels)
    entropy = float((-(prob * torch.log(prob.clamp_min(1e-12))).sum(1)).sum())

    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1, device=conf.device)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # First bin closed on the left (a confidence of exactly 0 must not be dropped), the
        # rest open on the left, or boundary samples get counted twice.
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if int(m.sum()) == 0:
            continue
        ece += float(m.sum()) * abs(float(conf[m].mean())
                                    - float(correct[m].float().mean()))

    # Tail counts: NLL is a mean and is dominated by the tail, so "more accurate" and
    # "higher loss" can hold at the same time.
    true_p = prob.gather(1, labels[:, None]).squeeze(1)
    snll = -torch.log(true_p.clamp_min(1e-30))
    return {
        "n": int(labels.numel()), "nll_sum": nll,
        "acc_sum": float(correct.sum()),
        "on_acc_sum": float((correct & is_pitch[labels]).sum()),
        "on_n": int(is_pitch[labels].sum()),
        "conf_sum": float(conf.sum()), "ent_sum": entropy, "ece_sum": ece,
        "max_snll": float(snll.max()),
        "n_snll_5": int((snll > 5).sum()),
        "n_snll_10": int((snll > 10).sum()),
        "tiny3": int((true_p < 1e-3).sum()),
        "tiny6": int((true_p < 1e-6).sum()),
    }


#: The keys `merge` accumulates.  `max_snll` is handled separately (max, not sum).
_ACC_KEYS = ("n", "nll_sum", "acc_sum", "on_acc_sum", "on_n", "conf_sum",
             "ent_sum", "ece_sum", "n_snll_5", "n_snll_10", "tiny3", "tiny6")


def new_acc():
    """An empty accumulator.  Used in tandem with `merge` / `finalize`."""
    acc = {key: 0 for key in _ACC_KEYS}
    acc["nll_sum"] = 0.0
    acc["max_snll"] = 0.0
    return acc


def merge(acc, metrics):
    """Merge one `flat_metrics` result into the accumulator (in place, and returned)."""
    for key in _ACC_KEYS:
        acc[key] += metrics[key]
    acc["max_snll"] = max(acc["max_snll"], metrics["max_snll"])
    return acc


def finalize(acc):
    """Accumulator -> readable means.  The only place that divides."""
    n = acc["n"]
    on = max(acc["on_n"], 1)
    if n == 0:
        raise ValueError("empty accumulator: merge at least one batch first")
    return {
        "n": n,
        "nll": acc["nll_sum"] / n,
        "acc": 100.0 * acc["acc_sum"] / n,
        "on_acc": 100.0 * acc["on_acc_sum"] / on,
        "on_pct": 100.0 * acc["on_n"] / n,
        "conf": 100.0 * acc["conf_sum"] / n,
        "entropy": acc["ent_sum"] / n,
        "ece": 100.0 * acc["ece_sum"] / n,
        "max_snll": acc["max_snll"],
        "pct_snll_5": 100.0 * acc["n_snll_5"] / n,
        "pct_snll_10": 100.0 * acc["n_snll_10"] / n,
        "pct_true_lt_1e-3": 100.0 * acc["tiny3"] / n,
        "pct_true_lt_1e-6": 100.0 * acc["tiny6"] / n,
    }


def mean_over_voices(per_voice):
    """
    `{0: metrics, 1: metrics, ...}` -> the mean of each numeric key.

    Only keys present for all voices are averaged, so any single voice's dict shows the
    full key set.
    """
    voices = sorted(per_voice)
    keys = [k for k, v in per_voice[voices[0]].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([per_voice[v][k] for v in voices])) for k in keys}


# ---------------------------------------------------------------------------
# Holdout windows and batches
# ---------------------------------------------------------------------------

def eval_windows(n_total, n_windows, lo=0.85, hi=0.95, seed=0):
    """
    Draw `n_windows` window indices at random from the holdout set, sorted.

    The default takes the validation span (85%..95%): training uses the first 85%,
    evaluation the last 5%, and this is the 10% in between.  The split ratios are the
    caller's, via `lo` / `hi`, not hardcoded.

    Uses its own seeded `RandomState` and never touches the global RNG, or two calls in one
    process would draw different windows.
    """
    lo_i, hi_i = int(lo * n_total), int(hi * n_total)
    available = hi_i - lo_i
    if available <= 0:
        raise ValueError(
            f"empty holdout range: n_total={n_total}, lo={lo}, hi={hi} "
            f"gives [{lo_i}, {hi_i})")
    size = min(int(n_windows), available)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(np.arange(lo_i, hi_i), size=size, replace=False))


def build_eval_batches(tensor_dataset, indices, center=CENTER,
                       batch_size=EVAL_BATCH, device='cpu'):
    """
    Cut window indices into `(tensor_chorale, tensor_metadata, central_column)` triples.

    `central_column` = `tensor_chorale[:, :, center]`, the whole column to be predicted: it
    carries all four voices' labels, so one transfer covers them.

    `tensor_dataset` is `ChoraleDataset.tensor_dataset` (a TensorDataset); `indices` is
    `eval_windows`'s return value or any integer sequence.
    """
    if not torch.is_tensor(indices):
        indices = torch.from_numpy(np.asarray(indices))
    indices = indices.long()

    batches = []
    for start in range(0, indices.numel(), batch_size):
        sel = indices[start:start + batch_size]
        tensor_chorale = tensor_dataset.tensors[0][sel].long().to(device)
        tensor_metadata = tensor_dataset.tensors[1][sel].long().to(device)
        batches.append((tensor_chorale, tensor_metadata,
                        tensor_chorale[:, :, center]))
    return batches


# ---------------------------------------------------------------------------
# Teacher forcing: one version per architecture
# ---------------------------------------------------------------------------

def eval_baseline(model, weights, batches, is_pitch):
    """
    Per-voice metrics of the baseline (four independent `VoiceModel`s) on the holdout
    batches.

    :param model: `DeepBach`
    :param weights: `{voice: state_dict}`
    :param is_pitch: the result of `is_pitch_masks(...)`
    :return: `{voice: finalize(...)}`

    Mutates the model in place (loads weights + eval): what is being scored is the
    **weights** themselves, not the model instance.
    """
    for voice in range(len(model.voice_models)):
        model.voice_models[voice].load_state_dict(weights[voice])
        model.voice_models[voice].eval()

    out = {}
    for voice in range(len(model.voice_models)):
        acc = new_acc()
        with torch.no_grad():
            for tensor_chorale, tensor_metadata, central in batches:
                vm = model.voice_models[voice]
                notes, _ = vm.preprocess_notes(tensor_chorale=tensor_chorale,
                                               time_index_ticks=CENTER)
                metas = vm.preprocess_metas(tensor_metadata=tensor_metadata,
                                            time_index_ticks=CENTER)
                logits = vm.forward(notes, metas)
                merge(acc, flat_metrics(logits, central[:, voice], is_pitch[voice]))
        out[voice] = finalize(acc)
    return out


def eval_shared(model, state, batches, is_pitch):
    """
    Per-role metrics of the shared trunk (scheme A) on the holdout batches.

    :param model: `SharedTrunkDeepBach`
    :param state: one state_dict — shared by all four roles
    :return: `{role: finalize(...)}`

    **This is a shortcut, not the generation path**: it hand-assembles the left and right
    tensors for `forward_all`, bypassing `RoleView` entirely.  Its validity rests on two
    identities (asserted by `selfcheck.check_preprocessing_equivalence`): the cross-voice
    conditioning metadata channels are elementwise identical (so `tm[:, 0]` equals
    `tm[:, r]`), and `RoleView.forward` is elementwise identical to `forward_all(...)[r]`.

    **With `role_conditioned=True` this shortcut is invalid** and raises `ValueError`: the
    trunk then consumes the `voice_id` channel and this code takes `tm[:, 0]` for all four
    roles, which amounts to feeding voice_id=0 to every head.

    The `RoleView` version is `teacher_forced_accuracy`: it applies to both architectures,
    is slower, but every step goes down the real path.
    """
    if getattr(model.model, 'role_conditioned', False):
        raise ValueError(
            "eval_shared() hardcodes tensor_metadata[:, 0] for all four roles, "
            "which is only equivalent to tensor_metadata[:, r] when the trunk "
            "does not consume voice_id; this model has role_conditioned=True. "
            "Use teacher_forced_accuracy(), or rebuild with "
            "role_conditioned=False.")
    model.model.load_state_dict(state)
    model.model.eval()

    out = {}
    for role in range(len(model.voice_models)):
        acc = new_acc()
        with torch.no_grad():
            for tensor_chorale, tensor_metadata, central in batches:
                left = tensor_chorale[:, :, :CENTER]
                right = reverse_tensor(tensor_chorale[:, :, CENTER + 1:], dim=2)
                left_metas = tensor_metadata[:, 0, :CENTER, :]
                right_metas = reverse_tensor(
                    tensor_metadata[:, 0, CENTER + 1:, :], dim=1)
                logits = model.model.forward_all(
                    left, tensor_chorale[:, :, CENTER], right,
                    (left_metas, tensor_metadata[:, 0, CENTER, :],
                     right_metas))[role]
                merge(acc, flat_metrics(logits, central[:, role], is_pitch[role]))
        out[role] = finalize(acc)
    return out


def teacher_forced_accuracy(model, batches, is_pitch, center=CENTER):
    """
    Teacher-forcing accuracy down the real inference path: per voice `preprocess_*` +
    `forward`.

    Applies to the baseline and the shared trunk alike — the shared trunk's
    `voice_models[r]` is a `RoleView` with the same three methods as `VoiceModel`.  It is
    also the control for the `eval_shared` shortcut: on the same windows the two must agree
    to within floating-point error, and a disagreement means the preprocessing paths have
    diverged.

    :return: dict with
        acc / on_acc       - per-voice accuracy (%) over all ticks / onset ticks only
        const_all/const_on - constant-prediction baseline: each voice always guesses its
                             own most frequent token
        on_pct             - onset ticks as a share of all ticks
        n_windows / n_all / n_on

    The all-ticks row is mostly guessing ties (onsets are only 20%..30%), so to judge
    whether the model has learned anything, read `on_acc` against `const_on`.
    """
    n_voices = len(is_pitch)
    masks = [m.detach().cpu() for m in is_pitch]

    hits = np.zeros(n_voices)
    tot = np.zeros(n_voices)
    ohits = np.zeros(n_voices)
    otot = np.zeros(n_voices)
    tok_all = [np.zeros(int(m.numel()), dtype=np.int64) for m in masks]
    tok_on = [np.zeros_like(counts) for counts in tok_all]
    n_all = np.zeros(n_voices, dtype=np.int64)
    n_on = np.zeros(n_voices, dtype=np.int64)
    n_windows = 0

    with torch.no_grad():
        for tensor_chorale, tensor_metadata, central in batches:
            n_windows += int(tensor_chorale.shape[0])
            for voice in range(n_voices):
                labels = central[:, voice].detach().cpu()
                on = masks[voice][labels].numpy()
                tok_all[voice] += np.bincount(labels.numpy(),
                                              minlength=tok_all[voice].size)
                tok_on[voice] += np.bincount(labels.numpy()[on],
                                             minlength=tok_on[voice].size)
                n_all[voice] += labels.numel()
                n_on[voice] += int(on.sum())

            for role in range(n_voices):
                vm = model.voice_models[role]
                notes, _ = vm.preprocess_notes(tensor_chorale=tensor_chorale,
                                               time_index_ticks=center)
                metas = vm.preprocess_metas(tensor_metadata=tensor_metadata,
                                            time_index_ticks=center)
                predicted = vm.forward(notes, metas).max(1)[1]
                labels = central[:, role]
                on = is_pitch[role].to(labels.device)[labels]
                hits[role] += float((predicted == labels).sum())
                tot[role] += labels.numel()
                ohits[role] += float(((predicted == labels) & on).sum())
                otot[role] += float(on.sum())

    acc = hits / np.maximum(tot, 1) * 100.0
    on_acc = ohits / np.maximum(otot, 1) * 100.0
    const_all = np.array([tok_all[v].max() / max(int(tok_all[v].sum()), 1) * 100.0
                          for v in range(n_voices)])
    const_on = np.array([tok_on[v].max() / max(int(tok_on[v].sum()), 1) * 100.0
                         for v in range(n_voices)])
    return {
        "acc": [float(x) for x in acc],
        "on_acc": [float(x) for x in on_acc],
        "const_all": [float(x) for x in const_all],
        "const_on": [float(x) for x in const_on],
        "acc_mean": float(acc.mean()),
        "on_acc_mean": float(on_acc.mean()),
        "const_on_mean": float(const_on.mean()),
        "on_pct": float(100.0 * n_on.sum() / max(int(n_all.sum()), 1)),
        "n_windows": int(n_windows),
        "n_all": [int(x) for x in n_all],
        "n_on": [int(x) for x in n_on],
    }


# ---------------------------------------------------------------------------
# Generation results
# ---------------------------------------------------------------------------

def _as_note_list(score_or_notes):
    """Path / music21 Score / note list -> note list."""
    obj = score_or_notes
    if isinstance(obj, (str, os.PathLike)):
        import music21
        obj = music21.converter.parse(str(obj))
    if hasattr(obj, "parts") and len(list(obj.parts)) > 0:
        obj = obj.parts[0].flatten().notes
    return list(obj)


def melody_key(notes):
    """
    A per-note fingerprint of the melody: `(midi, offset)`, with offset at 3 decimals.

    The offset has to be in it: `keep_melody=True` means "these notes are still in place",
    and comparing pitches alone would pass a bug that shifted everything by half a beat.
    """
    return [(int(n.pitch.midi), round(float(n.offset), 3))
            for n in _as_note_list(notes)]


def melody_preservation(output, input_notes):
    """
    Whether output voice 0 preserves the input melody verbatim.

    :param output: the generated score (path / Score / note list)
    :param input_notes: the input melody (same forms)
    :return: `{'exact', 'lost', 'extra', 'n_output', 'n_input', 'n_matched'}`

    `lost` / `extra` are set differences after pairing in order, so an `exact=False` points
    at which notes were dropped or added; a bare boolean cannot.
    """
    got = melody_key(output)
    want = melody_key(input_notes)
    matched = sum(1 for a, b in zip(got, want) if a == b)
    return {
        "exact": got == want,
        "lost": [x for x in want if x not in got],
        "extra": [x for x in got if x not in want],
        "n_output": len(got),
        "n_input": len(want),
        "n_matched": matched,
    }


def score_stats(score_or_path, dataset, ticks=None):
    """
    Per-voice statistics of one score: note count, density, range, distinct pitches, and
    the share falling inside the trained range.

    :param ticks: the density denominator (16th-note ticks).  None -> derived from the
        score as `highestTime * dataset.subdivision`
    :return: `(score, stats)`, with one `stats` entry per voice

    Density is measured the same way as `DENS_BACH`: **notes / tick**, i.e. the onset-tick
    share (the database encoding guarantees at most one note onset per tick).
    """
    import music21
    score = (music21.converter.parse(str(score_or_path))
             if isinstance(score_or_path, (str, os.PathLike)) else score_or_path)
    if ticks is None:
        ticks = int(round(float(score.flatten().highestTime)
                          * dataset.subdivision))
    denominator = max(int(ticks), 1)

    stats = []
    for voice, part in enumerate(score.parts):
        notes = list(part.flatten().notes)
        pitches = [int(n.pitch.midi) for n in notes]
        lo, hi = dataset.voice_ranges[voice]
        stats.append({
            "voice": voice,
            "notes": len(notes),
            "density_pct": round(100.0 * len(notes) / denominator, 1),
            "range": (min(pitches), max(pitches)) if pitches else None,
            "distinct": len(set(pitches)),
            "in_trained_range_pct": round(
                100.0 * sum(1 for p in pitches if lo <= p <= hi) / len(pitches), 1)
            if pitches else 0.0,
        })
    return score, stats


# ---------------------------------------------------------------------------
# Files / weight dirs
# ---------------------------------------------------------------------------

def md5_file(path, chunk=1 << 20):
    """File md5, read in chunks so a 2.6 GB cache file stays out of memory."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# music21 mints a fresh random id for each part on every write
# (`<score-part id="P<32 hex>">`), so the raw md5 of the same score written twice is never
# equal — that is a serialisation detail, not a result.
_PART_ID_RE = re.compile(rb'P[0-9a-f]{32}')


def canonical_md5(path):
    """
    The md5 with music21's random part ids wiped out — use this to compare two MusicXML
    files.

    It touches only that one random quantity; notes / durations / bars are untouched.  For
    harder evidence compare the event table as well (part / midi / offset /
    quarterLength): both must pass, because the normalisation itself has an escape hatch —
    if the id format changes the regex stops matching and this function silently degrades
    to the raw md5.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    return hashlib.md5(_PART_ID_RE.sub(b'P#', data)).hexdigest()


def weight_dir_signature(models_dir, expect_files=1):
    """
    A signature of the weight directory: per-file bytes / mtime / md5, plus subdirectory
    names.

    :param expect_files: the expected file count: 1 for the shared trunk, 4 for baseline.
        This is a **check**, not decoration: the loader matches filenames by `endswith`
        and takes the first hit in `os.listdir` order, so one extra file in the directory
        turns the model choice into a filesystem coincidence
    :return: dict with `exists` / `files` (each name+bytes+mtime+md5) / `dirs` /
        `md5s` (name -> md5, so two directories can be compared directly) / `n_files` /
        `as_expected`

    `dirs` holds subdirectories such as `_history/`: normally present (the loss curves sit
    next to the weights) but **not** weight files, so they are listed apart instead of
    mixed into `files`.
    """
    models_dir = os.path.abspath(models_dir)
    if not os.path.isdir(models_dir):
        return {"models_dir": models_dir, "exists": False, "files": [],
                "dirs": [], "md5s": {}, "n_files": 0,
                "expect_files": int(expect_files), "as_expected": False}

    entries = sorted(os.listdir(models_dir))
    names = [f for f in entries if os.path.isfile(os.path.join(models_dir, f))]
    dirs = [f for f in entries if f not in names]

    files, md5s = [], {}
    for name in names:
        path = os.path.join(models_dir, name)
        md5s[name] = md5_file(path)
        files.append({
            "name": name,
            "bytes": os.path.getsize(path),
            "md5": md5s[name],
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(os.path.getmtime(path))),
        })
    return {"models_dir": models_dir, "exists": True, "files": files,
            "dirs": dirs, "md5s": md5s, "n_files": len(names),
            "expect_files": int(expect_files),
            "as_expected": len(names) == int(expect_files)}


def stash(path):
    """
    Rename an existing directory out of the way.  **Never deletes.**

    A same-volume rename is a metadata operation, so even a 2.6 GB cache directory takes
    seconds.  If `path_old` already exists a timestamp suffix is added instead, which would
    otherwise overwrite the previous backup.

    :return: the new path; None when the path does not exist

    This module does not print, so the `[让位] xxx -> yyy` line is left to the caller.
    """
    if not os.path.isdir(path):
        return None
    dst = path + "_old"
    if os.path.exists(dst):
        dst = "{}_{}".format(path, time.strftime("old-%Y%m%d-%H%M%S"))
    os.rename(path, dst)
    return dst


def capture(fn, *args, **kwargs):
    """
    Run `fn`, returning `(result, everything it printed)`.

    Used to assert things like "no WARNING in the log": a shape mismatch while loading
    weights only degrades to a printed warning + zero padding, so without capturing the
    output you can only eyeball it.

    **Covers stdout only**; stderr is invisible, so assertions about it cannot reach stderr.

    `Tee` forwards every other attribute to the real stream via `__getattr__`: `tqdm`
    probes `isatty()` / `fileno()` / `encoding`, and implementing only `write` / `flush`
    would raise AttributeError on the first refresh.
    """
    buffer = io.StringIO()
    real = sys.stdout

    class Tee:
        def write(self, text):
            real.write(text)
            buffer.write(text)

        def flush(self):
            real.flush()

        def __getattr__(self, name):
            return getattr(real, name)

    with contextlib.redirect_stdout(Tee()):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()
