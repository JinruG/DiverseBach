"""
Training budget, pass snapshots, and the per-pass metrics ledger.

The budget counts optimizer steps, not epochs. A "segment" is one complete pass of
`steps_per_epoch` batches; `resolve_training_budget()` rounds `max_steps` up to a
multiple of `steps_per_epoch`, so checkpoints land only on segment boundaries.

Two invariants:

  * `save_pass_snapshot` fires on a **pass count**, `best` on a val-loss improvement.
    The two can fire in the same pass; they write to different directories and do not
    overwrite each other.
  * `save_history` must run **unconditionally at the end of every pass**: it is the
    only writer of the loss curve, and a missed pass leaves no trace.

Every `print` literal in this module is plain ASCII; see `_ascii()`.
"""

import os
import random
import shutil
import statistics

import torch

from DatasetManager.helpers import (atomic_torch_save, SNAPSHOT_NAMES,
                                    sibling_snapshot_dir)

# Default budget: 20 complete passes. 20 is defined here, once.
#
# The matching ladder lives in `DatasetManager.helpers.SNAPSHOT_NAMES`, which cannot be
# *derived* from this number (the import would be a cycle; see the comment there), so the
# two are edited together by hand and the assert below is what proves they still agree.
DEFAULT_TOTAL_PASSES = 20

# End-of-pass snapshot table: after pass N the weights are saved again under
# `models_epochNN/`, so every pass has a directory. Two-digit zero padding is
# required: `epoch10`/`epoch15`/`epoch20` match the old names, and `sorted()` gives
# `epoch01..epoch20` rather than `epoch1, epoch10, ...`. `best` is not in the table —
# it fires on a val-loss improvement. The assert below guards that boundary: table +
# `best` must exactly cover `SNAPSHOT_NAMES`, or write and read directories disagree.
PASS_SNAPSHOTS = {n: f'epoch{n:02d}' for n in range(1, DEFAULT_TOTAL_PASSES + 1)}
assert set(PASS_SNAPSHOTS.values()) | {'best'} == set(SNAPSHOT_NAMES), (
    f'PASS_SNAPSHOTS {PASS_SNAPSHOTS} + best does not cover SNAPSHOT_NAMES '
    f'{SNAPSHOT_NAMES}; the trainer and the loader would disagree about which '
    f'directories exist')


def snapshot_due_at_pass(passes_done):
    """Snapshot name due once `passes_done` passes are complete, else None."""
    return PASS_SNAPSHOTS.get(int(passes_done))


def save_pass_snapshot(model, models_dir, passes_done):
    """
    Save the weights into the snapshot directory due after `passes_done` passes.

    Unrelated to the `best` criterion (this counts passes); the two write to
    different directories and do not overwrite each other.

    :return: the snapshot name written, or None if not yet due
    """
    name = snapshot_due_at_pass(passes_done)
    if name is None:
        return None
    target = sibling_snapshot_dir(models_dir, name)
    model.save(models_dir=target)
    print(f'  [snapshot] pass {passes_done} weights -> {_ascii(target)}')
    return name


def odd_window(window):
    """The `best_window` actually used: widths are odd — an even one has no centre. 4 and 5 both mean 5."""
    window = int(window)
    if window < 1:
        raise ValueError(f'best_window must be >= 1, got {window}')
    return window if window % 2 else window + 1


def windowed_best_pass(val_losses, window=5):
    """
    The 0-based index of the pass with the lowest **windowed** val loss.

    Not the single lowest reading. On a plateau the minimum is usually one isolated dip
    whose neighbours are all worse, and that dip is noise: picking it picks the noise.
    Each pass is scored here by the median of its neighbourhood instead — median rather
    than mean, because the mean is dragged *down* by the very dip under test and would
    keep choosing it. The window is clipped at the two ends of the run, and `window=1`
    gives back the plain argmin (the old rule).

    Ties are the normal case on a plateau (a 3-wide median of `a, b, c` is just `b`, so
    equal readings give equal scores), and they are broken by the pass's own val loss and
    then by pass order: among passes that sit at the same level, the one with the better
    reading wins, and the dip the score was built to ignore is only ever reachable when
    its whole neighbourhood is at that level too.

    Pure, and a function of the curve alone: which pass a finished run should point at
    can be recomputed from `_history/` without the weights.
    """
    losses = [float(v) for v in val_losses]
    if not losses:
        raise ValueError('windowed_best_pass needs at least one pass')
    half = odd_window(window) // 2
    scores = [statistics.median(losses[max(0, p - half):p + half + 1])
              for p in range(len(losses))]
    return min(range(len(losses)), key=lambda p: (scores[p], losses[p], p))


def repoint_best_snapshot(model_repr, models_dir, snapshot):
    """
    Copy `snapshot`'s checkpoint over `best`, byte for byte.

    A file copy rather than a load/save round trip: `best/` then stays md5-identical to
    the ladder entry it came from, so "is `best` this pass?" is a digest comparison and
    not a tolerance. Atomic (temp file + `os.replace`), so a kill mid-copy leaves the
    previous best rather than a truncated file.

    Takes the model's `__repr__` rather than the model, so a finished run's `best` can be
    re-selected from the ladder alone, with no weights loaded and no model built.

    :return: True if copied, False if that snapshot holds no checkpoint for this model
    """
    source = os.path.join(sibling_snapshot_dir(models_dir, snapshot), model_repr)
    if not os.path.isfile(source):
        return False
    target = os.path.join(sibling_snapshot_dir(models_dir, 'best'), model_repr)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp_path = target + '.tmp'
    try:
        with open(source, 'rb') as src, open(tmp_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return True


def repoint_best_by_window(model, models_dir, history, window=5):
    """
    Point `best` at a finished run's lowest windowed-median pass.

    The per-pass rule keeps writing `best` on every new minimum — that stays, so a killed
    run still has a best — and this is what decides the directory's **final** content. It
    runs once, before the run's end-of-run evaluation, so the `test` split is measured on
    the weights the run actually hands over.

    :param history: the run's rows, as written by `make_history_row`
    :return: the snapshot name now in `best`, or None if nothing was copied
    """
    if not history:
        return None
    index = windowed_best_pass([row['val_loss'] for row in history], window)
    snapshot = PASS_SNAPSHOTS[int(history[index]['epoch']) + 1]
    if not repoint_best_snapshot(model.__repr__(), models_dir, snapshot):
        print(f'  [warn] no checkpoint for pass {index + 1} ({snapshot}); `best` '
              f'left at the lowest single-pass val loss')
        return None
    print(f'  [best] pass {index + 1} ({snapshot}) by {odd_window(window)}-wide '
          f'median val loss')
    return snapshot


# Fields of each `history` row. A dict rather than a tuple, so a new field costs no index counting.
#
# `train_acc` / `val_top3` / `val_top5` cost nothing extra: they come out of the same
# forward pass as `train_loss` / `val_loss` / `val_acc`, which is why they are per-pass
# rows here rather than a separate evaluation. `pass_time_s` is the wall time of that
# pass alone, which is what makes `epoch_time_s` in the run metrics a mean of measured
# passes rather than a division.
HISTORY_FIELDS = ('global_step', 'epoch', 'train_loss', 'train_acc', 'val_loss',
                  'val_acc', 'val_top3', 'val_top5', 'lr', 'pass_time_s')


def resolve_training_budget(steps_per_epoch=None, max_steps=None,
                            len_train_loader=None):
    """
    Fix the two budget numbers and round `max_steps` up to a whole segment.

    The repo's only place doing this arithmetic; both `train_model`s call it.
    Rounding up because segment boundaries are the checkpoint points — a half segment
    would leave the snapshot points off whole passes.

    This only does the arithmetic; truncating a pass to `steps_per_epoch` happens in
    `loss_and_acc(..., steps=steps_per_epoch)`, and neither half works alone.

    :param steps_per_epoch: None means the batch count of one complete pass
    :param max_steps: None means `DEFAULT_TOTAL_PASSES * steps_per_epoch`
    :param len_train_loader: `len(dataloader_train)`, used when the above is None;
        also the reachability check on an explicit `steps_per_epoch`
    :return: (steps_per_epoch, max_steps, total_passes), the last two in effect
    """
    if steps_per_epoch is None:
        if len_train_loader is None:
            raise ValueError(
                'steps_per_epoch is None and len_train_loader was not given; '
                'pass one of them')
        steps_per_epoch = int(len_train_loader)
    steps_per_epoch = int(steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError(
            f'steps_per_epoch must be positive, got {steps_per_epoch}; '
            f'a zero-length train loader means the split or batch_size is '
            f'wrong (drop_last=True drops a partial batch)')

    if max_steps is None:
        max_steps = DEFAULT_TOTAL_PASSES * steps_per_epoch
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f'max_steps must be positive, got {max_steps}')

    total_passes = -(-max_steps // steps_per_epoch)   # ceil

    # One pass runs at most len(dataloader) batches. Asking for more means no pass
    # reaches `steps_per_epoch`, `global_step` runs low and never reaches `max_steps`,
    # while training itself carries on. The only source is a caller writing
    # `steps_per_epoch` longer than a real pass: a configuration error.
    if len_train_loader is not None and steps_per_epoch > int(len_train_loader):
        print(f'  [warn] steps_per_epoch={steps_per_epoch} exceeds one real pass '
              f'({int(len_train_loader)} batch(es)); every pass will record fewer '
              f'steps than that and global_step will stop short of '
              f'max_steps={total_passes * steps_per_epoch}.')

    return steps_per_epoch, total_passes * steps_per_epoch, total_passes


def seed_everything(seed):
    """
    Seed `random` and torch (CPU + all CUDA devices), or do nothing when `seed` is None.

    **None is the default, not 0.** Seeding changes the shuffle order of every pass and
    the per-window tick offset in `preprocess_input`, so a seeded run is a different run
    from an unseeded one — the two must not be compared as though only the seed differed.
    The value in effect is stored in the history payload, so a curve's seed is never a
    guess afterwards.

    `random` and not only torch: the tick offset above is `random.randint`.

    This is not a determinism guarantee. cuDNN's recurrent backward pass uses atomics, so
    two identically-seeded runs still differ in their weights by ~1e-3 (README,
    "Training always starts from scratch").

    :return: the seed in effect (an int), or None
    """
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def make_history_row(global_step, epoch, train_loss, val_loss, val_acc, lr, *,
                     train_acc, val_top3, val_top5, pass_time_s):
    """
    One training history row. Field names in `HISTORY_FIELDS`.

    The first six stay positional (the order the curves have always been written in); the
    four measured alongside them are keyword-only and **required**, so a call site that
    predates them raises `TypeError` instead of writing a row with a silent hole in it.
    """
    return dict(zip(HISTORY_FIELDS,
                    (int(global_step), int(epoch), float(train_loss),
                     float(train_acc), float(val_loss), float(val_acc),
                     float(val_top3), float(val_top5), float(lr),
                     float(pass_time_s))))


def _ascii(text):
    """
    Squeeze arbitrary text to plain ASCII, for printing only.

    With stdout redirected Python takes the locale encoding (gbk here), and a
    character it cannot encode raises `UnicodeEncodeError` — inside the training loop.
    Model `repr`s, file paths and exception messages are not ours to choose, so they
    all pass this gate: what will not encode becomes `\\uXXXX`, ugly but always
    printable.
    """
    return str(text).encode('ascii', 'backslashreplace').decode('ascii')


# ---------------------------------------------------------------------------
#  History ledger — the loss curve, and its only copy
# ---------------------------------------------------------------------------

# `_history/` rather than the models dir top level — a hard constraint:
# `analysis.weight_dir_signature` counts top-level files only and
# `selfcheck.check_weights_dir` asserts that count, so one extra top-level file fails
# self-check. The name ends in `y`, not in the `)` that marks a weight file.
_HISTORY_DIRNAME = '_history'

# Bumped when the payload layout changes; readers go by key, so a mismatch means absent.
#
# v2 added the four measured columns to every row (`train_acc`, `val_top3`, `val_top5`,
# `pass_time_s`) and the run-level `run_metrics` block. A v1 payload read as v2 — or the
# reverse — is not detectable from the payload itself; this is the only marker.
HISTORY_FORMAT = 2


def history_path(models_dir, model_repr):
    """One model's per-pass history path; `model_repr` is both identity and filename."""
    return os.path.join(models_dir, _HISTORY_DIRNAME, model_repr)


def save_history(model, models_dir, *, batch_size, steps_per_epoch, max_steps,
                 global_step, best_val_loss, history, metrics=None):
    """
    Write this model's per-pass history to disk. **The only place the loss curve is
    written** (`save()` and `save_pass_snapshot()` write weights only).

    Called at the end of every pass, unlike `save_pass_snapshot`, which fires only at
    snapshot points. The file holds no weights and is rewritten whole each pass, so
    there is no intermediate state.

    The filename is the model's `repr`, which lets the four voices sharing a
    `models_dir` each write their own file (`VoiceModel.__repr__` carries
    `main_voice_index`).

    `batch_size` is stored rather than inferred from the path: the path has the tree
    name, not the batch size.

    `metrics` is the run-level block from `metrics.run_metrics_block` — parameter counts,
    wall times, peak VRAM, the test-split evaluation. Only the **final** call of a run has
    it; the per-pass calls leave the key at None, so "absent" and "this run has not
    finished" are distinguishable. It goes in this payload rather than a file of its own
    because `_history/` is one file per model: a second file there would be read as a
    second curve, and an extra file in `models_dir` fails `selfcheck.check_weights_dir`.

    No try/except: a failed write is a real fault and should end the run.

    :param history: rows from `make_history_row`, one per completed pass
    """
    repr_ = model.__repr__()
    if os.path.sep in repr_ or (os.path.altsep and os.path.altsep in repr_):
        # A repr with a path separator (a custom corpus name) would write outside the history dir.
        raise ValueError(
            f'model repr contains a path separator and cannot be used as a '
            f'history filename: {_ascii(repr_)}')
    path = history_path(models_dir, repr_)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_torch_save({
        'history_format': HISTORY_FORMAT,
        'kind': type(model).__name__,
        'model_repr': repr_,
        'batch_size': int(batch_size),
        'steps_per_epoch': int(steps_per_epoch),
        'max_steps': int(max_steps),
        'global_step': int(global_step),
        'best_val_loss': float(best_val_loss),
        'history': list(history),
        'run_metrics': metrics,
    }, path)
    print(f'  [history] {len(history)} pass(es) -> {_ascii(path)}')
