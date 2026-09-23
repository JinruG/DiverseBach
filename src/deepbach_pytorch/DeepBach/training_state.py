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

from DatasetManager.helpers import (atomic_torch_save, SNAPSHOT_NAMES,
                                    sibling_snapshot_dir)

# Default budget: 20 complete passes. 20 is defined here, once.
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

# Fields of each `history` row. A dict rather than a tuple, so a new field costs no index counting.
HISTORY_FIELDS = ('global_step', 'epoch', 'train_loss', 'val_loss',
                  'val_acc', 'lr')


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


def make_history_row(global_step, epoch, train_loss, val_loss, val_acc, lr):
    """One training history row. Field names in `HISTORY_FIELDS`."""
    return dict(zip(HISTORY_FIELDS,
                    (int(global_step), int(epoch), float(train_loss),
                     float(val_loss), float(val_acc), float(lr))))


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
HISTORY_FORMAT = 1


def history_path(models_dir, model_repr):
    """One model's per-pass history path; `model_repr` is both identity and filename."""
    return os.path.join(models_dir, _HISTORY_DIRNAME, model_repr)


def save_history(model, models_dir, *, batch_size, steps_per_epoch, max_steps,
                 global_step, best_val_loss, history):
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
    }, path)
    print(f'  [history] {len(history)} pass(es) -> {_ascii(path)}')
