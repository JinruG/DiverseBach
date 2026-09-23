import os
from itertools import islice

import music21
import torch

from music21 import note, harmony, expressions

# constants
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
REST_SYMBOL = 'rest'
OUT_OF_RANGE = 'OOR'
PAD_SYMBOL = 'XX'


# ---------------------------------------------------------------------------
# Data directory and checkpoint snapshot layout -- defined only here
# ---------------------------------------------------------------------------
#
# It lives here rather than in `deepbach_pytorch/__init__.py`: `__init__` imports
# `DeepBach.shared_trunk_model`, and the reverse import would be a cycle. This
# module is already a leaf used by both sides.
#
# Snapshot directory layout:
#
#     <tree>/data/models_<snapshot>/<arch>/<the model's repr>
#
# The same weight filename shows up in every snapshot directory because
# `VoiceModel.__repr__` / `SharedTrunkModel.__repr__` **do not contain the
# directory name** -- the directory is a location, not an identity. Conversely,
# different architectures must never share a directory: weight loading picks
# files by `endswith` + `os.listdir` order, and two candidates turn the model
# choice into an accident of the filesystem.
PACKAGE_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR          = os.path.join(PACKAGE_ROOT, 'data')
DATASET_CACHE_DIR = os.path.join(DATA_DIR, 'dataset_cache')

# The two architectures. `deepbach_pytorch.ARCHS` merely aliases this tuple, it
# is not a copy.
MODEL_ARCHS = ('baseline', 'shared_trunk')

# Every snapshot name, and **which one is loaded by default**.
# `best` plus `epoch01..epoch20`, one per pass.
# The two-digit zero padding is necessary: it makes sorting give chronological
# order, and it matches the earlier `epoch05`/`epoch10`/`epoch15`/`epoch20`
# spellings, which `sibling_snapshot_dir`'s segment test relies on.
# Must agree with the keys of `training_state.PASS_SNAPSHOTS` (an assert there
# guards it).
SNAPSHOT_NAMES   = ('best',) + tuple(f'epoch{n:02d}' for n in range(1, 21))
DEFAULT_SNAPSHOT = 'epoch20'


def snapshot_dir(snapshot=DEFAULT_SNAPSHOT, arch='baseline'):
    """
    The checkpoint directory (absolute path) of one snapshot, one arch.

    :param snapshot: one of `SNAPSHOT_NAMES`
    :param arch: one of `MODEL_ARCHS`
    """
    if snapshot not in SNAPSHOT_NAMES:
        raise ValueError(
            f'unknown snapshot {snapshot!r}; expected one of '
            f'{SNAPSHOT_NAMES}')
    if arch not in MODEL_ARCHS:
        raise ValueError(
            f'unknown arch {arch!r}; expected one of {MODEL_ARCHS}')
    return os.path.join(DATA_DIR, f'models_{snapshot}', arch)


def sibling_snapshot_dir(models_dir, snapshot):
    """
    The directory of another snapshot, derived from **the default snapshot's
    directory**.

    The standard layout is `<root>/models_<snapshot>/<arch>`, so switching
    snapshot means switching that middle segment, and the `models_dir` passed in
    is read as "the location of the default snapshot": if it matches the pattern
    the segment is swapped, if it does not (a custom directory such as
    `data/models_voice0`, used by voice-0) it degrades to the suffix
    `<models_dir>__<snapshot>`.

    Why the two must be kept apart: weight loading picks files by `endswith` +
    `os.listdir` order, and the suffix **does not contain the dataset name**, so
    two differently configured models land in the same directory and the model
    choice becomes an accident of the filesystem.
    """
    if snapshot not in SNAPSHOT_NAMES:
        raise ValueError(
            f'unknown snapshot {snapshot!r}; expected one of {SNAPSHOT_NAMES}')

    head, arch_component = os.path.split(models_dir)
    parent, leaf = os.path.split(head)
    if leaf.startswith('models_') and leaf[len('models_'):] in SNAPSHOT_NAMES:
        if snapshot == leaf[len('models_'):]:
            return models_dir
        return os.path.join(parent, f'models_{snapshot}', arch_component)

    if snapshot == DEFAULT_SNAPSHOT:
        return models_dir
    return f'{models_dir}__{snapshot}'


def atomic_torch_save(obj, path):
    """
    A `torch.save` that never leaves a half-written file at `path`.

    Every caller reads back with `torch.load` and decides whether to **rebuild**
    from `os.path.exists`, so a truncated file is the worst case: it exists, so
    nobody rebuilds, and `torch.load` refuses it.

    The trick is to save `path + '.tmp'` first and then rename; `os.replace` is a
    same-volume rename and therefore atomic: a reader sees either the old file or
    the complete new one. `fsync` makes this hold across a **reboot** too, not
    just a process crash.

    :param obj: anything `torch.save` accepts
    :param path: the final path; its parent directory must already exist
    """
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'wb') as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # leave no debris, then let the real failure propagate.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def strip_fermatas(score):
    """
    Remove every Fermata mark from a music21 score.

    A Fermata is a **conditioning input** for DeepBach (a metadata channel
    marking phrase ends), and `tensor_to_score` also draws that channel onto the
    top voice of a generated score. On an imported melody the channel is
    something the user's file already had, it is not part of the generated
    music, and it is usually not wanted in the output. Callers clear it before
    writing the file.

    :param score: a music21 stream.Score (modified in place)
    :return: how many marks were removed
    """
    removed = 0
    for part in score.parts:
        for element in part.flatten().notes:
            keep = [e for e in element.expressions
                    if not isinstance(e, expressions.Fermata)]
            removed += len(element.expressions) - len(keep)
            element.expressions = keep
    return removed


def standard_name(note_or_rest, voice_range=None):
    """
    Convert a music21 object to a str
    :param note_or_rest:
    :return:
    """
    if isinstance(note_or_rest, note.Note):
        if voice_range is not None:
            min_pitch, max_pitch = voice_range
            pitch = note_or_rest.pitch.midi
            if pitch < min_pitch or pitch > max_pitch:
                return OUT_OF_RANGE
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, note.Rest):
        return note_or_rest.name  # == 'rest' := REST_SYMBOL
    if isinstance(note_or_rest, str):
        return note_or_rest

    if isinstance(note_or_rest, harmony.ChordSymbol):
        return note_or_rest.figure
    if isinstance(note_or_rest, expressions.TextExpression):
        return note_or_rest.content


def standard_note(note_or_rest_string):
    """
    Convert a str that represents a music21 object back into that object
    :param note_or_rest_string:
    :return:
    """
    if note_or_rest_string == 'rest':
        return note.Rest()
    # every other extra symbol is treated as a rest
    elif (note_or_rest_string == END_SYMBOL
          or
          note_or_rest_string == START_SYMBOL
          or
          note_or_rest_string == PAD_SYMBOL):
        # print('Warning: Special symbol is used in standard_note')
        return note.Rest()
    elif note_or_rest_string == SLUR_SYMBOL:
        # print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    elif note_or_rest_string == OUT_OF_RANGE:
        # print('Warning: OUT_OF_RANGE used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


class ShortChoraleIteratorGen:
    """
    For debugging: calling it returns an iterator over 3 Bach chorales,
    like music21.corpus.chorales.Iterator()
    """

    def __init__(self):
        pass

    def __call__(self):
        it = (
            chorale
            for chorale in
            islice(music21.corpus.chorales.Iterator(), 3)
        )
        return it.__iter__()
