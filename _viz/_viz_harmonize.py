r"""
Record DeepBach harmonization (Gibbs sampling) as a scrubbable piano roll, by hooking
`DeepBach.parallel_gibbs` -- `harmonize()` drops the process tensors and the package has
no visualization code.

    python _viz\_viz_harmonize.py [--arch baseline|shared_trunk] [--rnn-type lstm|gru]
                                  [--rc true|false] [--dataset data256|data128]

Defaults come from `infer` (`ARCH` / `RNN_TYPE` / `ROLE_CONDITIONED` / `DATA_SET`); the
tree, the snapshot and the weights directory are resolved by `infer`'s own helpers
(`_set_data_set` / `_snapshot_for` / `_weights_dir`) rather than by a copy of that table.
`--help` lists the switches.

The four arms:

    --arch baseline                  <tree>/models_<snap>/baseline/      4 x VoiceModel
    --arch baseline --rnn-type gru   <tree>/models_<snap>/gru/           4 x VoiceModel
    --arch shared_trunk --rc false   <tree>/models_<snap>/shared_trunk/  1 x SharedTrunkModel
    --arch shared_trunk --rc true    <tree>/models_rc/shared_trunk/      1 x SharedTrunkModel

`rnn_type` is an axis on the baseline only; the shared trunk is LSTM. The arm
(`arch` / `rc` / `rnn_type`, plus a ready-made `arm_text`) is written into the payload's
`meta`, which is how the one static page labels all four.

Outputs, all in this script's own directory:

    harmonize.html       the page. Static and hand-maintained; this script never writes it
    frames.json          the payload: a frame every SNAPSHOT_EVERY rounds, plus the random
                         init and the final state, with per-frame per-voice confidence
    frames.js            the same JSON as `window.__FRAMES__ = ...;`, from the same
                         `json.dumps` -- a browser refuses to `fetch` a sibling file under
                         `file://`, and double-clicking the html is the normal way in
    harmonized_voice0.xml, run.log

`--reuse` re-emits `frames.js` from the existing `frames.json` and samples nothing.
"""

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_src(start):
    """Walk up for the dir holding `src/deepbach_pytorch`; return that `src`, else None."""
    d = start
    for _ in range(6):
        cand = os.path.join(d, 'src')
        if os.path.isdir(os.path.join(cand, 'deepbach_pytorch')):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


SRC = _find_src(HERE)
if SRC is None:
    print(f'[Fatal] 从 {HERE} 向上 6 层都没找到 src/deepbach_pytorch')
    print('        这个脚本必须待在仓库里（_viz/ 在仓库根下面一层）。')
    sys.exit(1)
REPO = os.path.dirname(SRC)
SRC_PKG = os.path.join(SRC, 'deepbach_pytorch')

# Both are the **parent dir** of the import: `src` for `deepbach_pytorch`, root for `infer`.
for _p in (REPO, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import argparse                   # noqa: E402
import torch                      # noqa: E402
import deepbach_pytorch as dbp    # noqa: E402
from torch import nn              # noqa: E402
from tqdm import tqdm             # noqa: E402
from deepbach_pytorch.DeepBach.helpers import cuda_variable, to_numpy  # noqa: E402
from deepbach_pytorch.DatasetManager.helpers import (                  # noqa: E402
    standard_note, REST_SYMBOL, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL,
    OUT_OF_RANGE, PAD_SYMBOL)

# The axis defaults and the weights/cache resolution all come from `infer`.
import infer                      # noqa: E402

# ---------------------------------------------------------------------------
# Guard: the import must land on this repo's src tree
# ---------------------------------------------------------------------------
_got = os.path.abspath(dbp.__file__)
if not _got.startswith(os.path.abspath(SRC_PKG) + os.sep):
    print(f'[Fatal] 期望本仓库 src 那份，实际拿到: {_got}')
    print(f'        期望前缀: {SRC_PKG}{os.sep}')
    print('        sys.path 里有东西抢在前面了 —— 装过的 deepbach_pytorch 会赢下裸 import。')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parameters (all overridable; see `_parse_args`)
# ---------------------------------------------------------------------------
MELODY = ('C:/Users/Grud/Documents/' + chr(0x603b) + chr(0x8c31)
          + '/melody/belaa-koska.musicxml')

# None = `infer._snapshot_for` decides from the axes; a spelled-out name wins.
SNAPSHOT = None

# 'lstm' or 'gru' (`db.RNN_TYPES`), the weights leaf `<tree>/models_<snapshot>/<cell>/`.
# None = `infer.RNN_TYPE`. Baseline only; shared_trunk is fixed to 'lstm'.
RNN_TYPE = None

NUM_ITERATIONS = 1700     # Gibbs rounds
TEMPERATURE = 0.3         # requested; the anneal starts at 1.1
BATCH_PER_VOICE = 64      # parallel proposals per round
MELODY_VOICE = 0          # voice index the melody is fixed in
SNAPSHOT_EVERY = 17       # rounds per recorded frame (plus the init and final state)

# Outputs go in the script's own directory; the html is static and hand-maintained.
OUT_DIR = HERE
OUT_XML = os.path.join(OUT_DIR, 'harmonized_voice0.xml')
OUT_JSON = os.path.join(OUT_DIR, 'frames.json')
OUT_JS = os.path.join(OUT_DIR, 'frames.js')

VOICE_NAMES = ['Soprano', 'Alto', 'Tenor', 'Bass']


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class Recorder:
    """Store one full-tensor frame every SNAPSHOT_EVERY rounds."""

    def __init__(self, snapshot_every):
        self.snapshot_every = snapshot_every
        self.frames = []        # [(iteration, np.ndarray (V, L) int16)]
        # One per `frames` row, indexed by voice number; unsampled voices (melody) are None.
        self.conf = []
        self.last_temp = None   # temperature_sa on the final round (measured)
        self.last_conf = None   # the latest round's conf, for `final()`
        self.calls = 0          # call count
        self.model = None
        self.dataset = None
        self.voice_indices = None
        self.ab_inputs = None   # the raw inputs the fidelity A/B needs
        self.started = False

    def begin(self, model, tensor_no_cuda, timesteps_ticks, voice_indices,
              ab_inputs):
        self.model = model
        self.dataset = model.dataset
        self.voice_indices = list(voice_indices)
        self.ab_inputs = ab_inputs
        self.started = True
        # Index -1 = the random init before round 0; `astype` copies, later writes miss it.
        self.frames.append((-1, self._slice(tensor_no_cuda, timesteps_ticks)
                            .astype(np.int16)))
        # All None: no forward yet. Shape matches the later frames.
        self.conf.append([None] * int(self.dataset.num_voices))

    def _slice(self, t, ts):
        return t[0, :, ts:-ts].detach().cpu().numpy()

    def step(self, iteration, temperature_sa, tensor_no_cuda, timesteps_ticks,
             probas):
        self.calls += 1
        self.last_temp = float(temperature_sa)
        conf = self._conf(probas)
        self.last_conf = conf    # kept for `final()`, which adds one more frame
        if iteration % self.snapshot_every == 0:
            self.frames.append((iteration, self._slice(tensor_no_cuda, timesteps_ticks)
                                .astype(np.int16)))
            self.conf.append(conf)

    def final(self, iteration, temperature_sa, tensor_no_cuda, timesteps_ticks):
        """Record the state after the last round.

        `step()` records only when `iteration % snapshot_every == 0`;
        the last round need not be a multiple.
        """
        if self.frames and self.frames[-1][0] >= iteration:
            return
        self.frames.append((iteration, self._slice(tensor_no_cuda, timesteps_ticks)
                            .astype(np.int16)))
        # The last round's softmax, carried over from `step()`.
        self.conf.append(self.last_conf or [None] * int(self.dataset.num_voices))

    def _conf(self, probas):
        """The mean of the model's own softmax maximum, over the ticks this round drew.

        Uses the un-temperatured `probas` (not `probas_pitch`), computed on the context
        before the write, estimated over `batch_size_per_voice` random ticks.
        """
        out = [None] * int(self.dataset.num_voices)
        for v in self.voice_indices:
            out[int(v)] = float(probas[v].max(dim=1).values.mean().item())
        return out


_REC = None  # the recorder; set to None to mute it during the A/B check


# ---------------------------------------------------------------------------
# parallel_gibbs with recording
#
# A verbatim copy of `DeepBach.parallel_gibbs` in
# src/deepbach_pytorch/DeepBach/model_manager.py, with the three insertions marked
# [ADDED n/3]. main() checks it against the original with a fixed seed.
# ---------------------------------------------------------------------------
def recording_parallel_gibbs(self,
                             tensor_chorale,
                             tensor_metadata,
                             timesteps_ticks,
                             num_iterations=1000,
                             batch_size_per_voice=16,
                             temperature=1.,
                             time_index_range_ticks=None,
                             voice_index_range=None,
                             voice_indices=None,
                             ):
    if voice_indices is None:
        voice_indices = list(range(voice_index_range[0], voice_index_range[1]))
    tensor_chorale = tensor_chorale.unsqueeze(0)
    tensor_chorale_no_cuda = tensor_chorale.clone()
    tensor_metadata = tensor_metadata.unsqueeze(0)
    tensor_metadata_cuda = cuda_variable(tensor_metadata)

    min_temperature = temperature
    temperature_sa = max(min_temperature, 1.1)

    # [ADDED 1/3] The random init frame: `generation()` already filled the to-generate
    # voices -- the state before round 0. ab_inputs holds post-unsqueeze tensors; pass [0].
    if _REC is not None and not _REC.started:
        _REC.begin(model=self,
                   tensor_no_cuda=tensor_chorale_no_cuda,
                   timesteps_ticks=timesteps_ticks,
                   voice_indices=voice_indices,
                   ab_inputs=(tensor_chorale_no_cuda.clone(),
                              tensor_metadata.clone(),
                              timesteps_ticks,
                              time_index_range_ticks,
                              voice_index_range,
                              list(voice_indices)))

    for iteration in tqdm(range(num_iterations)):
        temperature_sa = max(min_temperature, temperature_sa * 0.9993)

        time_indexes_ticks = {}
        probas = {}

        with torch.no_grad():
            tensor_chorale_cuda = cuda_variable(tensor_chorale_no_cuda)

            for voice_index in voice_indices:
                batch_notes = []
                batch_metas = []
                time_indexes_ticks[voice_index] = []

                for batch_index in range(batch_size_per_voice):
                    time_index_ticks = np.random.randint(*time_index_range_ticks)
                    time_indexes_ticks[voice_index].append(time_index_ticks)

                    notes, _ = self.voice_models[voice_index].preprocess_notes(
                        tensor_chorale=tensor_chorale_cuda[
                            :, :,
                            time_index_ticks - timesteps_ticks:
                            time_index_ticks + timesteps_ticks],
                        time_index_ticks=timesteps_ticks)
                    metas = self.voice_models[voice_index].preprocess_metas(
                        tensor_metadata=tensor_metadata_cuda[
                            :, :,
                            time_index_ticks - timesteps_ticks:
                            time_index_ticks + timesteps_ticks, :],
                        time_index_ticks=timesteps_ticks)

                    batch_notes.append(notes)
                    batch_metas.append(metas)

                batch_notes = list(map(list, zip(*batch_notes)))
                batch_notes = [torch.cat(lcr) if lcr[0] is not None else None
                               for lcr in batch_notes]
                batch_metas = list(map(list, zip(*batch_metas)))
                batch_metas = [torch.cat(lcr) for lcr in batch_metas]

                output = self.voice_models[voice_index].forward(batch_notes, batch_metas)
                probas[voice_index] = nn.Softmax(dim=1)(output)

            for voice_index in voice_indices:
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = to_numpy(probas[voice_index][batch_index])
                    probas_pitch = np.log(probas_pitch + 1e-12) / temperature_sa
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7
                    probas_pitch[probas_pitch < 0] = 0
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))
                    tensor_chorale_no_cuda[
                        0, voice_index,
                        time_indexes_ticks[voice_index][batch_index]
                    ] = int(pitch)

        # [ADDED 2/3] `probas` is the softmax this round sampled from, still in scope.
        if _REC is not None:
            _REC.step(iteration, temperature_sa, tensor_chorale_no_cuda,
                      timesteps_ticks, probas)

    # [ADDED 3/3] The state the caller gets back; see `Recorder.final`.
    if _REC is not None:
        _REC.final(num_iterations - 1, temperature_sa, tensor_chorale_no_cuda,
                   timesteps_ticks)

    return tensor_chorale_no_cuda[0, :, timesteps_ticks:-timesteps_ticks]


# ---------------------------------------------------------------------------
# Installing the hook
# ---------------------------------------------------------------------------
def install_hook(arch):
    """Patch `recording_parallel_gibbs` onto every class reachable.

    Returns (the patched classes, the class this run actually builds, that class's
    original method). The same source is imported as two module objects holding
    different `DeepBach` classes, so all four module names get checked.
    """
    targets = {}

    def add(cls):
        # `object` is excluded: no `parallel_gibbs`, and adding one would be global.
        if isinstance(cls, type) and cls is not object and hasattr(cls, 'parallel_gibbs'):
            targets[id(cls)] = cls

    add(dbp.DeepBach)
    add(getattr(dbp, 'SharedTrunkDeepBach', None))
    for modname in ('deepbach_pytorch.DeepBach.model_manager',
                    'DeepBach.model_manager',
                    'deepbach_pytorch.DeepBach.shared_trunk_model',
                    'DeepBach.shared_trunk_model'):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        add(getattr(mod, 'DeepBach', None))
        add(getattr(mod, 'SharedTrunkDeepBach', None))

    # The class this run actually builds, and its original method (for the A/B).
    wired = (getattr(dbp, 'SharedTrunkDeepBach', None) if arch == 'shared_trunk'
             else dbp.DeepBach)
    if wired is None:
        print(f'[Fatal] arch={arch} 但包里没有 SharedTrunkDeepBach')
        sys.exit(1)
    original = wired.parallel_gibbs

    for cls in targets.values():
        cls.parallel_gibbs = recording_parallel_gibbs

    assert wired.parallel_gibbs is recording_parallel_gibbs, \
        f'the patch did not land on the class arch={arch} builds -- the recorder ' \
        f'would stay silent ({wired.__module__}.{wired.__name__})'
    assert original is not recording_parallel_gibbs, \
        'the original was already the recording method -- call this once per process'
    return targets, wired, original


# ---------------------------------------------------------------------------
# Preflight check of the weights directory
# ---------------------------------------------------------------------------
VOCAB_KEYS = ['note_embeddings.%d.weight' % j for j in range(4)]

# The non-LSTM repr tail: a GRU checkpoint name ends in `,256,gru)`, LSTM in `,256)`.
GRU_TAIL = ',gru)'


def _cell_of(fname):
    """Which cell a weights file is: 'gru' or 'lstm'. The two tails are disjoint."""
    return 'gru' if fname.endswith(GRU_TAIL) else 'lstm'


def preflight_models_dir(models_dir, arch, rnn_type):
    """Check the weights dir holds exactly what (arch, rnn_type) calls for, and that the
    files agree on the vocab. Returns the vocab (note_embeddings.0..3 sizes).

    The three shapes that are correct:

        baseline, lstm   4 x VoiceModel(...,<voice>,20,20,2,256,0.5,256)
        baseline, gru    4 x VoiceModel(...,<voice>,20,20,2,256,0.5,256,gru)
        shared_trunk     1 x SharedTrunkModel(...,20,20,2,256,0.5,256,<rc>)

    "exactly" because the loader takes the first listdir match by `endswith(repr(model))`;
    one extra candidate makes it a filesystem accident.
    """
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f'weights directory does not exist: {models_dir}')

    want = 'SharedTrunkModel(' if arch == 'shared_trunk' else 'VoiceModel('
    other = 'VoiceModel(' if arch == 'shared_trunk' else 'SharedTrunkModel('
    present = [f for f in sorted(os.listdir(models_dir))
               if os.path.isfile(os.path.join(models_dir, f))
               and f.startswith(('VoiceModel(', 'SharedTrunkModel('))]

    stray = [f for f in present if f.startswith(other)]
    if stray:
        raise RuntimeError(
            f'arch={arch} wants {want}* files, but {models_dir} also holds '
            f'{len(stray)} {other}* file(s): {stray[:3]}\n'
            f'        两个架构的 repr 都以 ")" 结尾，同目录混放会让选哪份权重变成 '
            f'os.listdir 的偶然 —— 先把两个架构的目录分开。')

    mine = [f for f in present if f.startswith(want)]
    if not mine:
        raise FileNotFoundError(
            f'{models_dir} has no {want}* file for arch={arch} '
            f'(found: {present or "nothing matching either shape"})')

    wrong_cell = [f for f in mine if _cell_of(f) != rnn_type]
    if wrong_cell:
        raise RuntimeError(
            f'arch={arch} rnn_type={rnn_type} wants the '
            f'{"`" + GRU_TAIL + "`" if rnn_type == "gru" else "`,256)`"} tail, but '
            f'{models_dir} also holds {len(wrong_cell)} file(s) of the other cell: '
            f'{wrong_cell[:3]}\n'
            f'        两套 checkpoint 的名字只差尾部 {GRU_TAIL}，指向另一个 cell 的目录时'
            f'加载器一个都匹配不上，会静默走零填充、拿没训过的 embedding 跑完 —— '
            f'先把两个 cell 的目录分开，或者传对应的 --rnn-type。')

    if arch == 'shared_trunk':
        if len(mine) != 1:
            raise RuntimeError(
                f'arch=shared_trunk wants exactly one SharedTrunkModel file in '
                f'{models_dir}, found {len(mine)}: {mine}\n'
                f'        加载器取 listdir 里的第一个，多一个就是无声的偶然。')
        roles = [('shared trunk', mine[0])]
    else:
        per_voice = {}
        for fname in mine:
            # The number in the tail ,<voice>,20,20,2,256,0.5,256) is the voice index
            try:
                v = int(fname.split('),')[1].split(',')[0])
            except (IndexError, ValueError):
                print(f'[warn] 跳过无法解析声部号的文件: {fname}')
                continue
            per_voice.setdefault(v, []).append(fname)

        missing = [v for v in range(4) if v not in per_voice]
        if missing:
            raise FileNotFoundError(
                f'{models_dir} has no weights for voice(s) {missing} — '
                f'harmonize(keep_melody=True) needs all four. '
                f'(found: {sorted(per_voice)})')
        for v, names in per_voice.items():
            if len(names) != 1:
                raise RuntimeError(
                    f'voice {v} has {len(names)} candidate weight files in '
                    f'{models_dir}; the loader takes the first in listdir order, '
                    f'a silent accident — clean this up first: {names}')
        roles = [(f'voice {v}', per_voice[v][0]) for v in range(4)]

    vocabs = []
    for label, fname in roles:
        path = os.path.join(models_dir, fname)
        sd = torch.load(path, map_location='cpu', weights_only=True)
        got = tuple(sd[k].shape[0] for k in VOCAB_KEYS)
        vocabs.append(got)
        print(f'  {label}: vocab {got}  ({os.path.getsize(path) / 1e6:.1f} MB)')

    if len(set(vocabs)) != 1:
        # Every checkpoint stores note embeddings for all four voices, so they must agree.
        raise RuntimeError(
            f'the weights in {models_dir} disagree on the vocab: {vocabs} — '
            f'this directory was accumulated from mixed runs; decide which set to use.')
    return list(vocabs[0])


# ---------------------------------------------------------------------------
# Vocabulary: index -> pitch
# ---------------------------------------------------------------------------
def build_vocab(dataset):
    """Turn each voice's index2note_dicts into a form the browser can use directly.

    Symbols fall into three classes (split because they draw differently):
      '__' (SLUR_SYMBOL)                a tie, holds the previous note
      'rest' / 'START' / 'END' / 'XX'   silence
      'OOR' (OUT_OF_RANGE)              pitch lost, no midi value
    """
    assert dataset.index2note_dicts is not None, \
        'index2note_dicts is None — the dataset never ran compute_index_dicts()'

    vocab, unknown = [], []
    for v, mapping in enumerate(dataset.index2note_dicts):
        per_voice = {}
        for idx, note_str in mapping.items():
            if note_str == SLUR_SYMBOL:
                per_voice[str(idx)] = {'sym': 'slur'}
            elif note_str in (REST_SYMBOL, START_SYMBOL, END_SYMBOL, PAD_SYMBOL):
                per_voice[str(idx)] = {'sym': 'rest'}
            elif note_str == OUT_OF_RANGE:
                per_voice[str(idx)] = {'sym': 'oor'}
            else:
                try:
                    per_voice[str(idx)] = {
                        'midi': int(standard_note(note_str).pitch.midi)}
                except Exception as e:                       # noqa: BLE001
                    per_voice[str(idx)] = {'sym': 'unknown'}
                    unknown.append((v, idx, note_str, repr(e)))
        vocab.append(per_voice)
    if unknown:
        # Reported, not silent: the roll would omit the block otherwise
        print(f'[warn] {len(unknown)} 个音名无法解析成音高，将按 unknown 处理:')
        for u in unknown[:10]:
            print('        ', u)
    return vocab


# ---------------------------------------------------------------------------
# payload
#
# The structure the static page reads; defined once here on this side (the other
# side is `init()` in harmonize.html):
#     { meta: {...}, vocab: [ {<tok>: {midi}|{sym}} x V ], frames: [ {i, v, c} ] }
# ---------------------------------------------------------------------------
def write_payload(payload):
    """Write frames.json and frames.js; return both byte sizes."""
    blob = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        f.write(blob)

    # The js copy is the `file://` fallback: `<script src>` escapes the CORS blocking fetch.
    with open(OUT_JS, 'w', encoding='utf-8') as f:
        f.write('window.__FRAMES__ = ' + blob + ';\n')

    return os.path.getsize(OUT_JSON), os.path.getsize(OUT_JS)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
def _str2bool(text):
    """Written the same way as infer's function of that name."""
    if isinstance(text, bool):
        return text
    low = text.strip().lower()
    if low in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if low in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {text!r}')


def _axis_conflict(arch, rc, rnn_type_arg):
    """A meaningless (arch, axis) combination -> the [Fatal] text, else None.

    `rnn_type_arg` is the requested value, not the resolved one (shared_trunk resolves
    to 'lstm', so testing the resolved value would be dead code). Separate from `main()`
    so it is testable without loading a dataset; `_check_arms.py` covers the passing arms.
    """
    if arch == 'shared_trunk' and rnn_type_arg not in (None, 'lstm'):
        return (f'[Fatal] --rnn-type {rnn_type_arg} 配 --arch shared_trunk:'
                f'共享主干只有 LSTM，没有 cell 开关。\n'
                f'        GRU 是 baseline 的一个轴：--arch baseline --rnn-type gru')
    if arch == 'baseline' and rc:
        return ('[Fatal] --rc true 配 --arch baseline: baseline 没有这个开关，'
                '包会**静默忽略**它。\n'
                '        rc 只对 shared_trunk 有意义：--arch shared_trunk --rc true')
    return None


def _arm_text(arch, rc, rnn_type):
    """The short arm label the page shows; given by the payload, not inferred."""
    if arch == 'shared_trunk':
        return 'shared trunk · rc' if rc else 'shared trunk'
    return 'baseline · ' + rnn_type.upper()


def _live_vocab(model):
    """The per-voice note-embedding sizes of an already-built model, in voice order.

    baseline keeps them on `voice_models[0].note_embeddings`; shared_trunk's
    `voice_models[i]` is a `RoleView` that does not expose them, so they live on
    `.model`. Both are tried, without branching on `arch`.
    """
    first = getattr(model, 'voice_models', [])[:1]
    candidates = [next(iter(first), None), getattr(model, 'model', None)]
    for cand in candidates:
        emb = getattr(cand, 'note_embeddings', None)
        if emb is not None:
            return tuple(m.num_embeddings for m in emb)
    raise AttributeError(
        f'no note_embeddings reachable on {type(model).__name__} — neither '
        f'voice_models[0] nor .model; check 0 cannot read the live vocab')


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Record one harmonization run as frames.json + frames.js for "
                    "_viz/harmonize.html. Omitting every switch uses infer's CONFIG, "
                    "so it records exactly what `python infer.py` would generate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples -- the four arms this tree has:\n"
               "  python _viz\\_viz_harmonize.py                       "
               "infer's axes (baseline, GRU)\n"
               "  python _viz\\_viz_harmonize.py --rnn-type lstm        "
               "the same baseline on LSTM cells\n"
               "  python _viz\\_viz_harmonize.py --arch shared_trunk    "
               "one trunk, four role heads\n"
               "  python _viz\\_viz_harmonize.py --arch shared_trunk --rc true\n"
               "  python _viz\\_viz_harmonize.py --reuse               "
               "re-emit frames.js, sample nothing\n")
    ap.add_argument('--dataset', '--data-set', dest='dataset',
                    choices=list(infer.DATA_SETS), default=None,
                    help='override infer.DATA_SET: picks the tree, the dataset cache '
                         'and the weights root')
    ap.add_argument('--arch', choices=list(dbp.ARCHS), default=None,
                    help='override infer.ARCH. baseline = four independent '
                         'VoiceModels; shared_trunk = one trunk, four role heads')
    ap.add_argument('--rnn-type', '--cell', dest='rnn_type',
                    choices=list(dbp.RNN_TYPES), default=None,
                    help='override infer.RNN_TYPE: the baseline\'s recurrent cell, '
                         'selecting the weights leaf <tree>/models_<snap>/<cell>/. '
                         'Baseline only; shared_trunk is LSTM, and an explicit '
                         '--rnn-type gru with --arch shared_trunk is refused')
    ap.add_argument('--rc', nargs='?', const=True, default=None, type=_str2bool,
                    metavar='BOOL',
                    help='override infer.ROLE_CONDITIONED, shared_trunk only. '
                         'rc=true resolves to <tree>/models_rc/shared_trunk '
                         '(bare --rc means --rc true)')
    ap.add_argument('--snapshot', choices=list(dbp.SNAPSHOT_NAMES), default=None,
                    help='override the snapshot for this run only; without it each '
                         'family uses its own infer constant')
    ap.add_argument('--models-dir', default=None, metavar='DIR',
                    help='name the weights dir directly, skipping the axes. It moves '
                         'the weights only — the cache still comes from --dataset')
    ap.add_argument('--melody', default=None, metavar='FILE',
                    help='melody musicxml (default: the fixed belaa-koska line)')
    ap.add_argument('--iterations', type=int, default=None, metavar='N',
                    help=f'Gibbs rounds (default {NUM_ITERATIONS})')
    ap.add_argument('--temperature', type=float, default=None, metavar='T',
                    help=f'sampling temperature (default {TEMPERATURE})')
    ap.add_argument('--batch-per-voice', type=int, default=None, metavar='B',
                    help=f'parallel proposals per round (default {BATCH_PER_VOICE})')
    ap.add_argument('--melody-voice', type=int, default=None, choices=(0, 1, 2, 3),
                    help=f'which voice holds the melody (default {MELODY_VOICE})')
    ap.add_argument('--snapshot-every', type=int, default=None, metavar='K',
                    help=f'rounds per recorded frame (default {SNAPSHOT_EVERY})')
    ap.add_argument('--reuse', action='store_true',
                    help='sample nothing: re-emit frames.js from the existing '
                         'frames.json')
    return ap.parse_args(argv)


def main():
    global _REC

    args = _parse_args()

    # ------------------------------------------------------------------- --reuse
    if args.reuse:
        print('=' * 72)
        print('只重刷 frames.js（--reuse），不跑采样')
        print('=' * 72)
        if not os.path.exists(OUT_JSON):
            print(f'[Fatal] 没有 {OUT_JSON} —— 先不带 --reuse 跑一次')
            return 1
        with open(OUT_JSON, encoding='utf-8') as f:
            payload = json.load(f)
        n_json, n_js = write_payload(payload)
        m = payload['meta']
        # `.get`: a frames.json older than the arm fields is still valid
        print(f'来源      : {m["source"]}  {m["iterations"]} 轮  '
              f'{len(payload["frames"])} 帧  '
              f'[{m.get("arm_text", "arm unrecorded")}]')
        print(f'[ok]   {OUT_JSON}  ({n_json / 1024:.0f} KB)')
        print(f'[ok]   {OUT_JS}  ({n_js / 1024:.0f} KB)')
        return 0

    # -------------------------------------------------------- effective values
    arch = args.arch or infer.ARCH
    rc = infer.ROLE_CONDITIONED if args.rc is None else args.rc

    # `rnn_type` is baseline-only; shared_trunk is fixed 'lstm', not `infer.RNN_TYPE`.
    rnn_type = args.rnn_type or (infer.RNN_TYPE if arch == 'baseline' else 'lstm')

    # Pass `args.rnn_type`, not the resolved `rnn_type`: only explicit requests are refused.
    conflict = _axis_conflict(arch, rc, args.rnn_type)
    if conflict:
        print(conflict)
        return 1

    snapshot = args.snapshot or infer._snapshot_for(arch, rc, rnn_type)
    melody = args.melody or MELODY
    rounds = NUM_ITERATIONS if args.iterations is None else args.iterations
    temperature = TEMPERATURE if args.temperature is None else args.temperature
    batch = BATCH_PER_VOICE if args.batch_per_voice is None else args.batch_per_voice
    melody_voice = (MELODY_VOICE if args.melody_voice is None
                    else args.melody_voice)
    every = SNAPSHOT_EVERY if args.snapshot_every is None else args.snapshot_every

    # Switch the tree first: `_set_data_set` rebinds those three paths inside it.
    data_set = args.dataset or infer.DATA_SET
    infer._set_data_set(data_set)
    # `rnn_type` goes in by keyword; as a positional it lands in the wrong slot.
    models_dir = (os.path.abspath(args.models_dir) if args.models_dir
                  else infer._weights_dir(arch, rc, snapshot, rnn_type=rnn_type))

    print('=' * 72)
    print('DeepBach 配和声过程可视化')
    print('=' * 72)
    print(f'包        : {dbp.__file__}')
    print(f'repo      : {REPO}')
    print(f'树        : {data_set}    arch={arch}    rc={rc}    '
          f'rnn_type={rnn_type}')
    print(f'快照      : {snapshot}')
    print(f'权重目录  : {models_dir}')
    print(f'产物目录  : {OUT_DIR}')
    print(f'旋律      : {melody}')
    print(f'轮数      : {rounds}   temperature={temperature}   '
          f'batch/声部={batch}   melody_voice={melody_voice}')
    print(f'帧        : 每 {every} 轮一帧')
    print()

    if not os.path.exists(melody):
        print(f'[Fatal] 找不到旋律文件: {melody}')
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- dataset cache ----
    # Must be installed before the in-tree paths are read. A missing object file is fatal:
    # `DatasetManager` reads it as a rebuild, drops the tensor files and re-reads the corpus.
    infer._install_tree_cache(infer.TREE_CACHE_DIR)
    objs_path, tensors_path = infer._bach_cache_paths()
    print('--- 数据集缓存 ---')
    print(f'目录      : {infer.TREE_CACHE_DIR}')
    if not os.path.exists(objs_path):
        print(f'[Fatal] 对象缓存不存在: {objs_path}')
        print(f'        缺它会触发重建（删掉 {tensors_path} 再重读整个语料）。'
              f'换一棵已有的树（--dataset），或先建缓存。')
        return 1
    print(f'[ok]   {os.path.basename(objs_path)}  '
          f'({os.path.getsize(objs_path):,} B)')
    print()

    print('--- 权重目录飞行前检查 ---')
    print(f'目录      : {models_dir}')
    try:
        weight_vocabs = preflight_models_dir(models_dir, arch, rnn_type)
    except (FileNotFoundError, RuntimeError) as e:
        # Both are configuration errors; the exception text already gives the reason.
        print(f'[Fatal] {e}')
        return 1
    print()

    targets, wired, original = install_hook(arch)
    print('钩子已挂 : ' + ', '.join(sorted(
        f'{c.__module__}.{c.__name__}' for c in targets.values())))
    print(f'本次用的类: {wired.__module__}.{wired.__name__}')
    print()

    _REC = Recorder(every)

    t0 = time.time()
    dbp.harmonize(
        input_file=melody,
        output_path=OUT_XML,
        num_iterations=rounds,
        temperature=temperature,
        batch_size_per_voice=batch,
        melody_voice=melody_voice,
        keep_melody=True,
        models_dir=models_dir,
        arch=arch,
        role_conditioned=rc,
        rnn_type=rnn_type,
    )
    print(f'\n采样 + 写谱耗时 {(time.time() - t0) / 60:.1f} 分钟')

    rec = _REC
    ok = True

    print('\n' + '=' * 72)
    print('校验')
    print('=' * 72)

    # 0) The weight vocab must equal the dataset vocab, or loading took the zero-pad branch.
    live = _live_vocab(rec.model)
    if live == tuple(weight_vocabs):
        print(f'[ok]   权重 vocab {live} 与 dataset 完全一致，没有零填充')
    else:
        print(f'[FAIL] 权重 vocab {tuple(weight_vocabs)} != dataset vocab {live}')
        print('       加载时走了零填充分支，模型已劣化 —— 换一组匹配的权重再跑。')
        ok = False

    # 1) The hook really was called (otherwise the patch landed on the wrong class object).
    if rec.calls == 0:
        print('[FAIL] 记录器一次都没被调用 —— 补丁挂错了类对象')
        return 1
    print(f'[ok]   钩子触发 {rec.calls} 次（= 迭代轮数 {rounds}）')
    if rec.calls != rounds:
        print(f'[warn] 触发次数 {rec.calls} != 轮数 {rounds}')
        ok = False

    # 2) The melody voice is element-wise constant.
    mel = rec.frames[0][1][melody_voice]
    bad = [it for it, fr in rec.frames if not np.array_equal(fr[melody_voice], mel)]
    if bad:
        print(f'[FAIL] voice {melody_voice} 在帧 {bad[:5]} 里变了 —— 旋律不该被重采样')
        ok = False
    else:
        print(f'[ok]   voice {melody_voice}（旋律）在全部 {len(rec.frames)} 帧里逐元素恒定')

    # 3) The anneal's final value == the analytic one (the anneal starts at 1.1, so a
    # `rounds` too small never reaches the requested temperature).
    analytic_last = max(temperature, 1.1 * 0.9993 ** rounds)
    need = math.ceil(math.log(temperature / 1.1) / math.log(0.9993))

    if rec.last_temp is None:
        print('[FAIL] 一轮都没记录到 —— 钩子没进循环')
        ok = False
    elif abs(rec.last_temp - analytic_last) > 1e-5:
        print(f'[FAIL] 退火末值 {rec.last_temp:.6f} 与解析值 {analytic_last:.6f} 不符')
        ok = False
    else:
        print(f'[ok]   退火末值 {rec.last_temp:.4f}（解析值 {analytic_last:.4f} 一致）')
        if abs(analytic_last - temperature) > 1e-12:
            print(f'[note] 没有钳到请求的 temperature={temperature}：'
                  f'{rounds} 轮只降到 {rec.last_temp:.4f}，'
                  f'要钳住需要 {need} 轮。')
        else:
            print(f'[ok]   已钳到请求的 temperature={temperature}')

    # 4) Every resampled voice has a confidence in (0,1] on every frame (the melody is
    # not in voice_indices, its slot is None).
    gen = [int(v) for v in rec.voice_indices]
    if len(rec.conf) != len(rec.frames):
        print(f'[FAIL] 置信度 {len(rec.conf)} 份 != 帧数 {len(rec.frames)}')
        ok = False
    else:
        vals = np.array([c[v] for c in rec.conf for v in gen if c[v] is not None],
                        dtype=float)
        want = (len(rec.frames) - 1) * len(gen)   # the random init frame has no forward
        out_of_range = vals[(vals <= 0.0) | (vals > 1.0) | ~np.isfinite(vals)]
        if len(vals) != want:
            print(f'[FAIL] 置信度 {len(vals)} 个，应为 {want} 个'
                  f'（{len(gen)} 声部 × {len(rec.frames) - 1} 帧）')
            ok = False
        elif len(out_of_range):
            print(f'[FAIL] 有 {len(out_of_range)} 个置信度不在 (0,1] 里：'
                  f'{out_of_range[:5]} —— 概率不可能超过 1，取错了分布')
            ok = False
        else:
            last = rec.conf[-1]
            print(f'[ok]   置信度 {len(vals)} 个全在 (0,1]，末帧 ' + '  '.join(
                f'{VOICE_NAMES[v]} {last[v]:.3f}' for v in gen))
            print(f'         （旋律声部 {VOICE_NAMES[melody_voice]} 不参与重采样，'
                  f'没有置信度）')

    # 5) Fidelity A/B: the copy and `original` (the method of the class actually used,
    # see install_hook) are element-wise equal.
    print('\n--- 忠实性 A/B（固定种子，原版 vs 抄写本） ---')
    tc, tm, ts, tir, vir, vi = rec.ab_inputs
    ab_kw = dict(num_iterations=20, batch_size_per_voice=4, temperature=temperature,
                 time_index_range_ticks=tir, voice_index_range=vir,
                 voice_indices=vi)

    # ab_inputs holds post-unsqueeze tensors; feeding the original needs [0], else it is
    # unsqueezed again to 4-D.
    a_in, m_in = tc[0].clone(), tm[0].clone()

    _REC = None                       # muted, so the A/B frames stay out of the real data
    np.random.seed(20240915)
    a = original(rec.model, a_in.clone(), m_in.clone(), ts, **ab_kw)
    np.random.seed(20240915)
    b = recording_parallel_gibbs(rec.model, a_in.clone(), m_in.clone(), ts, **ab_kw)
    _REC = rec                        # restored; the payload write still needs it

    if torch.equal(a, b):
        print('[ok]   20 轮固定种子下两份逐元素相等')
    else:
        d = int((a != b).sum())
        print(f'[FAIL] 两份不等，{d} / {a.numel()} 个元素不同 —— 抄错了，停下来查')
        ok = False

    # ---- write the outputs ----
    print('\n' + '=' * 72)
    print('写产物')
    print('=' * 72)

    dataset = rec.dataset
    vocab = build_vocab(dataset)

    payload = {
        'meta': {
            'source': os.path.basename(melody),
            'iterations': rounds,
            'temperature': temperature,               # requested
            'actual_temp': round(analytic_last, 6),   # what the anneal actually reached
            'generated_voices': [int(v) for v in rec.voice_indices],
            'num_voices': int(dataset.num_voices),
            'ticks_per_bar': int(dataset.subdivision * 4),   # the input is in 4/4
            'total_ticks': int(rec.frames[0][1].shape[1]),
            'batch_per_voice': batch,  # confidence is averaged over this many ticks
            'voice_names': VOICE_NAMES,
            # The arm's identity; the static page labels from these, see `_arm_text`.
            'arch': arch,
            'rc': bool(rc),
            'rnn_type': rnn_type,
            'arm_text': _arm_text(arch, rc, rnn_type),
        },
        'vocab': vocab,
        # `c` is the confidence, indexed by voice number; unsampled voices are null.
        'frames': [{'i': int(i),
                    'v': [row.tolist() for row in fr],
                    'c': c}
                   for (i, fr), c in zip(rec.frames, rec.conf)],
    }

    n_json, n_js = write_payload(payload)
    print(f'[ok]   {OUT_JSON}  ({n_json / 1024:.0f} KB, {len(rec.frames)} 帧)')
    print(f'[ok]   {OUT_JS}  ({n_js / 1024:.0f} KB)')

    # The score harmonize() wrote itself must read back too
    import music21
    back = music21.converter.parse(OUT_XML)
    counts = [len([x for x in p.flatten().notes if x.isNote]) for p in back.parts]
    print(f'[ok]   {os.path.basename(OUT_XML)} 回读成功，各声部音符数 = {counts}')
    if not all(counts):
        print('[FAIL] 有声音部一个音都没有')
        ok = False

    print('\n' + '=' * 72)
    print('结论:', ' 全部通过' if ok else ' 有失败项，见上')
    print('=' * 72)
    print(f'打开页面  : {os.path.join(OUT_DIR, "harmonize.html")}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
