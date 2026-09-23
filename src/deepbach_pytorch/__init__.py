import os
import sys
import shutil

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from .DatasetManager.helpers import (
    DATA_DIR,
    DATASET_CACHE_DIR,
    MODEL_ARCHS,
    SNAPSHOT_NAMES,
    DEFAULT_SNAPSHOT,
    snapshot_dir,
    sibling_snapshot_dir,
    REST_SYMBOL,
    strip_fermatas,
    atomic_torch_save,
)

# Default load location: `models_<DEFAULT_SNAPSHOT>/<arch>/`.  Training writes every
# snapshot directory, telling them apart by the models_dir passed in.
MODELS_DIR              = snapshot_dir(DEFAULT_SNAPSHOT, 'baseline')
SHARED_TRUNK_MODELS_DIR = snapshot_dir(DEFAULT_SNAPSHOT, 'shared_trunk')

import torch
import music21
from music21 import expressions
from tqdm import tqdm

from .DeepBach.model_manager import DeepBach, BASELINE_HYPERS
from .DeepBach.voice_model import RNN_TYPES
from .DeepBach.shared_trunk_model import (
    SharedTrunkDeepBach,
    HYPERS as SHARED_TRUNK_HYPERS,
    DEFAULT_LSTM_HIDDEN_SIZE,
)
from .DatasetManager.dataset_manager import DatasetManager
from .DatasetManager.chorale_dataset import ChoraleDataset
from .DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from .DeepBach.training_state import DEFAULT_TOTAL_PASSES


# The two architectures.  An alias of `helpers.MODEL_ARCHS`, not a second definition.
ARCHS = MODEL_ARCHS

# `lstm_hidden_size=None` looks up this table.  Both architectures are 256/256.
_DEFAULT_LSTM_HIDDEN = {
    'baseline':     BASELINE_HYPERS['lstm_hidden_size'],
    'shared_trunk': DEFAULT_LSTM_HIDDEN_SIZE,
}


def _resolve_lstm_hidden_size(arch, lstm_hidden_size):
    """None -> this architecture's default; otherwise validate and pass through."""
    if lstm_hidden_size is None:
        return _DEFAULT_LSTM_HIDDEN[arch]
    if not isinstance(lstm_hidden_size, int) or lstm_hidden_size <= 0:
        raise ValueError(
            f'lstm_hidden_size must be a positive int or None, '
            f'got {lstm_hidden_size!r}')
    return lstm_hidden_size


# ---------------------------------------------------------------------------
# Custom corpus
# ---------------------------------------------------------------------------

def _make_custom_corpus_iterator(midi_dir):
    """
    Return a callable yielding music21 Scores parsed from a directory.

    Types: .mid .midi .xml .mxl .musicxml, in sorted order; unparseable files
    are skipped with a warning.
    """
    midi_dir = os.path.abspath(midi_dir)
    _SUPPORTED = ('.mid', '.midi', '.xml', '.mxl', '.musicxml')

    def corpus_it_gen():
        files = sorted(f for f in os.listdir(midi_dir)
                       if f.lower().endswith(_SUPPORTED))
        if not files:
            raise FileNotFoundError(
                f"No supported music files found in: {midi_dir}\n"
                f"Supported extensions: {_SUPPORTED}")
        print(f"  Custom corpus: {len(files)} files found in {midi_dir}")
        for fname in files:
            fpath = os.path.join(midi_dir, fname)
            try:
                score = music21.converter.parse(fpath)
                yield score
            except Exception as e:
                print(f"  Warning: skipping {fname}: {e}")

    return corpus_it_gen


def _load_or_create_chorale_dataset(corpus_it_gen, name, cache_dir, **kwargs):
    """
    Build a ChoraleDataset and cache it, mirroring DatasetManager's pattern.
    """
    dataset = ChoraleDataset(
        corpus_it_gen=corpus_it_gen,
        name=name,
        cache_dir=cache_dir,
        **kwargs)

    filepath = dataset.filepath
    if os.path.exists(filepath):
        print(f'Loading {dataset.__repr__()} from {filepath}')
        try:
            loaded = torch.load(filepath, weights_only=False)
            loaded.corpus_it_gen = corpus_it_gen
            print('(the corresponding TensorDataset is not loaded)')
            return loaded
        except Exception as e:
            print(f'Warning: could not load cached dataset ({e}); rebuilding.')

    print(f'Saving dataset object to {filepath}')
    atomic_torch_save(dataset, filepath)
    return dataset


def _get_default_dataset(custom_midi_dir=None, voice_ids=None):
    """
    Build or load the ChoraleDataset.

    Args:
        custom_midi_dir: optional folder of MIDI/XML files used instead of the
            built-in Bach chorales corpus.
        voice_ids: which score parts become "voices".  None -> [0,1,2,3].  A
            single index (e.g. [0]) builds a 1-voice dataset (`DeepBach` then
            builds one `VoiceModel` with `num_voices == 1`).

    Returns:
        ChoraleDataset (freshly created or loaded from cache)
    """
    metadatas = [
        FermataMetadata(),
        TickMetadata(subdivision=4),
        KeyMetadata()
    ]
    if voice_ids is None:
        voice_ids = [0, 1, 2, 3]
    else:
        voice_ids = [int(v) for v in voice_ids]
        if not voice_ids:
            raise ValueError('voice_ids must not be empty')

    chorale_dataset_kwargs = {
        'voice_ids':      voice_ids,
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4,
    }

    if custom_midi_dir is not None:
        custom_midi_dir = os.path.abspath(custom_midi_dir)
        if not os.path.isdir(custom_midi_dir):
            raise ValueError(
                f"custom_midi_dir is not a valid directory: {custom_midi_dir}")

        corpus_it_gen = _make_custom_corpus_iterator(custom_midi_dir)
        dataset_name  = 'custom_' + os.path.basename(custom_midi_dir)
        cache_dir     = DATASET_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        dataset = _load_or_create_chorale_dataset(
            corpus_it_gen=corpus_it_gen,
            name=dataset_name,
            cache_dir=cache_dir,
            **chorale_dataset_kwargs)
    else:
        dataset_manager = DatasetManager()
        dataset = dataset_manager.get_dataset(
            name='bach_chorales',
            **chorale_dataset_kwargs)

    return dataset


def _build_shared_trunk(dataset, models_dir=None, role_conditioned=False,
                        lstm_hidden_size=None):
    """
    Construct scheme A -- one shared LSTM trunk with four per-role heads.

    The single place the shared-trunk hyperparameters are applied; the expected
    checkpoint filename comes from the same `SHARED_TRUNK_HYPERS` table.

    Does NOT load weights or move the model to the GPU -- `_load_pretrained_weights`
    owns the loading.
    """
    if models_dir is None:
        models_dir = SHARED_TRUNK_MODELS_DIR
    else:
        models_dir = os.path.abspath(models_dir)
    return SharedTrunkDeepBach.build(
        dataset=dataset,
        models_dir=models_dir,
        role_conditioned=role_conditioned,
        lstm_hidden_size=lstm_hidden_size,
    )


def _get_default_model(dataset=None, models_dir=None, arch='baseline',
                       role_conditioned=False, lstm_hidden_size=None,
                       rnn_type='lstm'):
    """
    Instantiate a model of the requested architecture, weights NOT loaded.

    Args:
        dataset:           ChoraleDataset.  None -> build the default one.
        models_dir:        Weight directory.  None -> the architecture's own
                           default (`MODELS_DIR` / `SHARED_TRUNK_MODELS_DIR`).
        arch:              'baseline' (four independent VoiceModels) or
                           'shared_trunk' (one shared LSTM trunk with four
                           per-role heads, wrapped in four RoleViews).
        role_conditioned:  Shared trunk only.  False drops the voice_id channel
                           from the trunk, so one forward pass yields all four
                           roles' distributions at a tick.
        lstm_hidden_size:  None -> the architecture's default (256 for both).
        rnn_type:          'lstm' (default) or 'gru'.  Baseline only; a
                           non-default value with arch='shared_trunk' is refused.
                           Part of the checkpoint filename (`VoiceModel.__repr__`
                           gains a trailing `,gru`).

    Returns:
        DeepBach or SharedTrunkDeepBach, on the CPU, without weights.
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")

    # Refused rather than ignored: `SharedTrunkModel` has no cell switch, so this would
    # otherwise build an LSTM wearing a GRU label.
    if arch == 'shared_trunk' and rnn_type != 'lstm':
        raise ValueError(
            f"rnn_type={rnn_type!r} with arch='shared_trunk': the shared trunk is "
            f"LSTM only.  A GRU arm is arch='baseline' with rnn_type='gru'.")

    if dataset is None:
        dataset = _get_default_dataset()

    if arch == 'shared_trunk':
        return _build_shared_trunk(dataset, models_dir=models_dir,
                                   role_conditioned=role_conditioned,
                                   lstm_hidden_size=lstm_hidden_size)

    if models_dir is None:
        models_dir = MODELS_DIR
    return DeepBach.build(
        dataset=dataset,
        models_dir=models_dir,
        lstm_hidden_size=_resolve_lstm_hidden_size('baseline',
                                                   lstm_hidden_size),
        rnn_type=rnn_type,
    )


def _voice_weight_suffix(main_voice_index, voice_model=None, rnn_type=None,
                         **hyperparams):
    """
    The filename tail that identifies one voice's weight file.

    A weight file *is* the model's ``__repr__()``, so the suffix is
    ``',<i>,<note_emb>,<meta_emb>,<layers>,<hidden>,<dropout>,<linear>)'``.
    ``VoiceModel`` only -- ``SharedTrunkModel``'s repr has no per-voice index and
    its own ``save``/``load`` are self-consistent; route it through
    ``SharedTrunkModel.load``.

    The cell type is part of the suffix: ``,gru`` is appended for the non-default
    cell only, matching ``VoiceModel.__repr__`` (so every LSTM file keeps its
    name).  With `voice_model` given the value comes from the model itself, so
    loader and model agree by construction; `rnn_type` is for callers that have
    only hyperparameters (see `check_pretrained_weights`).
    """
    if voice_model is not None:
        attrs = ('note_embedding_dim', 'meta_embedding_dim', 'num_layers',
                 'lstm_hidden_size', 'dropout_lstm', 'hidden_size_linear')
        values = [getattr(voice_model, a) for a in attrs]
        rnn_type = getattr(voice_model, 'rnn_type', 'lstm')
    else:
        # Taken from BASELINE_HYPERS, not copied into literals, so a wrong filename
        # cannot outlive a table edit.  Table: `linear_hidden_size`, repr:
        # `hidden_size_linear`; both are accepted.
        if 'hidden_size_linear' in hyperparams:
            hyperparams = {**hyperparams,
                           'linear_hidden_size':
                               hyperparams['hidden_size_linear']}
        defaults = dict(BASELINE_HYPERS)
        # Pop first so it does not land in `values` below; an explicit `rnn_type=`
        # overrides the table's default.
        defaults.pop('rnn_type', None)
        defaults.update(hyperparams)
        defaults.setdefault('hidden_size_linear',
                            defaults['linear_hidden_size'])
        values = [defaults[a] for a in
                  ('note_embedding_dim', 'meta_embedding_dim', 'num_layers',
                   'lstm_hidden_size', 'dropout_lstm', 'hidden_size_linear')]
        if rnn_type is None:
            rnn_type = BASELINE_HYPERS['rnn_type']
    tail = '' if rnn_type == 'lstm' else f',{rnn_type}'
    return (f",{main_voice_index}," + ','.join(str(v) for v in values)
            + tail + ')')


def _ensure_models_dir(models_dir=None, arch='baseline'):
    if models_dir is None:
        models_dir = (SHARED_TRUNK_MODELS_DIR if arch == 'shared_trunk'
                      else MODELS_DIR)
    else:
        models_dir = os.path.abspath(models_dir)
    if not os.path.exists(models_dir):
        if arch == 'shared_trunk':
            raise FileNotFoundError(
                f"Shared-trunk model directory not found: {models_dir}\n"
                f"Train one with train_from_scratch(arch='shared_trunk'), or "
                f"point models_dir at an existing checkpoint directory.")
        raise FileNotFoundError(
            f"Model directory not found: {models_dir}\n"
            f"Please ensure pretrained weights are downloaded to this directory.")
    return models_dir


def _checkpoint_paths(model, models_dir=None):
    """
    The exact files `model` reads from / writes to, as absolute paths.

    Both architectures name a checkpoint after ``repr()`` of the object that saves
    it -- ``VoiceModel`` per baseline voice, ``SharedTrunkModel`` once for the
    trunk.  Names come from the live objects, which keeps loader, trainer's
    post-run check and `check_pretrained_weights` in agreement.

    The architecture is decided structurally (does it expose ``shared_model``?),
    not by ``isinstance``: the model files are reachable under two module names,
    so the same source becomes two class objects and ``isinstance`` can say False.

    Returns:
        list[str] -- four paths for the baseline, one for the shared trunk.
    """
    if models_dir is None:
        models_dir = getattr(model, 'models_dir', None)
    if not models_dir:
        # Both wrappers have `models_dir`; reaching here means the object handed in has
        # no checkpoint directory at all.
        raise ValueError(
            f'{type(model).__name__} has no models_dir; pass models_dir= '
            f'explicitly, or use a wrapper that has one')
    models_dir = os.path.abspath(models_dir)
    shared_model = getattr(model, 'shared_model', None)
    if shared_model is not None:
        return [os.path.join(models_dir, repr(shared_model))]
    return [os.path.join(models_dir, repr(vm)) for vm in model.voice_models]


def _load_pretrained_weights(model, models_dir=None, arch='baseline'):
    """
    Load pretrained weights for `model`, dispatching on the architecture.

    Baseline: one file per voice, zero-padding checkpoint tensors where the vocab
    has grown (note_embeddings, mlp_predictions).

    Shared trunk: one file serves all four roles, so this delegates to
    `SharedTrunkModel.load`.
    """
    models_dir = _ensure_models_dir(models_dir, arch=arch)
    print(f">>> Loading pretrained weights from {models_dir}...")

    if arch == 'shared_trunk':
        expected = _checkpoint_paths(model, models_dir)[0]
        if not os.path.exists(expected):
            present = sorted(
                f for f in os.listdir(models_dir)
                if os.path.isfile(os.path.join(models_dir, f)))
            raise FileNotFoundError(
                f"Shared-trunk checkpoint not found.\n"
                f"  expected: {os.path.basename(expected)}\n"
                f"  in:       {models_dir}\n"
                f"  present:  {present}\n"
                f"The filename is the model's repr, so a different "
                f"lstm_hidden_size or role_conditioned changes it. Use "
                f"check_pretrained_weights(arch='shared_trunk') to compare.")
        try:
            model.load(models_dir=models_dir)
        except RuntimeError as e:
            # `load_state_dict` is strict=True, so a shape/vocab mismatch reports a bare
            # torch error; add the file path here.
            raise RuntimeError(
                f"Failed to load shared-trunk checkpoint:\n"
                f"  {expected}\n"
                f"  {e}\n"
                f"See selfcheck.check_state_dict(model, path) for a "
                f"missing/extra/shape-mismatch diff against the architecture."
            ) from e
        print(f">>> Shared-trunk weights loaded (one file, four roles).")
        return model

    device_map = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(4):
        found = False
        vm = model.voice_models[i]
        # Derived from the model being loaded, so non-default lstm_hidden_size /
        # embedding / cell still match.
        target_suffix = _voice_weight_suffix(i, vm)

        for fname in os.listdir(models_dir):
            if not fname.endswith(target_suffix):
                continue

            model_path = os.path.join(models_dir, fname)
            print(f"\n  Loading voice {i}...")

            try:
                loaded_state  = torch.load(model_path,
                                           map_location=device_map,
                                           weights_only=True)
                current_state = model.voice_models[i].state_dict()
                had_mismatch  = False

                for key in list(loaded_state.keys()):
                    if key not in current_state:
                        continue
                    lt = loaded_state[key]
                    ct = current_state[key]

                    if lt.shape == ct.shape:
                        continue

                    had_mismatch = True
                    print(f"    WARNING: shape mismatch '{key}'")
                    print(f"      Checkpoint : {lt.shape}")
                    print(f"      Model      : {ct.shape}")

                    if 'note_embeddings' in key and lt.dim() == 2:
                        ckpt_v, edim = lt.shape
                        curr_v = ct.shape[0]
                        if curr_v > ckpt_v:
                            print(f"      Strategy : zero-pad {ckpt_v} -> {curr_v} rows")
                            padded = torch.zeros(curr_v, edim, dtype=lt.dtype)
                            padded[:ckpt_v] = lt
                            loaded_state[key] = padded
                        else:
                            print(f"      Strategy : skip (vocab shrank)")
                            del loaded_state[key]

                    elif 'mlp_predictions' in key and lt.dim() == 2:
                        ckpt_o, in_f = lt.shape
                        curr_o = ct.shape[0]
                        if curr_o > ckpt_o:
                            print(f"      Strategy : zero-pad output weight "
                                  f"{ckpt_o} -> {curr_o}")
                            padded = torch.zeros(curr_o, in_f, dtype=lt.dtype)
                            padded[:ckpt_o] = lt
                            loaded_state[key] = padded
                        else:
                            print(f"      Strategy : skip output weight (shrank)")
                            del loaded_state[key]

                    elif 'mlp_predictions' in key and lt.dim() == 1:
                        ckpt_o = lt.shape[0]
                        curr_o = ct.shape[0]
                        if curr_o > ckpt_o:
                            print(f"      Strategy : zero-pad output bias "
                                  f"{ckpt_o} -> {curr_o}")
                            padded = torch.zeros(curr_o, dtype=lt.dtype)
                            padded[:ckpt_o] = lt
                            loaded_state[key] = padded
                        else:
                            print(f"      Strategy : skip output bias (shrank)")
                            del loaded_state[key]

                    elif key.startswith(('lstm_left.', 'lstm_right.')):
                        # Never a vocab change: these two recurrent layers' shapes depend
                        # only on the cell type, and `nn.LSTM` and `nn.GRU` share their
                        # parameter names exactly, differing only in the gate dimension
                        # (4*hidden vs 3*hidden).  Skipping + strict=False would silently
                        # leave random recurrent weights, so this raises instead.
                        hidden = vm.lstm_hidden_size
                        raise RuntimeError(
                            f"Checkpoint architectures do not match.\n"
                            f"  file:       {model_path}\n"
                            f"  key:        {key}\n"
                            f"  checkpoint: {tuple(lt.shape)}  "
                            f"({lt.shape[0] / hidden:g} gates x hidden)\n"
                            f"  model:      {tuple(ct.shape)}  "
                            f"({ct.shape[0] / hidden:g} gates x hidden)\n"
                            f"The recurrent cells differ -- 4 gates is LSTM, 3 is "
                            f"GRU -- and the two share their parameter key names, so "
                            f"this cannot be reconciled by resizing.  Point "
                            f"models_dir at the directory written by this cell type; "
                            f"VoiceModel's trailing `,gru` names the GRU files.")

                    else:
                        print(f"      Strategy : skip (unhandled mismatch)")
                        del loaded_state[key]

                model.voice_models[i].load_state_dict(loaded_state, strict=False)
                status = "with mismatch handling" if had_mismatch else "successfully"
                print(f"  OK: Voice {i} weights loaded {status}")
                found = True

            except RuntimeError as e:
                print(f"  ERROR: Voice {i} weight loading failed: {e}")
                import traceback; traceback.print_exc()
                raise

            break

        if not found:
            raise FileNotFoundError(
                f"Weights for voice {i} not found.\n"
                f"Expected file ending with: {target_suffix}\n"
                f"Search directory: {models_dir}")

    print("\nOK: All weights loaded successfully\n")


def _to_cuda_if_available(model):
    if torch.cuda.is_available():
        model.cuda()
        print(f"OK: Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not detected, using CPU mode (slower)")
    return model


def _count_fermatas(score):
    """Number of notes carrying a Fermata expression anywhere in the score."""
    total = 0
    for part in score.parts:
        for element in part.flatten().notes:
            if any(isinstance(e, expressions.Fermata)
                   for e in element.expressions):
                total += 1
    return total


def _apply_fermatas(dataset, metadata_tensor, user_score,
                    sequence_length_ticks, derive_fermatas):
    """
    Fill in the fermata metadata channel, which is DeepBach's cadential cue.

    If the file has no fermatas and `derive_fermatas` is set, invent them with the
    training convention (`FermataMetadata.generate`: last beat of every two bars)
    and also close on the final note; otherwise the channel stays as it is.

    Returns:
        (metadata_tensor, derived) -- `derived` is True when the channel was
        invented here rather than read from the file.
    """
    n_in_file = _count_fermatas(user_score)
    if n_in_file:
        print(f"  Fermatas: using {n_in_file} found in the input file")
        return metadata_tensor, False
    if not derive_fermatas:
        print("  Fermatas: none in the input and derive_fermatas=False "
              "-> channel left at zero (no cadential cue)")
        return metadata_tensor, False

    derived = torch.as_tensor(
        FermataMetadata().generate(sequence_length_ticks)).long()
    derived[max(sequence_length_ticks - 4, 0):] = 1     # the final note closes too
    metadata_tensor = dataset.set_fermatas(metadata_tensor, derived)
    print(f"  Fermatas: none in the input -> derived {int(derived.sum())} ticks "
          f"(every 2 bars + final note)")
    return metadata_tensor, True


def _strip_fermatas(score):
    """Alias for DatasetManager.helpers.strip_fermatas."""
    return strip_fermatas(score)


def _report_melody_range(dataset, melody_part, melody_voice):
    """
    Report how many melody notes fall outside the fixed voice's trained range.

    Out-of-range notes become a single OUT_OF_RANGE token and lose their pitch, so
    the generated voices have little to follow.
    """
    pitches = [n.pitch.midi for n in melody_part.flatten().notes if n.isNote]
    if not pitches:
        return

    fits = []
    for v in range(dataset.num_voices):
        vlo, vhi = dataset.voice_ranges[v]
        oor = sum(1 for p in pitches if p < vlo or p > vhi)
        fits.append((v, vlo, vhi, oor))

    cur = fits[melody_voice]
    print(f"  Melody range : {min(pitches)}..{max(pitches)} ({len(pitches)} notes)")
    print(f"  Voice {melody_voice} trained range: {cur[1]}..{cur[2]}"
          f"  ->  {cur[3]} note(s) out of range "
          f"({100.0 * cur[3] / len(pitches):.1f}%)")
    if not cur[3]:
        return

    print("  Out-of-range notes become a single 'OOR' token - their pitch is lost,")
    print("  so the generated voices have little to follow.")
    better = [f for f in fits if f[3] < cur[3]]
    if better:
        b = min(better, key=lambda f: f[3])
        print(f"  Closer fit: melody_voice={b[0]} "
              f"(trained {b[1]}..{b[2]}, {b[3]} out of range) - set melody_voice, "
              f"and for a full score put the melody in that part.")
    else:
        print("  No voice fits better - transpose the melody into "
              f"{cur[1]}..{cur[2]} instead.")


def _prepare_inference_model(models_dir=None, arch='baseline',
                            role_conditioned=False, lstm_hidden_size=None,
                            custom_midi_dir=None, rnn_type='lstm'):
    """
    Build -> load weights -> move to GPU -> eval mode, for an inference entry.

    `arch` must reach all four consistently: a correct model with the wrong loader
    silently loads nothing.  `rnn_type` reaches only `_get_default_model` -- the
    loader derives the expected filename from the model it is handed.
    """
    dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model = _get_default_model(dataset, models_dir=models_dir, arch=arch,
                              role_conditioned=role_conditioned,
                              lstm_hidden_size=lstm_hidden_size,
                              rnn_type=rnn_type)
    _load_pretrained_weights(model, models_dir, arch=arch)
    _to_cuda_if_available(model)
    model.eval_phase()
    return dataset, model


def harmonize(input_file, output_path="output.xml", num_iterations=500,
              temperature=1.0, batch_size_per_voice=8, voice_index_range=None,
              random_init=None, melody_voice=0, keep_melody=True,
              models_dir=None, custom_midi_dir=None, derive_fermatas=True,
              fermata_marks="none", arch='baseline', role_conditioned=False,
              lstm_hidden_size=None, rnn_type='lstm'):
    """
    Harmonise a melody file using DeepBach.

    Reads a single-voice (or multi-voice) MIDI / MusicXML file, keeps the melody
    in *melody_voice* fixed, and generates the remaining voices via Gibbs sampling.
    A melody-only file is accepted: the melody is encoded with the trained note
    range and vocabulary of *melody_voice* and the other voices are filled with
    rests.

    Args:
        input_file:           Path to a .mid / .xml / .mxl melody file.
        output_path:          Where to write the harmonised MusicXML.
        num_iterations:       Gibbs sampling iterations.  Typical range: 200–1000.
        temperature:          Sampling temperature.  1.0 = standard DeepBach;
                              <1 more conservative, >1 more varied.
        batch_size_per_voice: Parallel Gibbs proposals per step (8 is default).
        voice_index_range:    [start, end] of voice indices to regenerate.  None
                              lets keep_melody / melody_voice decide.
        random_init:          Randomly initialise harmony voices before sampling.
                              None → True.
        melody_voice:         Voice index the melody occupies (0 = soprano,
                              1 = alto, 2 = tenor, 3 = bass).  The melody is
                              encoded with that voice's trained note range, so set
                              this to the voice whose range contains it.  The
                              melody is never resampled.
        keep_melody:          True: regenerate only voices other than
                              melody_voice; False: regenerate all four.
        models_dir:           Directory holding the weight file(s).  None -> the
                              architecture's own default, the `DEFAULT_SNAPSHOT`
                              set (`data/models_<snapshot>/<arch>`).  Do not mix
                              architectures in one directory: lookup is by
                              filename suffix.
        arch:                 'baseline' (four independent VoiceModels) or
                              'shared_trunk'.  Only the model changes -- the
                              sampler and the corpus are identical.
        role_conditioned:     Shared trunk only; must match the checkpoint or the
                              filename will not resolve.
        lstm_hidden_size:     None -> 256, both architectures.
        rnn_type:             'lstm' (default) or 'gru'.  Baseline only, and part
                              of the *checkpoint filename* (the `gru` arm's files
                              end in `,gru)`), so it must match the weights in
                              `models_dir`.  A mismatch raises rather than loading
                              a model with randomly initialised recurrent weights.
        custom_midi_dir:      If training used a custom corpus, pass the same
                              directory here so the dataset vocab matches.
        derive_fermatas:      Invent fermatas when the input has none, with the
                              training convention (last beat of every two bars,
                              plus the final note).  Default True, since an imported
                              melody normally carries none and the channel is the
                              model's only explicit cadential cue.
        fermata_marks:        Whether the fermata channel stays drawn on the output
                              score.  It is a *conditioning input*, not part of the
                              generated music, and the marks are purely notational.
                                "none"    (default) no fermata marks at all
                                "input"   keep only the input file's own fermatas
                                "channel" draw the channel as-is

    Returns:
        music21.stream.Score — the harmonised score (also written to output_path).
    """
    print("\n" + "=" * 70)
    print("DeepBach Harmony Generation")
    print("=" * 70)

    input_file  = os.path.abspath(input_file)
    output_path = os.path.abspath(output_path)

    print("\n>>> Initializing model...")
    dataset, model = _prepare_inference_model(
        models_dir=models_dir, arch=arch, role_conditioned=role_conditioned,
        lstm_hidden_size=lstm_hidden_size, custom_midi_dir=custom_midi_dir,
        rnn_type=rnn_type)

    print(f"\n>>> Reading input melody: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    try:
        user_score = music21.converter.parse(input_file)
    except Exception as e:
        raise ValueError(
            f"Failed to parse input file: {input_file}\n"
            f"Supported formats: .mid .midi .xml .mxl\n"
            f"Error: {e}")

    num_voices  = dataset.num_voices
    if melody_voice not in range(num_voices):
        raise ValueError(f"melody_voice must be 0-{num_voices - 1}, "
                         f"got: {melody_voice}")
    if not len(user_score.parts):
        raise ValueError(f"Input file contains no parts: {input_file}")
    melody_end  = user_score.flatten().highestTime

    print(">>> Converting melody to tensor...")
    # Which parts actually have notes: reaching the part count does not prove this is not
    # a melody-only score, and an empty staff would be taken for the melody.
    populated = [i for i, part in enumerate(user_score.parts)
                 if any(n.isNote for n in part.flatten().notes)]
    if not populated:
        raise ValueError(f"Input file contains no notes: {input_file}")

    try:
        if len(user_score.parts) >= num_voices and melody_voice in populated:
            # Full score: keep the existing part layout.
            melody_part  = user_score.parts[melody_voice]
            score_tensor = dataset.get_score_tensor(
                user_score, offsetStart=0., offsetEnd=melody_end)
        else:
            # Melody-only input: encode with the range/vocab of the slot it will occupy,
            # fill the other voices with rests, and make up the (num_voices, length)
            # tensor the sampler wants.
            if melody_voice in populated:
                source_index = melody_voice
            else:
                source_index = populated[0]
                print(f"  Voice {melody_voice} carries no notes in the input "
                      f"-> using part {source_index} as the melody; the other "
                      f"voice(s) will be generated.")
            if len(populated) > 1:
                print(f"  Input has {len(populated)} parts with notes "
                      f"({populated}) but the model works with {num_voices} "
                      f"voices; only part {source_index} is used as the melody.")
            melody_part = user_score.parts[source_index]
            melody_row  = dataset.part_to_tensor(
                melody_part, melody_voice, offsetStart=0., offsetEnd=melody_end)
            rest_index  = dataset.note2index_dicts[melody_voice][REST_SYMBOL]
            score_tensor = torch.full((num_voices, melody_row.shape[1]),
                                      rest_index, dtype=torch.long)
            score_tensor[melody_voice] = melody_row[0]
    except Exception as e:
        raise ValueError(f"Failed to convert melody to tensor: {e}")

    sequence_length_ticks = score_tensor.shape[1]
    if sequence_length_ticks == 0:
        raise ValueError(f"Input file has no music to harmonise "
                         f"(parsed length 0 ticks): {input_file}")
    print(f"  Melody length: {sequence_length_ticks} ticks "
          f"({sequence_length_ticks / 16:.1f} measures)")

    try:
        metadata_tensor = dataset.get_metadata_tensor(user_score)
    except Exception as e:
        raise ValueError(f"Failed to build metadata tensor: {e}")

    # model.generation() requires metadata and chorale to be the same length; the two come
    # from different music21 duration fields, so a score with an anacrusis or a short
    # final bar does not line up.
    if metadata_tensor.size(1) != sequence_length_ticks:
        md_len = metadata_tensor.size(1)
        if md_len < sequence_length_ticks:
            pad = torch.zeros(metadata_tensor.size(0),
                              sequence_length_ticks - md_len,
                              metadata_tensor.size(2),
                              dtype=metadata_tensor.dtype)
            metadata_tensor = torch.cat([metadata_tensor, pad], dim=1)
            print(f"  Metadata covers {md_len} of {sequence_length_ticks} ticks "
                  f"-> zero-padded (tick/key unknown for the tail)")
        else:
            metadata_tensor = metadata_tensor[:, :sequence_length_ticks, :]
            print(f"  Metadata covers {md_len} ticks, melody has "
                  f"{sequence_length_ticks} -> truncated")
        # voice_id is a per-voice constant channel; write it back after the re-slicing.
        voice_id_index = len(dataset.metadatas)
        for v in range(num_voices):
            metadata_tensor[v, :, voice_id_index] = v

    metadata_tensor, fermatas_derived = _apply_fermatas(
        dataset, metadata_tensor, user_score, sequence_length_ticks,
        derive_fermatas)
    _report_melody_range(dataset, melody_part, melody_voice)

    actual_random_init = random_init if random_init is not None else True

    if voice_index_range is not None:
        actual_voice_indices = list(range(voice_index_range[0],
                                          voice_index_range[1]))
    elif keep_melody:
        actual_voice_indices = [i for i in range(num_voices)
                                if i != melody_voice]
    else:
        actual_voice_indices = list(range(num_voices))
    if not actual_voice_indices:
        raise ValueError("No voice selected for generation "
                         f"(voice_index_range={voice_index_range})")
    # voice_index_range is a contiguous span and cannot express "every voice but the inner
    # ones"; the enumeration in voice_indices does the real work, and this span is only
    # accounting for the tensor padding inside generation().
    actual_voice_range = [min(actual_voice_indices),
                          max(actual_voice_indices) + 1]

    print(f"\n>>> Generation Parameters:")
    print(f"  Iterations:  {num_iterations}")
    print(f"  Temperature: {temperature}")
    print(f"  Keep melody: {keep_melody}  "
          f"(voices regenerated: {actual_voice_indices})")

    print(f"\n>>> Generating harmony (this may take a while)...")
    try:
        final_score, _, _ = model.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
            tensor_chorale=score_tensor,
            tensor_metadata=metadata_tensor,
            time_index_range_ticks=None,
            voice_index_range=actual_voice_range,
            voice_indices=actual_voice_indices,
            random_init=actual_random_init,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice)
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")

    # The channel has already done its work as a conditioning input; whether it stays on
    # the score is a purely notational choice.
    if fermata_marks not in ("input", "none", "channel"):
        raise ValueError('fermata_marks must be "input", "none" or "channel", '
                         f"got: {fermata_marks!r}")
    if fermata_marks == "none" or (fermata_marks == "input" and fermatas_derived):
        n_marks = _strip_fermatas(final_score)
        if n_marks:
            what = ("derived (the input had none)" if fermatas_derived
                    else "read from the input file")
            print(f"\n>>> Notation: removed {n_marks} fermata mark(s) — {what} "
                  f"(fermata_marks={fermata_marks!r})")

    print(f"\n>>> Saving results...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        final_score.write('musicxml', fp=output_path)
    except Exception as e:
        raise RuntimeError(f"Could not save output file: {e}")

    print("=" * 70)
    print(f"OK: Saved to: {output_path}")
    print("=" * 70 + "\n")
    return final_score


def generate_from_scratch(sequence_length_ticks=64, num_iterations=500,
                          temperature=1.0, batch_size_per_voice=8,
                          voice_index_range=None, random_init=True,
                          models_dir=None, custom_midi_dir=None,
                          arch='baseline', role_conditioned=False,
                          lstm_hidden_size=None, rnn_type='lstm'):
    """
    Generate a four-voice Bach-style chorale from scratch (no melody input).

    Args:
        sequence_length_ticks: Length in 16th-note ticks.  64 = 4 measures,
                               128 = 8 measures.
        num_iterations:        Gibbs sampling iterations.
        temperature:           Sampling diversity.  1.0 = default.
        batch_size_per_voice:  Parallel Gibbs proposals per step.
        voice_index_range:     [start, end] voices to generate.  None = all four.
        random_init:           Randomly initialise before sampling.
        models_dir:            Directory holding the weight file(s).  None ->
                               the architecture's own default.
        custom_midi_dir:       Custom corpus directory (must match training).
        arch:                  'baseline' or 'shared_trunk'; see `harmonize`.
        role_conditioned:      Shared trunk only; must match the checkpoint.
        lstm_hidden_size:      None -> 256, both architectures.
        rnn_type:              'lstm' (default) or 'gru'; baseline only, and part
                               of the checkpoint filename.  See `harmonize`.

    Returns:
        music21.stream.Score
    """
    print("\n" + "=" * 70)
    print("DeepBach Free Composition - Generating from scratch")
    print("=" * 70)

    print("\n>>> Initializing model...")
    dataset, model = _prepare_inference_model(
        models_dir=models_dir, arch=arch, role_conditioned=role_conditioned,
        lstm_hidden_size=lstm_hidden_size, custom_midi_dir=custom_midi_dir,
        rnn_type=rnn_type)

    actual_voice_range = voice_index_range if voice_index_range is not None else [0, 4]

    print(f"\n>>> Generation Parameters:")
    print(f"  Length:      {sequence_length_ticks} ticks "
          f"({sequence_length_ticks / 16:.1f} measures)")
    print(f"  Iterations:  {num_iterations}")
    print(f"  Temperature: {temperature}")
    print(f"  Voice range: {actual_voice_range}")

    print(f"\n>>> Generating composition...")
    try:
        score, _, _ = model.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
            tensor_chorale=None,
            tensor_metadata=None,
            time_index_range_ticks=None,
            voice_index_range=actual_voice_range,
            random_init=random_init,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice)
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")

    print("=" * 70)
    print("OK: Generation complete!")
    print("=" * 70 + "\n")
    return score


def _train_shared_trunk(steps_per_epoch=None, max_steps=None, batch_size=32,
                        lr=1e-3,
                        lr_patience=3, lr_factor=0.5,
                        models_dir=None, role_conditioned=False,
                        custom_midi_dir=None, dataset=None,
                        lstm_hidden_size=None):
    """
    Train scheme A -- one shared LSTM trunk with four per-role heads -- on the
    Bach chorales.

    Differs from the baseline path only inside the model: one `SharedTrunkModel`
    wrapped in four `RoleView`s (instead of four `VoiceModel`s), and one optimizer
    over it with loss = sum of the four cross-entropies.  The data is untouched --
    same `ChoraleDataset`, same tensor cache.

    The budget counts **optimizer steps**: `steps_per_epoch` None -> one full pass,
    `max_steps` None -> `DEFAULT_TOTAL_PASSES * steps_per_epoch`, rounded up to
    whole passes so every save point lands on a pass boundary (see
    `training_state.resolve_training_budget`).

    One checkpoint set per pass is written to `<models_dir>` and its siblings
    (`models_epochNN/`, `models_best/`); see `DatasetManager.helpers` for the
    layout.  Always trains from scratch; there is no resume.

    Args:
        steps_per_epoch:   Steps in one pass.  None -> `len(train_loader)`.
        max_steps:         Total optimizer steps.  None -> `DEFAULT_TOTAL_PASSES *
                           steps_per_epoch`, rounded up to whole passes.
        batch_size:        Training batch size.
        lr:                Initial Adam learning rate.
        lr_patience:       ReduceLROnPlateau patience, in passes.
        lr_factor:         LR reduction multiplier.
        models_dir:        The DEFAULT_SNAPSHOT checkpoint directory, and the base
                           the other snapshot directories derive from.  None ->
                           <package>/data/models_<DEFAULT_SNAPSHOT>/shared_trunk/.
        role_conditioned:  False drops the voice_id channel from the trunk, so one
                           forward pass yields all four roles' distributions at a
                           tick.  True runs the trunk once per role.
        custom_midi_dir:   Optional folder of MIDI/XML used instead of the
                           built-in Bach corpus.
        dataset:           Optional pre-built dataset.  None -> the default Bach
                           chorales.
        lstm_hidden_size:  Override the trunk's LSTM width.  None -> 256.

    Returns:
        The trained SharedTrunkDeepBach with its best-validation-loss checkpoint
        loaded (from `models_best/`, not `models_dir`).
    """
    # The default matches `_build_shared_trunk`; resolve it here too, since the concrete
    # directory is needed below for the warning and the makedirs.
    if models_dir is None:
        models_dir = SHARED_TRUNK_MODELS_DIR
    else:
        models_dir = os.path.abspath(models_dir)

    if os.path.exists(models_dir):
        print(f"\n>>> WARNING: models directory already exists; "
              f"weights may be overwritten: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    print("\n>>> Initializing dataset and shared-trunk model...")
    if dataset is None:
        dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)

    model = _build_shared_trunk(dataset, models_dir=models_dir,
                                role_conditioned=role_conditioned,
                                lstm_hidden_size=lstm_hidden_size)
    model.cuda()
    print(f'Model:      {model.model!r}')

    print(f"\n{'='*70}")
    print("Starting Training (scheme A: shared trunk, four role heads)")
    print(f"{'='*70}")
    print(f"Steps/pass: {steps_per_epoch if steps_per_epoch else 'one full pass'}")
    print(f"Max steps:  {max_steps if max_steps else f'{DEFAULT_TOTAL_PASSES} passes'}")
    print(f"Snapshot:   {DEFAULT_SNAPSHOT} -> {models_dir}")
    print(f"            (every pass is saved; `best` is a sibling of the above)")
    print(f"Batch size: {batch_size}")
    print(f"LR:         {lr}  (patience={lr_patience}, factor={lr_factor})")
    print("Loss:       CE(soprano) + CE(alto) + CE(tenor) + CE(bass)")
    print(f"{'='*70}\n")

    model.train(
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        max_steps=max_steps,
        lr=lr,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
    )

    # The best snapshot is a sibling of the models_dir passed in; check before loading, so
    # the error is not a FileNotFoundError from deep inside torch.load.
    best_dir = sibling_snapshot_dir(models_dir, 'best')
    ckpt = _checkpoint_paths(model, best_dir)[0]
    if not os.path.exists(ckpt):
        raise RuntimeError(
            f'Training finished but no checkpoint was written to {ckpt}; '
            f'every pass failed to improve on the validation loss.')
    model.load(models_dir=best_dir)
    print(f">>> Returned model carries the BEST-validation checkpoint from "
          f"{best_dir}")
    return model


def train_from_scratch(steps_per_epoch=None, max_steps=None, batch_size=32,
                       lr=1e-3,
                       lr_patience=3, lr_factor=0.5,
                       models_dir=None, custom_midi_dir=None,
                       arch='baseline', role_conditioned=False,
                       dataset=None, rnn_type='lstm'):
    """
    Train DeepBach from scratch on the Bach chorales (or a custom corpus).

    All four VoiceModels are trained sequentially.  Each voice saves only its
    best-validation-loss checkpoint.

    The budget counts **optimizer steps**: `steps_per_epoch` None -> one full pass,
    `max_steps` None -> `DEFAULT_TOTAL_PASSES * steps_per_epoch`, rounded up to
    whole passes so every save point lands on a pass boundary (see
    `training_state.resolve_training_budget`).

    One checkpoint set per pass is written to `<models_dir>` and its siblings
    (`models_epochNN/`, `models_best/`); see `DatasetManager.helpers` for the
    layout.  Always from scratch; there is no resume and no skipped voices.

    Args:
        steps_per_epoch: Steps in one pass.  None -> `len(train_loader)`.
        max_steps:       Total optimizer steps per voice.  None ->
                         `DEFAULT_TOTAL_PASSES * steps_per_epoch`, rounded up to
                         whole passes.
        batch_size:      Training batch size.  512 with an 85/10 split is the
                         reference setting.
        lr:              Initial Adam learning rate.
        lr_patience:     ReduceLROnPlateau patience (passes with no val
                         improvement before the LR is halved).
        lr_factor:       LR reduction multiplier (new_lr = lr * lr_factor).
        models_dir:      The DEFAULT_SNAPSHOT checkpoint directory, and the base
                         the other snapshot directories derive from.  None →
                         <package>/data/models_<DEFAULT_SNAPSHOT>/baseline/.
        custom_midi_dir: Optional folder of MIDI/XML files used instead of the
                         built-in Bach chorales corpus.
        arch:            'baseline' (four independent VoiceModels) or
                         'shared_trunk'.  Both train on the same 4-voice Bach
                         tensor cache.
        role_conditioned: Shared-trunk route only.  See `_train_shared_trunk`.
        dataset:         Optional pre-built dataset, used instead of the default
                         Bach corpus.  `build_dataset(voice_ids=[0])` yields a
                         1-voice dataset, i.e. one `VoiceModel` with
                         `num_voices == 1` -- the single-voice setting.
        rnn_type:        'lstm' (default) or 'gru'; baseline only.  Each voice's
                         `__repr__()` gains a trailing `,gru` for the non-default
                         cell, so a GRU run needs its own `models_dir`
                         (`<tree>/models_<snapshot>/gru/`).

    Returns:
        DeepBach model with best-epoch weights loaded (from `models_best/`).

    Raises:
        ValueError: ``arch`` is not one of the two known routes.
        ValueError: ``rnn_type`` is not one of ``RNN_TYPES``.
        ValueError: two of the baseline's voices would write to the same
                    checkpoint, or a voice's ``__repr__`` cannot be a filename.
                    See ``DeepBach._assert_distinct_voice_reprs``.
    """
    # Validate rather than fall back: an unknown arch landing on baseline would silently
    # train four ordinary VoiceModels.
    if arch not in ARCHS:
        raise ValueError(
            f"arch must be one of {ARCHS}, got {arch!r}")
    if arch == 'shared_trunk':
        # Refused rather than ignored: `_train_shared_trunk` has no `rnn_type`, so this
        # would train an LSTM wearing a gru label.
        if rnn_type != 'lstm':
            raise ValueError(
                f"rnn_type={rnn_type!r} with arch='shared_trunk': the shared "
                f"trunk is LSTM only.  A GRU arm is arch='baseline' with "
                f"rnn_type='gru'.")
        # All keyword arguments; positional ones would silently shift if the signature is
        # reordered.
        return _train_shared_trunk(
            steps_per_epoch=steps_per_epoch, max_steps=max_steps,
            batch_size=batch_size, lr=lr,
            lr_patience=lr_patience, lr_factor=lr_factor,
            models_dir=models_dir, role_conditioned=role_conditioned,
            custom_midi_dir=custom_midi_dir, dataset=dataset)

    print("\n" + "=" * 70)
    print("DeepBach Training from Scratch")
    print("=" * 70)

    if models_dir is None:
        models_dir = MODELS_DIR
    else:
        models_dir = os.path.abspath(models_dir)

    if os.path.exists(models_dir):
        print(f"\n>>> WARNING: models directory already exists; "
              f"weights may be overwritten: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    print("\n>>> Initializing dataset and model...")
    if dataset is None:
        dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model   = _get_default_model(dataset, models_dir=models_dir,
                                 arch='baseline', rnn_type=rnn_type)

    _to_cuda_if_available(model)

    # Deliberately no piece count: the number depends on the filter rules, and hardcoding
    # it would silently go stale.
    corpus_desc = custom_midi_dir or 'Bach Chorales (built-in corpus)'
    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}")
    print(f"Voices:     {dataset.num_voices}")
    print(f"Cell:       {rnn_type}")
    print(f"Steps/pass: {steps_per_epoch if steps_per_epoch else 'one full pass'}")
    print(f"Max steps:  {max_steps if max_steps else f'{DEFAULT_TOTAL_PASSES} passes'}")
    print(f"Snapshot:   {DEFAULT_SNAPSHOT} -> {models_dir}")
    print(f"            (every pass is saved; `best` is a sibling of the above)")
    print(f"Batch size: {batch_size}")
    print(f"LR:         {lr}  (patience={lr_patience}, factor={lr_factor})")
    print(f"Corpus:     {corpus_desc}")
    print(f"{'='*70}\n")

    try:
        model.train(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            max_steps=max_steps,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )
    except ValueError:
        # The voice-identity contract (`_assert_distinct_voice_reprs`): re-raise as-is,
        # wrapping it in RuntimeError would hide which of the two it was.
        raise
    except Exception as e:
        raise RuntimeError(f"Error during training: {e}")

    # The best snapshot is a sibling of the models_dir passed in.  Check that every
    # voice's file is present before loading, or a randomly initialised voice comes back.
    best_dir = sibling_snapshot_dir(models_dir, 'best')
    missing = [i for i in range(model.num_voices)
               if not os.path.exists(os.path.join(
                   best_dir, model.voice_models[i].__repr__()))]
    if missing:
        raise RuntimeError(
            f'Training finished but no checkpoint for voice(s) {missing} in '
            f'{best_dir}.  Every pass failed to improve on the validation '
            f'loss, or the file was removed mid-run.')
    model.load(models_dir=best_dir)

    print(f"\n{'='*70}")
    print(f"OK: Training complete!  Snapshots written under: "
          f"{os.path.dirname(best_dir)}")
    for _snap in SNAPSHOT_NAMES:
        _d = sibling_snapshot_dir(models_dir, _snap)
        _mark = '   <- what inference loads by default' \
            if _snap == DEFAULT_SNAPSHOT else ''
        print(f"    {_snap:<8} {_d}{_mark}")
    print(f"{'='*70}\n")
    return model


def build_dataset(custom_midi_dir=None, voice_ids=None):
    """
    Build (or load from cache) the ChoraleDataset without constructing a model.

    Args:
        custom_midi_dir: optional custom corpus directory.
        voice_ids: which score parts become voices.  None -> [0,1,2,3].  `[0]`
            gives a 1-voice dataset; pass the result to
            `train_from_scratch(dataset=...)`.

    Returns:
        ChoraleDataset
    """
    return _get_default_dataset(custom_midi_dir=custom_midi_dir,
                                voice_ids=voice_ids)


def build_model(dataset=None, models_dir=None, load_weights=True,
                custom_midi_dir=None, arch='baseline', role_conditioned=False,
                lstm_hidden_size=None, rnn_type='lstm'):
    """
    Construct a DeepBach model and optionally load weights.

    The lower-level alternative to harmonize() / generate_from_scratch() for
    callers that need the DeepBach object itself (e.g. to call model.generation()
    with custom tensor_chorale / tensor_metadata).

    Args:
        dataset:      ChoraleDataset (from build_dataset()).  None → build one.
        models_dir:   Weight directory.  None → the architecture's own default,
                      the `DEFAULT_SNAPSHOT` set
                      (`data/models_<snapshot>/<arch>`).
        load_weights: If True, load weights from models_dir.
        custom_midi_dir: Used when dataset is None.
        arch:         'baseline' or 'shared_trunk'; see `_get_default_model`.
        role_conditioned: Shared trunk only; must match the checkpoint.
        lstm_hidden_size: None → 256, both architectures.
        rnn_type:     'lstm' (default) or 'gru'; baseline only, and part of the
                      checkpoint filename.  Must match the weights in
                      `models_dir`.

    Returns:
        DeepBach model (on GPU if available, weights loaded if load_weights=True).
    """
    if dataset is None:
        dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model = _get_default_model(dataset, models_dir=models_dir, arch=arch,
                              role_conditioned=role_conditioned,
                              lstm_hidden_size=lstm_hidden_size,
                              rnn_type=rnn_type)
    if load_weights:
        _load_pretrained_weights(model, models_dir, arch=arch)
    _to_cuda_if_available(model)
    return model


def create_model(dataset=None, models_dir=None, arch='baseline',
                 role_conditioned=False, lstm_hidden_size=None,
                 rnn_type='lstm'):
    """
    Instantiate a DeepBach model without loading weights.

    Equivalent to build_model(load_weights=False).

    Args:
        dataset:    ChoraleDataset.  None → build default Bach chorales dataset.
        models_dir: Directory where weights will be saved during training.
                    None → the architecture's own default.
        arch:       'baseline' or 'shared_trunk'; see `_get_default_model`.
        role_conditioned: Shared trunk only.
        lstm_hidden_size: None → 256, both architectures.
        rnn_type:   'lstm' (default) or 'gru'; baseline only.  A GRU run's files
                    end in `,gru)` and never match an LSTM suffix, so it needs
                    its own `models_dir`.

    Returns:
        DeepBach model (no weights, not moved to GPU).
    """
    return _get_default_model(dataset, models_dir=models_dir, arch=arch,
                              role_conditioned=role_conditioned,
                              lstm_hidden_size=lstm_hidden_size,
                              rnn_type=rnn_type)


def load_model(dataset=None, models_dir=None, custom_midi_dir=None,
               arch='baseline', role_conditioned=False, lstm_hidden_size=None,
               rnn_type='lstm'):
    """
    Load a fully ready DeepBach model (weights loaded, on GPU if available).

    Shorthand for build_model(load_weights=True).

    Args:
        dataset:         ChoraleDataset.  None → build default.
        models_dir:      Weight directory.  None → the architecture's own default.
        custom_midi_dir: Used when dataset is None.
        arch:            'baseline' or 'shared_trunk'; see `_get_default_model`.
        role_conditioned: Shared trunk only; must match the checkpoint.
        lstm_hidden_size: None → 256, both architectures.
        rnn_type:        'lstm' (default) or 'gru'; baseline only, and part of
                         the checkpoint filename.  Must match the weights in
                         `models_dir`.

    Returns:
        DeepBach model ready for inference.
    """
    if dataset is None:
        dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model = _get_default_model(dataset, models_dir=models_dir, arch=arch,
                              role_conditioned=role_conditioned,
                              lstm_hidden_size=lstm_hidden_size)
    _load_pretrained_weights(model, models_dir, arch=arch)
    _to_cuda_if_available(model)
    return model


def _expected_weight_suffix(voice_index, arch, voice_model=None,
                            role_conditioned=False, lstm_hidden_size=None,
                            rnn_type=None):
    """
    The filename tail that identifies this architecture's checkpoint.

    Baseline: ``VoiceModel``'s tail, with the per-voice index.  Shared trunk:
    ``SharedTrunkModel``'s tail -- no index (one file serves all four roles), but
    it does carry ``role_conditioned``.

    Both derive from the tables the models are built from (`SHARED_TRUNK_HYPERS` /
    the live VoiceModel), so a checkpoint written by this package is findable by
    the check that looks for it.  `rnn_type` reaches only the baseline branch --
    the shared trunk is LSTM only.
    """
    if arch == 'shared_trunk':
        hs = _resolve_lstm_hidden_size('shared_trunk', lstm_hidden_size)
        return (f",{SHARED_TRUNK_HYPERS['note_embedding_dim']},"
                f"{SHARED_TRUNK_HYPERS['meta_embedding_dim']},"
                f"{SHARED_TRUNK_HYPERS['num_layers']},"
                f"{hs},"
                f"{SHARED_TRUNK_HYPERS['dropout_lstm']},"
                f"{SHARED_TRUNK_HYPERS['linear_hidden_size']},"
                f"{role_conditioned})")
    return _voice_weight_suffix(voice_index, voice_model, rnn_type=rnn_type)


def check_pretrained_weights(models_dir=None, voice_model=None, arch='baseline',
                             role_conditioned=False, lstm_hidden_size=None,
                             rnn_type='lstm'):
    """
    Check whether the weight file(s) this architecture needs are present.

    Args:
        models_dir: directory to check.  None → the architecture's own default,
            the `DEFAULT_SNAPSHOT` set (`data/models_<snapshot>/<arch>`).  For
            another snapshot set, pass `dataset_helpers.snapshot_dir(name, arch)`.
        voice_model: baseline only.  Any one VoiceModel; when given, its
            hyperparameters define the filenames looked for.  Omit for the
            package defaults (20,20,2,256,0.5,256).
        arch: 'baseline' (four VoiceModel files) or 'shared_trunk' (one file
            serving all four roles).
        role_conditioned: shared trunk only; part of the expected filename.
        lstm_hidden_size: None → the architecture's default (256 for both).
        rnn_type: 'lstm' (default) or 'gru'; baseline only, and part of the
            expected filename.  Ignored when `voice_model` is given, where the
            model's own cell is authoritative.

    Returns:
        dict with keys:
            complete         (bool)     — every required file was found
            arch             (str)      — which route was checked
            models_dir       (str)      — the directory that was checked
            found_weights    (int)      — how many required files were found
            required_weights (int)      — 4 for baseline, 1 for shared trunk
            missing_voices   (list)     — baseline only: indices not found
            expected         (list)     — per file: {'voice', 'suffix', 'found',
                                          'file'}; `voice` is None for the
                                          shared trunk
            present_files    (list)     — files actually in the directory
            present_dirs     (list)     — sub-directories (e.g. `_history`)
            missing_files    (list)     — the suffixes with no match
            error            (str)      — only when models_dir does not exist

    Note:
        A False here is a statement about filenames, not about loadability;
        `selfcheck.check_weights_dir` adds the md5 and a key/shape diff.
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    if models_dir is None:
        models_dir = (SHARED_TRUNK_MODELS_DIR if arch == 'shared_trunk'
                      else MODELS_DIR)

    if not os.path.exists(models_dir):
        return {
            'complete':         False,
            'arch':             arch,
            'models_dir':       models_dir,
            'found_weights':    0,
            'required_weights': 4 if arch == 'baseline' else 1,
            'missing_voices':   [0, 1, 2, 3] if arch == 'baseline' else [],
            'expected':         [],
            'present_files':    [],
            'present_dirs':     [],
            'missing_files':    [],
            'error':            f'Model directory does not exist: {models_dir}',
        }

    # Files only: a directory whose name happens to end in the weight suffix would satisfy
    # the check, and `_history/` sitting next to the checkpoints is the norm.
    entries = sorted(os.listdir(models_dir))
    present_files = [f for f in entries
                     if os.path.isfile(os.path.join(models_dir, f))]
    present_dirs = [f for f in entries if f not in present_files]

    voices = [None] if arch == 'shared_trunk' else [0, 1, 2, 3]
    expected = []
    for i in voices:
        suffix = _expected_weight_suffix(
            i, arch, voice_model=voice_model,
            role_conditioned=role_conditioned,
            lstm_hidden_size=lstm_hidden_size,
            rnn_type=rnn_type)
        match = next((f for f in present_files if f.endswith(suffix)), None)
        expected.append({'voice': i, 'suffix': suffix,
                         'found': match is not None, 'file': match})

    found_weights = sum(1 for e in expected if e['found'])
    missing_files = [e['suffix'] for e in expected if not e['found']]
    required = len(voices)

    result = {
        'complete':         found_weights == required,
        'arch':             arch,
        'models_dir':       models_dir,
        'found_weights':    found_weights,
        'required_weights': required,
        'expected':         expected,
        'present_files':    present_files,
        'present_dirs':     present_dirs,
        'missing_files':    missing_files,
    }
    if arch == 'baseline':
        result['missing_voices'] = [e['voice'] for e in expected
                                    if not e['found']]
    return result


def get_device_info():
    """
    Return information about the compute device that will be used.

    Returns:
        dict with keys device_type ('cuda' or 'cpu'), device_name (GPU name or
        'CPU'), cuda_available (bool).
    """
    if torch.cuda.is_available():
        return {
            'device_type':    'cuda',
            'device_name':    torch.cuda.get_device_name(0),
            'cuda_available': True,
        }
    return {
        'device_type':    'cpu',
        'device_name':    'CPU',
        'cuda_available': False,
    }



def __getattr__(name):
    """
    Lazy submodule access (PEP 562): `db.analysis`, `db.selfcheck`.

    Not a hard import at the bottom of this file: both modules import
    `deepbach_pytorch` themselves, so that would be a partially-initialised cycle.
    Resolving on first attribute access means the package is fully initialised.
    """
    if name in ('analysis', 'selfcheck'):
        import importlib
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'harmonize',
    'generate_from_scratch',
    'train_from_scratch',
    'build_dataset',       # build / load ChoraleDataset
    'build_model',         # build model + optionally load weights + move to GPU
    'create_model',        # build model only, no weights, no GPU move
    'load_model',          # build model + load weights + move to GPU
    'check_pretrained_weights',
    'get_device_info',
    'DeepBach',
    'SharedTrunkDeepBach',
    'DatasetManager',
    'ChoraleDataset',
    'FermataMetadata',
    'TickMetadata',
    'KeyMetadata',
    'PACKAGE_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'SHARED_TRUNK_MODELS_DIR',
    'DATASET_CACHE_DIR',
    'ARCHS',
    'RNN_TYPES',           # ('lstm', 'gru'): the baseline's recurrent cell
    'analysis',            # measurement helpers (lazy: see __getattr__)
    'selfcheck',           # invariant checks (lazy: see __getattr__)
]


if __name__ == '__main__':
    print(f"DeepBach PyTorch — package root: {PACKAGE_ROOT}")
    info = get_device_info()
    print(f"Compute device : {info['device_name']}")
    for arch, default_dir in (('baseline', MODELS_DIR),
                              ('shared_trunk', SHARED_TRUNK_MODELS_DIR)):
        status = check_pretrained_weights(arch=arch)
        what = ('all 4 voices found' if arch == 'baseline'
                else 'the trunk file (one serves all four roles)')
        if status['complete']:
            print(f"{arch:>13} weights : OK ({what})")
            print(f"  In               : {status['models_dir']}")
        else:
            print(f"{arch:>13} weights : MISSING "
                  f"{status['found_weights']}/{status['required_weights']}")
            print(f"  Expected in      : {status['models_dir']}")
            for entry in status['expected']:
                if not entry['found']:
                    print(f"  Missing name ends: {entry['suffix']}")
            if status['present_files']:
                print(f"  Present          : {status['present_files']}")
            if status['present_dirs']:
                print(f"  Present (dirs)   : {status['present_dirs']}")
            if status.get('error'):
                print(f"  Error            : {status['error']}")
