"""
@author: Gaetan Hadjeres
Modified: train() now forwards lr / lr_patience / lr_factor to VoiceModel.train_model()
"""

from DatasetManager.metadata import FermataMetadata
import numpy as np
import os
import torch
from DeepBach.helpers import cuda_variable, to_numpy

from torch import optim, nn
from tqdm import tqdm

from DeepBach.training_state import _ascii
from DeepBach.voice_model import RNN_TYPES, VoiceModel

# Let cuDNN pick the fastest kernel for fixed-size inputs.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# Single source of truth for the baseline hyperparameters (symmetric with `shared_trunk_model.HYPERS`).
BASELINE_HYPERS = dict(
    note_embedding_dim=20,
    meta_embedding_dim=20,
    num_layers=2,
    lstm_hidden_size=256,
    dropout_lstm=0.5,
    linear_hidden_size=256,
    # 'lstm' | 'gru': the recurrent cell, not a third architecture; the GRU arm just
    # swaps this baseline's two cells. Listed here for `build()`'s `**overrides` check.
    rnn_type='lstm',
)


class DeepBach:
    def __init__(self,
                 dataset,
                 note_embedding_dim,
                 meta_embedding_dim,
                 num_layers,
                 lstm_hidden_size,
                 dropout_lstm,
                 linear_hidden_size,
                 models_dir=None,
                 rnn_type='lstm',
                 ):
        self.dataset = dataset
        self.num_voices = self.dataset.num_voices
        self.num_metas = len(self.dataset.metadatas) + 1
        self.activate_cuda = torch.cuda.is_available()

        self.voice_models = [VoiceModel(
            dataset=self.dataset,
            main_voice_index=main_voice_index,
            note_embedding_dim=note_embedding_dim,
            meta_embedding_dim=meta_embedding_dim,
            num_layers=num_layers,
            lstm_hidden_size=lstm_hidden_size,
            dropout_lstm=dropout_lstm,
            hidden_size_linear=linear_hidden_size,
            models_dir=models_dir,
            rnn_type=rnn_type,
        )
            for main_voice_index in range(self.num_voices)
        ]

    @classmethod
    def build(cls, dataset, models_dir=None, lstm_hidden_size=None, **overrides):
        """
        Build four independent VoiceModels from `BASELINE_HYPERS`.

        `lstm_hidden_size=None` -> 256. `**overrides` overrides the rest of that
        table; the CLI's `--note_embedding_dim` / `--num_layers` come in here.
        """
        unknown = sorted(set(overrides) - set(BASELINE_HYPERS))
        if unknown:
            raise TypeError(
                f'unknown hyperparameters: {unknown}; '
                f'BASELINE_HYPERS has {sorted(BASELINE_HYPERS)}')
        # Validate at the entry point, so a bad value fails before four VoiceModels are built and the dataset loaded.
        rnn_type = overrides.get('rnn_type', BASELINE_HYPERS['rnn_type'])
        if rnn_type not in RNN_TYPES:
            raise ValueError(
                f'rnn_type must be one of {RNN_TYPES}, got {rnn_type!r}')
        if lstm_hidden_size is None:
            lstm_hidden_size = BASELINE_HYPERS['lstm_hidden_size']
        elif not isinstance(lstm_hidden_size, int) or lstm_hidden_size <= 0:
            raise ValueError(
                f'lstm_hidden_size must be a positive int or None, '
                f'got {lstm_hidden_size!r}')

        return cls(
            dataset=dataset,
            models_dir=models_dir,
            **{**BASELINE_HYPERS, **overrides,
               'lstm_hidden_size': lstm_hidden_size},
        )

    @property
    def models_dir(self):
        # All four VoiceModels share one `models_dir`, so the first will do. Mirrors
        # `SharedTrunkDeepBach`; `db._checkpoint_paths` looks for it on the wrapper.
        return self.voice_models[0].models_dir

    def cuda(self, main_voice_index=None):
        if self.activate_cuda:
            if main_voice_index is None:
                for voice_index in range(self.num_voices):
                    self.cuda(voice_index)
            else:
                self.voice_models[main_voice_index].cuda()

    def load(self, main_voice_index=None, models_dir=None):
        """
        Load each voice's checkpoint one by one.

        :param models_dir: the directory to read; None -> each VoiceModel's own
            `models_dir`. Snapshot sets live in **sibling** directories
            (`models_best/` and `models_epoch01/` ..., see `DatasetManager.helpers`),
            hence reading another set without rebuilding the model.
        """
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.load(main_voice_index=voice_index, models_dir=models_dir)
        else:
            self.voice_models[main_voice_index].load(models_dir=models_dir)

    def save(self, main_voice_index=None, models_dir=None):
        """Save each voice's checkpoint one by one. `models_dir` as in `load`."""
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.save(main_voice_index=voice_index, models_dir=models_dir)
        else:
            self.voice_models[main_voice_index].save(models_dir=models_dir)

    def train(self,
              main_voice_index=None,
              lr: float = 1e-3,
              lr_patience: int = 3,
              lr_factor: float = 0.5,
              **kwargs):
        """
        Train the four voice models in turn.

        `**kwargs` go straight through to `VoiceModel.train_model`, which is how the
        budget (`batch_size` / `steps_per_epoch` / `max_steps`) is set; not declared
        here so the four voices get the *same* budget. `lr` is not forwarded — it
        builds the optimizer above.

        The four voices share one `models_dir` and each writes under its own
        `__repr__()`; `_assert_distinct_voice_reprs` checks they really are distinct.
        """
        if main_voice_index is None:
            self._assert_distinct_voice_reprs()
            for voice_index in range(self.num_voices):
                self.train(
                    main_voice_index=voice_index,
                    lr=lr,
                    lr_patience=lr_patience,
                    lr_factor=lr_factor,
                    **kwargs,
                )
        else:
            voice_model = self.voice_models[main_voice_index]
            if self.activate_cuda:
                voice_model.cuda()
            optimizer = optim.Adam(voice_model.parameters(), lr=lr)
            voice_model.train_model(
                optimizer=optimizer,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                **kwargs,
            )

    def _assert_distinct_voice_reprs(self):
        """
        Every voice's `__repr__()` must be unique and usable as a filename.

        The four voices write into **one** `models_dir` and are told apart by this
        string alone (it is the filename of the checkpoint and of the history file).
        Equal reprs raise nothing — the later silently overwrites the earlier — hence
        one check before training starts. A repr with a path separator or `..` would
        write outside `models_dir`.
        """
        reprs = [model.__repr__() for model in self.voice_models]
        for index, repr_ in enumerate(reprs):
            if os.path.sep in repr_ or (os.path.altsep and os.path.altsep in repr_) \
                    or '..' in repr_:
                raise ValueError(
                    f'voice {index} has a repr that cannot be used as a filename '
                    f'(path separator or ".."):\n  {_ascii(repr_)}')
        seen = {}
        for index, repr_ in enumerate(reprs):
            if repr_ in seen:
                raise ValueError(
                    f'voices {seen[repr_]} and {index} have identical reprs, so '
                    f'they would write to the same checkpoint and overwrite each '
                    f'other:\n  {_ascii(repr_)}')
            seen[repr_] = index
        print(f'[voices] {len(reprs)} distinct reprs, ok')

    def eval_phase(self):
        for voice_model in self.voice_models:
            voice_model.eval()

    def train_phase(self):
        for voice_model in self.voice_models:
            voice_model.train()

    def generation(self,
                   temperature=1.0,
                   batch_size_per_voice=8,
                   num_iterations=None,
                   sequence_length_ticks=160,
                   tensor_chorale=None,
                   tensor_metadata=None,
                   time_index_range_ticks=None,
                   voice_index_range=None,
                   fermatas=None,
                   random_init=True,
                   voice_indices=None,
                   ):
        self.eval_phase()

        if tensor_chorale is None:
            tensor_chorale = self.dataset.random_score_tensor(sequence_length_ticks)
        else:
            sequence_length_ticks = tensor_chorale.size(1)

        if tensor_metadata is None:
            test_chorale = next(self.dataset.corpus_it_gen().__iter__())
            tensor_metadata = self.dataset.get_metadata_tensor(test_chorale)
            if tensor_metadata.size(1) < sequence_length_ticks:
                tensor_metadata = tensor_metadata.repeat(
                    1, sequence_length_ticks // tensor_metadata.size(1) + 1, 1)
            tensor_metadata = tensor_metadata[:, :sequence_length_ticks, :]
        else:
            assert tensor_metadata.size(1) == sequence_length_ticks

        if fermatas is not None:
            tensor_metadata = self.dataset.set_fermatas(tensor_metadata, fermatas)

        timesteps_ticks = self.dataset.sequences_size * self.dataset.subdivision // 2

        if time_index_range_ticks is None:
            time_index_range_ticks = [timesteps_ticks,
                                      sequence_length_ticks + timesteps_ticks]
        else:
            a_ticks, b_ticks = time_index_range_ticks
            assert 0 <= a_ticks < b_ticks <= sequence_length_ticks
            time_index_range_ticks = [a_ticks + timesteps_ticks,
                                      b_ticks + timesteps_ticks]

        if voice_index_range is None:
            voice_index_range = [0, self.dataset.num_voices]

        # `voice_index_range` is a contiguous [start, end). For "everything but one
        # voice" use `voice_indices`; a range cannot express that set.
        if voice_indices is None:
            voice_indices = list(range(voice_index_range[0], voice_index_range[1]))

        tensor_chorale_padded = self.dataset.extract_score_tensor_with_padding(
            tensor_score=tensor_chorale,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks)
        tensor_metadata_padded = self.dataset.extract_metadata_with_padding(
            tensor_metadata=tensor_metadata,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks)

        if random_init:
            a, b = time_index_range_ticks
            random_chunk = self.dataset.random_score_tensor(b - a)
            for voice_index in voice_indices:
                tensor_chorale_padded[voice_index, a:b] = random_chunk[voice_index, :]

        tensor_chorale_final = self.parallel_gibbs(
            tensor_chorale=tensor_chorale_padded,
            tensor_metadata=tensor_metadata_padded,
            num_iterations=num_iterations,
            timesteps_ticks=timesteps_ticks,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice,
            time_index_range_ticks=time_index_range_ticks,
            voice_index_range=voice_index_range,
            voice_indices=voice_indices,
        )

        metadata_index = 0
        for i, metadata in enumerate(self.dataset.metadatas):
            if isinstance(metadata, FermataMetadata):
                metadata_index = i
                break

        score = self.dataset.tensor_to_score(
            tensor_score=tensor_chorale_final,
            fermata_tensor=tensor_metadata[:, :, metadata_index])

        return score, tensor_chorale_final, tensor_metadata

    def parallel_gibbs(self,
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

        return tensor_chorale_no_cuda[0, :, timesteps_ticks:-timesteps_ticks]