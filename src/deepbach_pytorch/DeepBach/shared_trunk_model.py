"""
Plan A — one shared trunk plus four role heads.

Three premises:
1. `get_metadata_tensor` tiles metadata with `repeat(num_voices, 1)`, so fermata /
   tick / key are elementwise identical across voices and only the last channel
   (voice_id) varies with the role. The trunk's left/right inputs are therefore
   role-independent.
2. In `preprocess_notes`, `mask_entry` + `other_voices_indexes` give each role a
   **different subset in a different order**, so `mlp_center` must be four heads and
   cannot be shared.
3. The sampling path touches the model only in `DeepBach.parallel_gibbs`, which
   requires `voice_models[i]` to expose `preprocess_notes` / `preprocess_metas` /
   `forward`; `RoleView` wraps the shared model as four VoiceModel-shaped views, so
   `model_manager.py` needs no change.

`role_conditioned` picks the variant:

    False   the trunk drops the voice_id channel; one forward gives all four roles'
            conditional distributions.
    True    the trunk keeps voice_id; four forwards over shared weights, with no
            joint score.
"""

import os
import random

import torch
from torch import nn, optim

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.helpers import (atomic_torch_save, snapshot_dir,
                                    sibling_snapshot_dir, DEFAULT_SNAPSHOT)
from DeepBach.data_utils import mask_entry, reverse_tensor
from DeepBach.helpers import cuda_variable, init_hidden
from DeepBach.model_manager import DeepBach
from DeepBach.training_state import (make_history_row,
                                     resolve_training_budget,
                                     save_history, save_pass_snapshot)

try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = 'cuda'
except ImportError:  # PyTorch < 2.4
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_DEVICE = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_THIS_DIR)
# Default output = the **default snapshot** directory `data/models_<snapshot>/shared_trunk`;
# the layout is defined in `DatasetManager.helpers`, not repeated as literals here.
#
# It needs a directory of its own: weight loading matches filenames by `endswith` and
# both architectures' `__repr__`s end in `)`, so a shared directory would make the
# model choice an accident of `os.listdir` order.
_DEFAULT_MODELS_DIR = snapshot_dir(DEFAULT_SNAPSHOT, 'shared_trunk')

# Single source of truth for Plan A hyperparameters. `SharedTrunkDeepBach.build`
# constructs from it, and `_checkpoint_paths` / `check_pretrained_weights` derive the
# checkpoint filenames from the same table.
# 256/256 matches the baseline `BASELINE_HYPERS` (the 256/256 of 2018 upstream).
HYPERS = dict(
    note_embedding_dim=20,
    meta_embedding_dim=20,
    num_layers=2,
    lstm_hidden_size=256,
    dropout_lstm=0.5,
    linear_hidden_size=256,
)
DEFAULT_LSTM_HIDDEN_SIZE = HYPERS['lstm_hidden_size']


class SharedTrunkModel(nn.Module):
    """
    One LSTM trunk serving four roles.

    Shared: `note_embeddings` (by voice position), `meta_embeddings` (by metadata
    channel, including voice_id when role_conditioned), `lstm_left` / `lstm_right`.
    Per role: `mlp_center[4]`, `mlp_predictions[4]` (vocab 129/110/115/129).
    """

    def __init__(self,
                 dataset: ChoraleDataset,
                 note_embedding_dim: int,
                 meta_embedding_dim: int,
                 num_layers: int,
                 lstm_hidden_size: int,
                 dropout_lstm: float,
                 hidden_size_linear=200,
                 models_dir: str = None,
                 role_conditioned: bool = False,
                 ):
        super().__init__()
        self.dataset = dataset
        self.note_embedding_dim = note_embedding_dim
        self.meta_embedding_dim = meta_embedding_dim
        self.num_notes_per_voice = [len(d) for d in dataset.note2index_dicts]
        self.num_voices = self.dataset.num_voices
        self.num_metas_per_voice = [
            metadata.num_values for metadata in dataset.metadatas
        ] + [self.num_voices]
        # The dataset's metadata channel count, voice_id included; the sampling interface slices tensors by it.
        self.num_metas = len(self.dataset.metadatas) + 1
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_lstm = dropout_lstm
        self.hidden_size_linear = hidden_size_linear
        self.role_conditioned = role_conditioned

        self.models_dir = models_dir or _DEFAULT_MODELS_DIR

        # How many metadata channels the trunk consumes. voice_id is the last channel
        # and enters the trunk only when role_conditioned; taken from len(metadatas)
        # rather than hardcoded to 3.
        self.trunk_metas = self.num_metas if role_conditioned \
            else len(self.dataset.metadatas)

        self.note_embeddings = nn.ModuleList(
            [nn.Embedding(num_notes, note_embedding_dim)
             for num_notes in self.num_notes_per_voice]
        )
        self.meta_embeddings = nn.ModuleList(
            [nn.Embedding(num_metas, meta_embedding_dim)
             for num_metas in self.num_metas_per_voice[:self.trunk_metas]]
        )
        trunk_input = (note_embedding_dim * self.num_voices
                       + meta_embedding_dim * self.trunk_metas)
        self.lstm_left = nn.LSTM(
            input_size=trunk_input,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True)
        self.lstm_right = nn.LSTM(
            input_size=trunk_input,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True)
        self.mlp_center = nn.ModuleList([
            nn.Sequential(
                nn.Linear(note_embedding_dim * (self.num_voices - 1)
                          + meta_embedding_dim * self.trunk_metas,
                          hidden_size_linear),
                nn.ReLU(),
                nn.Linear(hidden_size_linear, lstm_hidden_size))
            for _ in range(self.num_voices)
        ])
        self.mlp_predictions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.lstm_hidden_size * 3, hidden_size_linear),
                nn.ReLU(),
                nn.Linear(hidden_size_linear, num_notes))
            for num_notes in self.num_notes_per_voice
        ])

        if torch.cuda.is_available():
            self._scaler = (GradScaler(_AMP_DEVICE) if _AMP_DEVICE
                            else GradScaler())
        else:
            self._scaler = None

    # ---------------------------------------------------------------------------
    # Embedding — elementwise isomorphic to VoiceModel.embed, split into pieces
    # ---------------------------------------------------------------------------

    def _embed_notes(self, notes):
        """
        (B, T, num_voices) -> (B, T, num_voices * note_embedding_dim)

        Looks up by **voice position**, independent of voice_id, so all four roles
        share one set of embeddings.
        """
        batch_size, timesteps, num_voices = notes.size()
        return torch.cat([
            self.note_embeddings[vid](notes[:, :, vid])[:, :, None, :]
            for vid in range(num_voices)
        ], 2).view(batch_size, timesteps,
                   num_voices * self.note_embedding_dim)

    def _embed_center_notes(self, center_notes, other_voices_indexes):
        """
        (B, num_voices - 1) -> (B, 1, (num_voices - 1) * note_embedding_dim)

        `other_voices_indexes` fixes the subset and its order, so this step cannot be
        shared across roles.
        """
        batch_size = center_notes.size(0)
        return torch.cat([
            self.note_embeddings[vid](center_notes[:, k].unsqueeze(1))
            for k, vid in enumerate(other_voices_indexes)
        ], 1).view(batch_size, 1,
                   len(list(other_voices_indexes)) * self.note_embedding_dim)

    def _embed_metas(self, metas):
        """(B, T, trunk_metas) -> (B, T, trunk_metas * meta_embedding_dim)"""
        batch_size, timesteps, _ = metas.size()
        return torch.cat([
            self.meta_embeddings[vid](metas[:, :, vid])[:, :, None, :]
            for vid in range(self.trunk_metas)
        ], 2).view(batch_size, timesteps,
                   self.trunk_metas * self.meta_embedding_dim)

    def _embed_center_metas(self, center_metas):
        """(B, trunk_metas) -> (B, 1, trunk_metas * meta_embedding_dim)"""
        batch_size = center_metas.size(0)
        return torch.cat([
            self.meta_embeddings[vid](center_metas[:, vid].unsqueeze(1))
            for vid in range(self.trunk_metas)
        ], 1).view(batch_size, 1,
                   self.trunk_metas * self.meta_embedding_dim)

    def _trunk(self, left_notes, right_notes, left_metas, right_metas):
        """
        One trunk forward; returns (left_vec, right_vec), each (B, lstm_hidden_size).

        Role-independent, so `forward_all` runs the trunk only once.
        """
        batch_size = left_notes.size(0)
        left = torch.cat([self._embed_notes(left_notes),
                          self._embed_metas(left_metas)], 2)
        right = torch.cat([self._embed_notes(right_notes),
                           self._embed_metas(right_metas)], 2)
        hidden = init_hidden(self.num_layers, batch_size, self.lstm_hidden_size)
        left_vec, _ = self.lstm_left(left, hidden)
        hidden = init_hidden(self.num_layers, batch_size, self.lstm_hidden_size)
        right_vec, _ = self.lstm_right(right, hidden)
        return left_vec[:, -1, :], right_vec[:, -1, :]

    def _head(self, main_voice_index, left_vec, center_notes, center_metas,
              right_vec):
        """This role's center -> this role's prediction head."""
        center = torch.cat([self._embed_center_notes(
            center_notes, self.other_voices_indexes(main_voice_index)),
            self._embed_center_metas(center_metas)], 2)
        center_vec = self.mlp_center[main_voice_index](center[:, 0, :])
        return self.mlp_predictions[main_voice_index](
            torch.cat([left_vec, center_vec, right_vec], 1))

    def other_voices_indexes(self, main_voice_index):
        return [i for i in range(self.num_voices) if i != main_voice_index]

    # ---------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------

    def _time_major(self, notes):
        """
        (B, num_voices, T) -> (B, T, num_voices); left/right must be transposed to enter
        the LSTM, as with `ln.transpose(1, 2)` in `VoiceModel.forward`.

        center is not transposed: it feeds the MLP only, and the (B, num_voices - 1)
        it already has is exactly what `_embed_center_notes` wants.
        """
        left_notes, center_notes, right_notes = notes
        return (left_notes.transpose(1, 2), center_notes,
                right_notes.transpose(1, 2))

    def forward_role(self, main_voice_index, notes, metas):
        """
        One role's conditional distribution. `notes` / `metas` have the same shapes as
        VoiceModel.forward's inputs (left/right notes (B, num_voices, T), untransposed),
        and center is already the post-mask_entry (B, num_voices - 1). The sampling path
        goes through here (`RoleView.forward`).
        """
        left_notes, center_notes, right_notes = self._time_major(notes)
        left_metas, center_metas, right_metas = metas
        if self.role_conditioned:
            # voice_id enters the trunk, so each role has different trunk inputs and
            # needs its own forward.
            left_metas = self._with_voice_id(left_metas, main_voice_index)
            right_metas = self._with_voice_id(right_metas, main_voice_index)
            center_metas = self._with_voice_id(center_metas, main_voice_index)
        left_vec, right_vec = self._trunk(left_notes, right_notes,
                                          left_metas, right_metas)
        return self._head(main_voice_index, left_vec, center_notes,
                          center_metas, right_vec)

    def forward_all(self, left_notes, current_notes, right_notes, metas):
        """
        One trunk forward gives all four roles' conditional distributions at one tick.

        :param left_notes: (B, num_voices, T_left)
        :param current_notes: (B, num_voices), the **unmasked** whole column; masking is
            done here per role, since the four roles use different subsets.
        :param right_notes: (B, num_voices, T_right)
        :param metas: the (left, center, right) triple sliced from **one** voice. Legal
            because fermata / tick / key are elementwise identical across all four.
        :return: a length-4 list, each (B, num_notes_per_voice[r])
        """
        left_notes, _, right_notes = self._time_major(
            (left_notes, current_notes, right_notes))
        left_metas, center_metas, right_metas = metas
        # left/right metas enter the LSTM as (B, T, trunk_metas); center feeds the MLP
        # only and stays (B, trunk_metas). Do not broadcast center here, or
        # _embed_center_metas would receive a 3-D tensor.
        left_metas = self._as_sequence(left_metas)
        right_metas = self._as_sequence(right_metas)

        if not self.role_conditioned:
            # One trunk, four heads.
            left_vec, right_vec = self._trunk(left_notes, right_notes,
                                              left_metas, right_metas)
            return [self._head(r, left_vec,
                               mask_entry(current_notes, entry_index=r, dim=1),
                               center_metas, right_vec)
                    for r in range(self.num_voices)]

        outputs = []
        for r in range(self.num_voices):
            left_vec, right_vec = self._trunk(
                left_notes, right_notes,
                self._with_voice_id(left_metas, r),
                self._with_voice_id(right_metas, r))
            outputs.append(self._head(
                r, left_vec, mask_entry(current_notes, entry_index=r, dim=1),
                self._with_voice_id(center_metas, r), right_vec))
        return outputs

    def forward(self, *inputs):
        """The nn.Module entry point for `forward_all`, same signature."""
        return self.forward_all(*inputs)

    def _as_sequence(self, metas):
        """(B, trunk_metas) -> (B, 1, trunk_metas); returned as is if already 3-D."""
        if metas.dim() == 2:
            return metas[:, None, :]
        return metas

    def _with_voice_id(self, metas, main_voice_index):
        """
        Overwrite the last channel (voice_id) with main_voice_index.

        With role_conditioned=False the trunk consumes only the first trunk_metas
        channels, so the extra column is never looked up and this is the identity on
        that path.
        """
        metas = metas.clone()
        metas[..., -1] = main_voice_index
        return metas

    # ---------------------------------------------------------------------------
    # Sampling entry point: wrapping the shared model as four VoiceModel-shaped views
    # ---------------------------------------------------------------------------

    def role(self, main_voice_index):
        return RoleView(self, main_voice_index)

    # ---------------------------------------------------------------------------
    # Save / load
    # ---------------------------------------------------------------------------

    def save(self, models_dir=None):
        target_dir = models_dir or self.models_dir
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, self.__repr__())
        # Atomic write: being killed midway would leave a truncated file that `load()` rejects outright, and the old best would be gone.
        atomic_torch_save(self.state_dict(), save_path)
        print(f'Model {self.__repr__()} saved to {target_dir}')

    def load(self, models_dir=None):
        target_dir = models_dir or self.models_dir
        load_path = os.path.join(target_dir, self.__repr__())
        state_dict = torch.load(load_path,
                                map_location=lambda storage, loc: storage,
                                weights_only=True)
        print(f'Loading {self.__repr__()} from {target_dir}')
        self.load_state_dict(state_dict)

    def __repr__(self):
        # Must differ from VoiceModel's repr, or the joint model would load a single
        # role's old weights. hidden size is in the filename too, so a capacity change
        # is a file change.
        return (f'SharedTrunkModel('
                f'{self.dataset.__repr__()},'
                f'{self.note_embedding_dim},'
                f'{self.meta_embedding_dim},'
                f'{self.num_layers},'
                f'{self.lstm_hidden_size},'
                f'{self.dropout_lstm},'
                f'{self.hidden_size_linear},'
                f'{self.role_conditioned})')

    # ---------------------------------------------------------------------------
    # Training — one trunk forward, loss = CE0 + CE1 + CE2 + CE3
    # ---------------------------------------------------------------------------

    def preprocess_input(self, tensor_chorale, tensor_metadata):
        """
        Slice one tick's inputs plus the four roles' labels.

        Differs from VoiceModel.preprocess_input in two ways only: the random tick is
        drawn once (shared by the four roles), and four labels are returned, not one.
        """
        batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()
        offset = random.randint(0, self.dataset.subdivision)
        time_index_ticks = chorale_length_ticks // 2 + offset

        left_notes = tensor_chorale[:, :, :time_index_ticks]
        right_notes = reverse_tensor(tensor_chorale[:, :, time_index_ticks + 1:],
                                     dim=2)
        current_notes = tensor_chorale[:, :, time_index_ticks]
        labels = [current_notes[:, r] for r in range(self.num_voices)]

        # The three metadata channels are identical across voices, so any voice will
        # do; voice_id is overwritten in forward.
        metas = tensor_metadata[:, 0]
        left_metas = metas[:, :time_index_ticks, :]
        right_metas = reverse_tensor(metas[:, time_index_ticks + 1:, :], dim=1)
        center_metas = metas[:, time_index_ticks, :]

        return ((left_notes, current_notes, right_notes),
                (left_metas, center_metas, right_metas),
                labels)

    def loss_and_acc(self, dataloader, optimizer=None, phase='train', steps=None):
        average_loss = 0
        average_acc = [0.0] * self.num_voices
        batches_run = 0

        if phase == 'train':
            self.train()
        elif phase in ('eval', 'test'):
            self.eval()
        else:
            raise NotImplementedError(f'Unknown phase: {phase}')

        loss_fn = nn.CrossEntropyLoss()

        for tensor_chorale, tensor_metadata in dataloader:
            # `steps` caps the batches this pass runs; None = the whole dataloader.
            # Training must pass `steps_per_epoch`, or the budget arithmetic comes loose
            # from the real step count. Validation passes nothing and runs the full set:
            # val loss is both the best criterion and the `ReduceLROnPlateau` input.
            if steps is not None and batches_run >= steps:
                break

            tensor_chorale = cuda_variable(tensor_chorale).long()
            tensor_metadata = cuda_variable(tensor_metadata).long()

            for v in range(self.num_voices):
                max_idx = self.note_embeddings[v].num_embeddings - 1
                tensor_chorale[:, v, :] = tensor_chorale[:, v, :].clamp(0, max_idx)

            notes, metas, labels = self.preprocess_input(tensor_chorale,
                                                         tensor_metadata)

            def compute():
                outputs = self.forward_all(*notes, metas)
                return outputs, sum(loss_fn(outputs[r], labels[r])
                                    for r in range(self.num_voices))

            if self._scaler is not None and phase == 'train':
                with autocast(_AMP_DEVICE or 'cuda'):
                    outputs, loss = compute()
                optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                outputs, loss = compute()
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            average_loss += loss.item()
            for r in range(self.num_voices):
                average_acc[r] += self.accuracy(outputs[r], labels[r]).item()
            batches_run += 1

        # The denominator is the **number of batches actually run**, not len(dataloader);
        # keeping the latter once the `steps` above truncates training would silently
        # shrink loss and acc.
        # This count is also where global_step comes from (not from the optimizer's step
        # counter, which a mid-run optimizer rebuild zeroes and AMP skips), counted in
        # **batches**.
        return (average_loss / batches_run,
                [acc / batches_run for acc in average_acc],
                batches_run)

    def accuracy(self, weights, target):
        batch_size, = target.size()
        pred = nn.Softmax(dim=1)(weights).max(1)[1].type_as(target)
        return (pred == target).float().sum() / batch_size * 100

    def train_model(self,
                    batch_size=16,
                    steps_per_epoch=None,
                    max_steps=None,
                    optimizer=None,
                    lr_patience: int = 3,
                    lr_factor: float = 0.5,
                    ):
        """
        Train the shared trunk; the budget counts **optimizer steps**, not epochs.

        `steps_per_epoch=None` -> the batch count of one complete pass; `max_steps=None`
        -> `DEFAULT_TOTAL_PASSES x steps_per_epoch`. Both are fixed by
        `resolve_training_budget`, which rounds `max_steps` up to a whole segment, so
        the loop is written as a pass loop.

        A training pass runs only `steps_per_epoch` steps; a validation pass is **not
        truncated**, because val loss is both the best-checkpoint criterion and the input
        to `ReduceLROnPlateau`. `num_epochs` / `resume` are removed with no compatibility
        alias.

        `val_acc` is a **per-role list** (a scalar in VoiceModel) and history stores its
        mean, so the two architectures' `val_acc` columns are not directly comparable.

        Paired with `VoiceModel.train_model`: changing one means changing the other.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=lr_patience, factor=lr_factor)

        # The four roles share one data split, so the DataLoaders are built once only.
        dataloader_train, dataloader_val, _ = self.dataset.data_loaders(
            batch_size=batch_size)

        steps_per_epoch, max_steps, total_passes = resolve_training_budget(
            steps_per_epoch, max_steps, len(dataloader_train))

        best_val_loss = float('inf')
        history = []
        global_step = 0

        print(f'  budget: {total_passes} pass(es) x {steps_per_epoch} step(s) '
              f'= {max_steps} step(s)')

        for epoch in range(total_passes):
            print(f'=== SharedTrunkModel — Pass {epoch} '
                  f'(step {global_step}/{max_steps}) ===')

            loss, acc, n_train = self.loss_and_acc(dataloader_train,
                                                   optimizer=optimizer,
                                                   phase='train',
                                                   steps=steps_per_epoch)
            print(f'  Train  loss: {loss:.4f}  '
                  f'acc: {[f"{a:.2f}" for a in acc]}')

            # The third return value is the number of batches actually run, used by the
            # training pass only (added into global_step); global_step counts optimizer
            # steps, so it cannot come from validation.
            val_loss, val_acc, _ = self.loss_and_acc(dataloader_val,
                                                     optimizer=None,
                                                     phase='test')
            print(f'  Val    loss: {val_loss:.4f}  '
                  f'acc: {[f"{a:.2f}" for a in val_acc]}')

            global_step += n_train

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(models_dir=sibling_snapshot_dir(self.models_dir,
                                                          'best'))
                # ASCII only, for the reason given in voice_model.py: '✓' will not encode in GBK.
                print(f'  [ok] Best model saved (val_loss={val_loss:.4f})')

            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f'  LR reduced: {old_lr:.2e} → {new_lr:.2e}')

            save_pass_snapshot(self, self.models_dir, epoch + 1)

            history.append(make_history_row(
                global_step, epoch, loss, val_loss,
                sum(val_acc) / len(val_acc), optimizer.param_groups[0]['lr']))
            # Unconditional at the end of every pass; the only writer of the loss curve,
            # so do not add a condition. Note val_acc here is the per-role **mean**, not
            # VoiceModel's scalar.
            save_history(self, self.models_dir,
                         batch_size=batch_size,
                         steps_per_epoch=steps_per_epoch,
                         max_steps=max_steps,
                         global_step=global_step,
                         best_val_loss=best_val_loss,
                         history=history)

        print(f'SharedTrunkModel done. Best val loss: {best_val_loss:.4f}')


class SharedTrunkDeepBach(DeepBach):
    """
    The Plan A manager.

    It subclasses `DeepBach` because the sampling logic (`parallel_gibbs` /
    `generation`) is unchanged: it sees the model only through three methods on
    `voice_models[i]`, and `RoleView` satisfies that interface. So only construction
    and training are replaced:

      * `__init__`  — build **one** shared model, then treat four RoleViews as the four
        voice models
      * `train`     — one optimizer, one trunk forward, loss = CE0+CE1+CE2+CE3

    `eval_phase` / `train_phase` / `generation` / `parallel_gibbs` / `cuda` /
    `load` / `save` are all inherited as they are.
    """

    def __init__(self,
                 dataset,
                 note_embedding_dim,
                 meta_embedding_dim,
                 num_layers,
                 lstm_hidden_size,
                 dropout_lstm,
                 linear_hidden_size,
                 models_dir=None,
                 role_conditioned=False,
                 ):
        # No super().__init__(): it would build four independent VoiceModels, exactly
        # what Plan A removes.
        self.dataset = dataset
        self.num_voices = self.dataset.num_voices
        self.num_metas = len(self.dataset.metadatas) + 1
        self.activate_cuda = torch.cuda.is_available()

        self.model = SharedTrunkModel(
            dataset=self.dataset,
            note_embedding_dim=note_embedding_dim,
            meta_embedding_dim=meta_embedding_dim,
            num_layers=num_layers,
            lstm_hidden_size=lstm_hidden_size,
            dropout_lstm=dropout_lstm,
            hidden_size_linear=linear_hidden_size,
            models_dir=models_dir,
            role_conditioned=role_conditioned,
        )
        self.voice_models = [self.model.role(i)
                             for i in range(self.num_voices)]

    @property
    def shared_model(self):
        return self.model

    @property
    def models_dir(self):
        # All four RoleViews have this property (each pointing at `self._model`), and one
        # is needed here too: `db._checkpoint_paths(model)` finds the directory through
        # `getattr(model, 'models_dir', None)`, and without it `os.path.abspath(None)`
        # raises TypeError.
        return self.model.models_dir

    # The three overrides below only remove redundancy: the parent repeats per role, but
    # the four roles share one set of weights, so save / load / moving devices once each
    # is enough (otherwise the same path is written four times).
    #
    # `models_dir` must be forwarded explicitly: `main_voice_index` is the first
    # positional parameter and is ignored, so `model.load(some_dir)` would silently bind
    # to it, weights seeming to load while the constructor's directory is really read.
    def cuda(self, main_voice_index=None):
        if self.activate_cuda:
            self.model.cuda()

    def load(self, main_voice_index=None, models_dir=None):
        self.model.load(models_dir=models_dir)

    def save(self, main_voice_index=None, models_dir=None):
        self.model.save(models_dir=models_dir)

    @classmethod
    def build(cls, dataset, models_dir=None, role_conditioned=False,
              lstm_hidden_size=None, **overrides):
        """
        Build the shared trunk model from `HYPERS` (no weight loading, no device move).

        `lstm_hidden_size=None` -> `DEFAULT_LSTM_HIDDEN_SIZE` (256). Passing another
        value explicitly is not an error, just an experiment, and prints a one-line note.

        `**overrides` overrides the rest of `HYPERS`; the CLI's `--note_embedding_dim` /
        `--num_layers` come in here.
        """
        unknown = sorted(set(overrides) - set(HYPERS))
        if unknown:
            raise TypeError(f'unknown hyperparameters: {unknown}; '
                            f'HYPERS has {sorted(HYPERS)}')
        if lstm_hidden_size is None:
            lstm_hidden_size = DEFAULT_LSTM_HIDDEN_SIZE
        elif not isinstance(lstm_hidden_size, int) or lstm_hidden_size <= 0:
            raise ValueError(
                f'lstm_hidden_size must be a positive int or None, '
                f'got {lstm_hidden_size!r}')
        elif lstm_hidden_size != DEFAULT_LSTM_HIDDEN_SIZE:
            print(f'[SharedTrunkDeepBach] note: lstm_hidden_size='
                  f'{lstm_hidden_size} differs from the shared-trunk default '
                  f'{DEFAULT_LSTM_HIDDEN_SIZE}; one trunk serving four roles '
                  f'divides the recurrent capacity by four.')

        return cls(
            dataset=dataset,
            models_dir=models_dir if models_dir is not None
            else _DEFAULT_MODELS_DIR,
            role_conditioned=role_conditioned,
            **{**HYPERS, **overrides, 'lstm_hidden_size': lstm_hidden_size},
        )

    def train(self,
              main_voice_index=None,
              lr: float = 1e-3,
              lr_patience: int = 3,
              lr_factor: float = 0.5,
              **kwargs):
        """
        Train the shared model.

        `main_voice_index` is accepted but **ignored**: the four roles share one set of
        weights, so training one alone means nothing. The parameter is kept only so
        `DeepBach`'s call sites need no change.

        `skip_voices` is removed; a caller passing it still gets a TypeError from
        `train_model`.
        """
        if main_voice_index is not None:
            print(f'[SharedTrunkDeepBach] main_voice_index={main_voice_index} '
                  f'ignored: the trunk is shared, training is joint.')
        if self.activate_cuda:
            self.model.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train_model(
            optimizer=optimizer,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            **kwargs,
        )


class RoleView:
    """
    A "single-role facade" over the shared trunk.

    Forwards `preprocess_notes` / `preprocess_metas` / `forward` / `eval()` / `train()` /
    `load()` / `save()` / `cuda()` to the shared model, so the sampler and
    `model_manager.py` can be reused as they are.
    """

    def __init__(self, model: SharedTrunkModel, main_voice_index: int):
        self._model = model
        self.main_voice_index = main_voice_index

    # --- Sampling interface (shapes as in VoiceModel) ------------------------

    def preprocess_notes(self, tensor_chorale, time_index_ticks):
        left_notes = tensor_chorale[:, :, :time_index_ticks]
        right_notes = reverse_tensor(tensor_chorale[:, :, time_index_ticks + 1:],
                                     dim=2)
        central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks],
                                   entry_index=self.main_voice_index,
                                   dim=1)
        label = tensor_chorale[:, self.main_voice_index, time_index_ticks]
        # Must be a (notes, label) pair: the caller writes `notes, _ = ...`.
        return (left_notes, central_notes, right_notes), label

    def preprocess_metas(self, tensor_metadata, time_index_ticks):
        metas = tensor_metadata[:, self.main_voice_index]
        if not self._model.role_conditioned:
            # voice_id is the last channel and does not enter the trunk; the first three
            # channels are elementwise identical across voices, so any voice is equivalent.
            metas = metas[..., :self._model.trunk_metas]
        left_metas = metas[:, :time_index_ticks, :]
        right_metas = reverse_tensor(metas[:, time_index_ticks + 1:, :], dim=1)
        center_metas = metas[:, time_index_ticks, :]
        return left_metas, center_metas, right_metas

    def forward(self, *inputs):
        notes, metas = inputs
        return self._model.forward_role(self.main_voice_index, notes, metas)

    # --- Let the rest of DeepBach work as it is ------------------------------

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def cuda(self):
        self._model.cuda()

    def parameters(self):
        return self._model.parameters()

    def load(self, models_dir=None):
        self._model.load(models_dir=models_dir)

    def save(self, models_dir=None):
        self._model.save(models_dir=models_dir)

    @property
    def dataset(self):
        return self._model.dataset

    @property
    def models_dir(self):
        return self._model.models_dir

    def __repr__(self):
        return f'{self._model!r}[role {self.main_voice_index}]'
