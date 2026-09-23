"""
@author: Gaetan Hadjeres

DeepBach training / generation entry point.  Corpus: the built-in Bach chorales (361 after
`DatasetManager.chorale_filter`).

    python deepBach.py                              # scheme A (shared trunk), 20 passes
    python deepBach.py --arch baseline --train       # four independent VoiceModels

`--arch` is the real fork: shared_trunk -> `SharedTrunkDeepBach` (one shared LSTM trunk +
four role heads); baseline -> `DeepBach` (four independent VoiceModels).  Both share one
tensor cache — arch changes the model, not the corpus, so scheme A needs no cache of its
own.
"""

import os
import sys

import click

# `_THIS_DIR` makes the **top-level** `DatasetManager` / `DeepBach` resolve when this file
# is run directly; `_SRC_DIR` must be inserted here too, or the `import deepbach_pytorch as
# db` below silently lands on the site-packages copy (it imports without error, but every
# `db.*` afterwards runs against another package, see `selfcheck.import_shadow_guard`).
# Both go to the front of the list; their relative order does not matter.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deepbach_pytorch as db
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

# Models always come from the package (`db.DeepBach`), not `from DeepBach.model_manager
# import DeepBach`: the package root is on `sys.path`, so that file has two module names
# and executes twice into **two different class objects**; a model built under the absolute
# name fails `isinstance(..., SharedTrunkDeepBach)` (which is exactly what
# `db._checkpoint_paths` / `db._load_pretrained_weights` branch on).

# The two architectures are stored apart: `data/models_<snapshot>/baseline` holds the four
# VoiceModel weights, `.../shared_trunk` holds scheme A's single file.  Both reprs end in
# `)` and weights are looked up by `endswith`, so sharing one directory means picking the
# model by filesystem traversal order (see `_voice_weight_suffix`).
#
# Both constants point at the `DEFAULT_SNAPSHOT` set; snapshot sets are sibling
# directories, switched with `db.snapshot_dir(name, arch)`.  Taken from the package rather
# than re-derived here: a re-derivation is a second literal that can drift.
_MODELS_DIR = db.MODELS_DIR
_SHARED_TRUNK_MODELS_DIR = db.SHARED_TRUNK_MODELS_DIR


@click.command()
@click.option('--arch', default='shared_trunk',
              type=click.Choice(['shared_trunk', 'baseline']),
              help='shared_trunk = scheme A (one trunk, four role heads); '
                   'baseline = four independent VoiceModels')
@click.option('--role_conditioned', is_flag=True,
              help='shared_trunk only: let voice_id into the trunk')
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=None, type=int,
              help='hidden size of the LSTMs (default: 256, both architectures)')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=512,
              help='training batch size')
@click.option('--steps_per_epoch', default=None, type=int,
              help='optimizer steps per pass over the training set '
                   '(default: len(train_loader))')
@click.option('--max_steps', default=None, type=int,
              help='total optimizer steps (default: DEFAULT_TOTAL_PASSES '
                   'passes, currently 20). Rounded up to a whole number of '
                   'passes; the effective value is printed')
@click.option('--lr', default=1e-3,
              help='initial learning rate for Adam')
@click.option('--lr_patience', default=3,
              help='ReduceLROnPlateau patience (passes without val improvement)')
@click.option('--lr_factor', default=0.5,
              help='ReduceLROnPlateau reduction factor')
@click.option('--train', 'do_train', is_flag=True,
              help='train the specified model for max_steps')
@click.option('--no_cudnn', is_flag=True,
              help='disable cuDNN (its LSTM crashes at teardown on some '
                   'Windows builds after a successful run)')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
def main(arch,
         role_conditioned,
         note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         steps_per_epoch,
         max_steps,
         lr,
         lr_patience,
         lr_factor,
         do_train,
         no_cudnn,
         num_iterations,
         sequence_length_ticks,
         ):
    if no_cudnn:
        import torch
        torch.backends.cudnn.enabled = False

    dataset, deepbach = _build_bach(
        arch=arch,
        role_conditioned=role_conditioned,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size,
        batch_size=batch_size, steps_per_epoch=steps_per_epoch,
        max_steps=max_steps, lr=lr,
        lr_patience=lr_patience, lr_factor=lr_factor, do_train=do_train,
    )

    if not do_train:
        return

    deepbach.load()
    deepbach.cuda()

    print('Generation')
    score, tensor_chorale, tensor_metadata = deepbach.generation(
        num_iterations=num_iterations,
        sequence_length_ticks=sequence_length_ticks,
    )
    score.show('txt')
    score.show()


def _missing_checkpoints(deepbach):
    """
    Training weights that should exist on disk but do not.

    The paths come from `db._checkpoint_paths` (the single definition of model checkpoint
    naming, and the one `check_pretrained_weights` uses).  No arch branch here: the shared
    trunk's `voice_models` are four `RoleView`s whose `__repr__` looks like
    `SharedTrunkModel(...)[role 0]` — a human-readable label, not a filename — so
    assembling paths the baseline way would report all four as missing.
    """
    return [p for p in db._checkpoint_paths(deepbach)
            if not os.path.exists(p)]


def _build_bach(arch, role_conditioned, note_embedding_dim, meta_embedding_dim,
                num_layers, lstm_hidden_size, dropout_lstm, linear_hidden_size,
                batch_size, steps_per_epoch, max_steps, lr, lr_patience,
                lr_factor, do_train):
    """
    The built-in Bach chorales route.

    `lstm_hidden_size=None` is resolved by `build()` per architecture (256 for both).  Do
    not write `lstm_hidden_size or 256`: `0` would be swallowed as the default and the
    capacity silently rewritten.

    `steps_per_epoch` and `max_steps` both None is the reference budget; the real values
    are computed by `training_state.resolve_training_budget` from the current cache and
    printed, so nothing is hardcoded here.
    """
    dataset_manager = DatasetManager()

    metadatas = [
        FermataMetadata(),
        TickMetadata(subdivision=4),
        KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }
    dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
    )

    # The dimensions go in via `**overrides` rather than named parameters: `HYPERS` /
    # `BASELINE_HYPERS` are the only definition and the CLI flags merely override them, so
    # `--num_layers 3` either takes effect or raises TypeError inside `build()`; it cannot
    # silently do nothing.
    dims = dict(
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size,
    )

    if arch == 'shared_trunk':
        deepbach = db.SharedTrunkDeepBach.build(
            dataset=dataset,
            models_dir=_SHARED_TRUNK_MODELS_DIR,
            role_conditioned=role_conditioned,
            lstm_hidden_size=lstm_hidden_size,
            **dims,
        )
    elif arch == 'baseline':
        deepbach = db.DeepBach.build(
            dataset=dataset,
            models_dir=_MODELS_DIR,
            lstm_hidden_size=lstm_hidden_size,
            **dims,
        )
    else:
        # Same shape as `db.train_from_scratch`: an unknown arch raises outright, never
        # falls back to baseline.
        raise ValueError(f'unknown arch {arch!r}; '
                         f'expected "shared_trunk" or "baseline"')

    if do_train:
        print(f'Training {arch} (batch_size={batch_size}, lr={lr}) ...')
        deepbach.cuda()
        deepbach.train(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            max_steps=max_steps,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )
        # Training writes db.SNAPSHOT_NAMES weight sets, each in its own sibling directory
        # (see `DatasetManager.helpers`).  The `deepbach.load()` below reads `models_dir`,
        # i.e. the `DEFAULT_SNAPSHOT` set, not best.
        base = _SHARED_TRUNK_MODELS_DIR if arch == 'shared_trunk' else _MODELS_DIR
        print('Snapshots written:')
        for _name in db.SNAPSHOT_NAMES:
            print(f'  [{_name}] {db.sibling_snapshot_dir(base, _name)}')
        print(f'Loading {db.DEFAULT_SNAPSHOT} for generation.')

        missing = _missing_checkpoints(deepbach)
        if missing:
            print('\n[ERROR] Pretrained model weights not found:')
            for p in missing:
                print(f'  {p}')
            print('\nTrain from scratch first with:')
            # No hardcoded `--max_steps`: any literal holds for one cache only.  Passing
            # neither budget flag means the reference budget, whose real values
            # `resolve_training_budget` computes from the cache.
            print('  python deepBach.py --train --batch_size 512\n')
            raise SystemExit(1)

    return dataset, deepbach


DO_TRAIN = True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # With explicit arguments the command line wins: without this fork the hardcoded
        # args below would override sys.argv, and `--help` would start training instead of
        # printing help.
        main()
    else:
        # No arguments = the delivery default: scheme A + the reference budget.
        #
        # Deliberately no `--steps_per_epoch` / `--max_steps`: both None is that reference
        # budget, computed by `resolve_training_budget` from the current cache and printed
        # at the start of training.
        args = [
            '--arch',                  'shared_trunk',
            '--batch_size',            '512',
            '--lr',                    '1e-3',
            '--lr_patience',           '3',
            '--lr_factor',             '0.5',
            '--num_iterations',        '500',
            '--sequence_length_ticks', '64',
        ]
        if DO_TRAIN:
            args.append('--train')

        main(standalone_mode=True, args=args)
