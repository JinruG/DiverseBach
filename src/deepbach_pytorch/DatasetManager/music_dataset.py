from abc import ABC, abstractmethod
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
import torch

from DatasetManager.helpers import atomic_torch_save

# On Windows num_workers > 0 needs the 'spawn' start method, which fails when
# running as a package. Linux/macOS use 'fork' and can be > 0.
_TRAIN_NUM_WORKERS = 0 if sys.platform == 'win32' else 4
_USE_PIN_MEMORY = torch.cuda.is_available()


def cast_collate(batch):
    """
    Stack one batch, widening back to int64.

    :param batch: a list of (score_tensor, metadata_tensor)
    :return: (notes, metas), both int64
    """
    notes = torch.stack([sample[0] for sample in batch])
    metas = torch.stack([sample[1] for sample in batch])
    return notes.long(), metas.long()


class MusicDataset(ABC):
    """
    Abstract Base Class for music datasets
    """

    def __init__(self, cache_dir):
        self._tensor_dataset = None
        self.cache_dir = cache_dir

    @abstractmethod
    def iterator_gen(self): pass

    @abstractmethod
    def make_tensor_dataset(self): pass

    @abstractmethod
    def get_score_tensor(self, score): pass

    @abstractmethod
    def get_metadata_tensor(self, score): pass

    @abstractmethod
    def transposed_score_and_metadata_tensors(self, score, semi_tone): pass

    @abstractmethod
    def extract_score_tensor_with_padding(self, tensor_score, start_tick, end_tick): pass

    @abstractmethod
    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick): pass

    @abstractmethod
    def empty_score_tensor(self, score_length): pass

    @abstractmethod
    def random_score_tensor(self, score_length): pass

    @abstractmethod
    def tensor_to_score(self, tensor_score): pass

    @property
    def tensor_dataset(self):
        """Loads or builds (and caches) the TensorDataset."""
        if self._tensor_dataset is None:
            if self.tensor_dataset_is_cached():
                print(f'Loading TensorDataset for {self.__repr__()}')
                # weights_only=False: the cache holds a TensorDataset object, not a
                # pure tensor dict.
                #
                # Catch Exception, not RuntimeError: this cache is about 2.7 GB and
                # rebuilding takes 45 minutes, while `tensor_dataset_is_cached()` is
                # only `os.path.exists`, so a truncated file passes for valid and is
                # refused here. A truncated one raises RuntimeError, a zero-byte one
                # raises EOFError, and catching narrowly would let the latter escape
                # as a crash instead of taking the rebuild path.
                try:
                    self._tensor_dataset = torch.load(
                        self.tensor_dataset_filepath, weights_only=False)
                except Exception as exc:
                    print(f'Warning: cached TensorDataset is unreadable '
                          f'({type(exc).__name__}: {exc}); rebuilding.')
                    try:
                        os.remove(self.tensor_dataset_filepath)
                    except OSError:
                        pass
                    self._tensor_dataset = self._build_and_cache_tensor_dataset()
            else:
                print(f'Creating {self.__repr__()} TensorDataset'
                      f' since it is not cached')
                self._tensor_dataset = self._build_and_cache_tensor_dataset()
        return self._tensor_dataset

    def _build_and_cache_tensor_dataset(self):
        tensor_dataset = self.make_tensor_dataset()
        filepath = self.tensor_dataset_filepath
        # atomic write: a half-written cache is the worst outcome -- it exists, so
        # nobody rebuilds, and torch.load refuses it.
        atomic_torch_save(tensor_dataset, filepath)
        print(f'TensorDataset for {self.__repr__()} '
              f'saved in {filepath}')
        return tensor_dataset

    @tensor_dataset.setter
    def tensor_dataset(self, value):
        self._tensor_dataset = value

    def tensor_dataset_is_cached(self):
        return os.path.exists(self.tensor_dataset_filepath)

    @property
    def tensor_dataset_filepath(self):
        tensor_datasets_cache_dir = os.path.join(self.cache_dir, 'tensor_datasets')
        # makedirs rather than mkdir: it also creates the parent when it is missing
        os.makedirs(tensor_datasets_cache_dir, exist_ok=True)
        return os.path.join(tensor_datasets_cache_dir, self.__repr__())

    @property
    def filepath(self):
        datasets_cache_dir = os.path.join(self.cache_dir, 'datasets')
        os.makedirs(datasets_cache_dir, exist_ok=True)
        return os.path.join(datasets_cache_dir, self.__repr__())

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        """
        Return the three DataLoaders (train, val, eval).

        val / eval use num_workers=0 and no pin_memory.
        """
        assert sum(split) < 1

        dataset = self.tensor_dataset
        num_examples = len(dataset)
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset   = TensorDataset(*dataset[int(a * num_examples):
                                               int((a + b) * num_examples)])
        eval_dataset  = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=_TRAIN_NUM_WORKERS,
            pin_memory=_USE_PIN_MEMORY,
            persistent_workers=(_TRAIN_NUM_WORKERS > 0),
            drop_last=True,
            collate_fn=cast_collate,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=cast_collate,
        )
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=cast_collate,
        )
        return train_dl, val_dl, eval_dl
