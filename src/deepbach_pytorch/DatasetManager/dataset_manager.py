import os

import music21
import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.helpers import ShortChoraleIteratorGen, atomic_torch_save
from DatasetManager.metadata import TickMetadata, FermataMetadata, KeyMetadata
from DatasetManager.music_dataset import MusicDataset

all_datasets = {
    'bach_chorales': {
        'dataset_class_name': ChoraleDataset,
        'corpus_it_gen':      music21.corpus.chorales.Iterator
    },
    'bach_chorales_test': {
        'dataset_class_name': ChoraleDataset,
        'corpus_it_gen':      ShortChoraleIteratorGen()
    },
}


class DatasetManager:
    def __init__(self):
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_dir = os.path.join(os.path.dirname(self.package_dir),
                                      'data', 'dataset_cache')
        # makedirs rather than mkdir: it also creates the parent when it is missing
        # (a brand-new install directory).
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_dataset(self, name: str, **dataset_kwargs) -> MusicDataset:
        if name in all_datasets:
            return self.load_if_exists_or_initialize_and_save(
                name=name,
                **all_datasets[name],
                **dataset_kwargs
            )
        else:
            print(f'Dataset with name {name} is not registered in all_datasets variable')
            raise ValueError

    def load_if_exists_or_initialize_and_save(self,
                                              dataset_class_name,
                                              corpus_it_gen,
                                              name,
                                              **kwargs):
        kwargs.update({
            'name':          name,
            'corpus_it_gen': corpus_it_gen,
            'cache_dir':     self.cache_dir,
        })
        dataset = dataset_class_name(**kwargs)

        if os.path.exists(dataset.filepath):
            print(f'Loading {dataset.__repr__()} from {dataset.filepath}')
            # weights_only=False: the cache holds a Dataset object, not pure tensors.
            #
            # Catch Exception, not RuntimeError: a truncated pickle raises
            # RuntimeError, a zero-byte one raises EOFError, and both cases are
            # handled the same way -- rebuild.
            try:
                loaded = torch.load(dataset.filepath, weights_only=False)
                loaded.cache_dir = self.cache_dir
                dataset = loaded
                print(f'(the corresponding TensorDataset is not loaded)')
            except Exception as exc:
                print(f'Warning: cached dataset is unreadable '
                      f'({type(exc).__name__}: {exc}); rebuilding.')
                if os.path.exists(dataset.tensor_dataset_filepath):
                    os.remove(dataset.tensor_dataset_filepath)
                tensor_dataset = dataset.tensor_dataset
                dataset.tensor_dataset = None
                atomic_torch_save(dataset, dataset.filepath)
                print(f'{dataset.__repr__()} saved in {dataset.filepath}')
                dataset.tensor_dataset = tensor_dataset
        else:
            print(f'Creating {dataset.__repr__()}, '
                  f'both tensor dataset and parameters')
            if os.path.exists(dataset.tensor_dataset_filepath):
                os.remove(dataset.tensor_dataset_filepath)
            # triggers make_tensor_dataset() and caches it
            tensor_dataset = dataset.tensor_dataset
            # the dataset parameters and the tensor cache are saved separately
            dataset.tensor_dataset = None
            atomic_torch_save(dataset, dataset.filepath)
            print(f'{dataset.__repr__()} saved in {dataset.filepath}')
            dataset.tensor_dataset = tensor_dataset

        return dataset


if __name__ == '__main__':
    dataset_manager = DatasetManager()
    subdivision = 4
    metadatas = [
        TickMetadata(subdivision=subdivision),
        FermataMetadata(),
        KeyMetadata()
    ]
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales_test',
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=8,
        subdivision=subdivision
    )
    (train_dataloader, val_dataloader, test_dataloader) = \
        bach_chorales_dataset.data_loaders(batch_size=128, split=(0.85, 0.10))
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))
