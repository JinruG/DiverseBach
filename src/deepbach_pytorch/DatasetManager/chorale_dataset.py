import music21
import torch
import numpy as np

from music21 import interval, stream
from torch.utils.data import TensorDataset
from tqdm import tqdm

from DatasetManager.helpers import standard_name, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, \
    standard_note, OUT_OF_RANGE, REST_SYMBOL
from DatasetManager import chorale_filter
from DatasetManager.metadata import FermataMetadata
from DatasetManager.music_dataset import MusicDataset

#: Storage precision of the cache. The tensors hold only small integer indices
#: (a vocabulary of about 130 entries), so int16 is lossless, and it halves both
#: the disk footprint and the resident memory. `cast_collate` turns them back
#: into int64 for the embedding layer.
_STORAGE_DTYPES = {
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
}
_DTYPE_MAX = {name: torch.iinfo(dtype).max
              for name, dtype in _STORAGE_DTYPES.items()}


def resolve_storage_dtype(storage_dtype):
    """Map a storage-dtype name (or a torch dtype) to a torch dtype."""
    if isinstance(storage_dtype, torch.dtype):
        return storage_dtype
    try:
        return _STORAGE_DTYPES[storage_dtype]
    except KeyError:
        raise ValueError(f'unknown storage_dtype {storage_dtype!r}; '
                         f'expected one of {sorted(_STORAGE_DTYPES)}')


class ChoraleDataset(MusicDataset):
    """
    Class for all chorale-like datasets
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 voice_ids,
                 metadatas=None,
                 sequences_size=8,
                 subdivision=4,
                 cache_dir=None,
                 transposition_semitones=None,
                 window_stride=1,
                 storage_dtype='int32'):
        """
        :param transposition_semitones: the set of transposition amounts (in
            semitones) to build windows for, or None to take every offset in each
            window's feasible range (min..max). An explicit set is filtered per
            window down to the feasible ones.
        :param window_stride: step between adjacent window starts, in units of
            `window_step()`. 1 means tiling the whole piece on the finest grid.
        :param storage_dtype: name of the dtype the cache uses.
        """
        super(ChoraleDataset, self).__init__(cache_dir=cache_dir)
        self.voice_ids = voice_ids
        self.num_voices = len(voice_ids)
        self.name = name
        self.sequences_size = sequences_size
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.corpus_it_gen = corpus_it_gen
        self.voice_ranges = None
        self.metadatas = metadatas
        self.subdivision = subdivision
        self.transposition_semitones = (
            tuple(transposition_semitones)
            if transposition_semitones is not None else None)
        self.window_stride = window_stride
        self.storage_dtype = storage_dtype
        # Filled by compute_window_layout(); window_counts[i] is the number of
        # windows the i-th movement contributes, and their sum is the number of
        # samples in the cache. The builder preallocates by that number, so a
        # miscount is a hard error rather than a silently truncated cache.
        self.window_counts = None
        self.corpus_layout = None

    def extended_params_are_default(self):
        """Whether all the quartet-era parameters are at their defaults.

        The repr is both the cache filename and part of the weight filename, so a
        default-constructed dataset must keep the original five-field repr byte
        for byte; only a non-default configuration appends new fields.
        """
        return (self.transposition_semitones is None
                and self.window_stride == 1
                and self.storage_dtype == 'int32')

    def __repr__(self):
        # The class name rather than a literal: ChoraleBeatsDataset shares this
        # repr layout, and the suffix rules below have to cover it too.
        base = (f'{type(self).__name__}('
                f'{self.voice_ids},'
                f'{self.name},'
                f'{[metadata.name for metadata in self.metadatas]},'
                f'{self.sequences_size},'
                f'{self.subdivision}')
        if self.extended_params_are_default():
            return base + ')'
        return (base +
                f',{self.transposition_semitones},'
                f'{self.window_stride},'
                f'{self.storage_dtype})')

    def window_step(self):
        """
        Distance between adjacent window starts, in quarter notes.

        A tick-level dataset is one tick; ChoraleBeatsDataset overrides it to one measure.
        """
        return 1 / self.subdivision

    def window_positions(self, lowest_offset, highest_offset):
        """
        All window start offsets of one movement.

        The first window ends `sequences_size` after its start, so the earliest
        window has to come one window width before the first note -- that is the
        `-(sequences_size - window_step())` term below.
        """
        step = self.window_step()
        return np.arange(lowest_offset - (self.sequences_size - step),
                         highest_offset,
                         step * self.window_stride)

    def transposition_offsets(self, min_t, max_t):
        """
        Which semitone offsets to build for a window whose feasible range is
        [min_t, max_t].

        Without an explicit set it is the range itself; with one it is the
        intersection with that range, iterated in the set's own order.
        """
        if self.transposition_semitones is None:
            return range(min_t, max_t + 1)
        return [s for s in self.transposition_semitones if min_t <= s <= max_t]

    def flat_parts(self, chorale):
        """
        Flatten each voice once.

        The returned list is in voice order, for `voice_range_in_subsequence`
        and the like.
        """
        return [part.flatten() for part in chorale.parts[:self.num_voices]]

    def movement_bounds(self, chorale):
        """
        The (lowest, highest) offset of one movement, in quarter notes.

        Computed once per movement.
        """
        flat = chorale.flatten()
        return flat.lowestOffset, flat.highestOffset

    def compute_window_layout(self):
        """
        Count how many windows each movement contributes, before any build.

        This is what makes exact preallocation possible. It repeats the position
        / transposition logic of the fill pass rather than caching each
        position's result: storing every (position, semitones) pair at corpus
        scale costs more in memory than recomputing costs in time.
        """
        print('Computing window layout')
        self.window_counts = []
        self.corpus_layout = []
        for chorale in tqdm(self.iterator_gen()):
            flat = self.flat_parts(chorale)
            lowest, highest = self.movement_bounds(chorale)
            self.corpus_layout.append((lowest, highest))
            count = 0
            for offset_start in self.window_positions(lowest, highest):
                offset_end = offset_start + self.sequences_size
                ranges = self.voice_range_in_subsequence(flat, offset_start,
                                                         offset_end)
                min_t, max_t = self.min_max_transposition(ranges)
                count += len(self.transposition_offsets(min_t, max_t))
            self.window_counts.append(count)

    def iterator_gen(self):
        """
        The **single entry point** for corpus iteration: the fill loops of
        `compute_index_dicts` / `compute_window_layout` / `compute_voice_ranges`
        / `make_tensor_dataset`, and the four stages of `_retrain.py`, all go
        through here, so the filter rules hang off this one place only.

        Order:
          1. `with_source` -- the only step that returns a **different object**
             (pieces whose file is pinned in `PART_SOURCES`); the rules have to
             judge the score that will really be trained on, so it comes before
             `is_valid`.
          2. `is_valid` -- filter first, normalize after, a discarded piece is
             not normalized; R4's part removal also happens here and must come
             before the rules, otherwise R3 rejects the piece early.
          3. `normalize_chorale` -- R2, normalizes **in place**.
          4. `dedupe` -- R5 wraps outermost, because the predicate cannot see
             pieces already let through, and the comparison has to happen
             **after** normalization.

        In-place changes are safe: `chorales.Iterator()` does not cache, so
        changes do not accumulate across passes; the transforms are idempotent;
        `flatten()` shares the same elements with the original object, so the
        four consumers see them alike. `DEDUPE_ROWS` is currently False, i.e.
        R5 is a pass-through for now.
        """
        return chorale_filter.dedupe(
            chorale_filter.normalize_chorale(chorale)
            for chorale in map(chorale_filter.with_source,
                               self.corpus_it_gen())
            if self.is_valid(chorale))

    def make_tensor_dataset(self):
        """
        Build the TensorDataset from the corpus. Called only when there is no cache.
        """
        print('Making tensor dataset')
        self.compute_index_dicts()
        self.compute_voice_ranges()
        self.compute_window_layout()
        self.check_vocab_fits_storage()

        dtype = resolve_storage_dtype(self.storage_dtype)
        num_windows = sum(self.window_counts)
        window_length = int(round(self.sequences_size * self.subdivision))
        # Preallocate the whole output exactly: the shape comes from the window
        # layout above.
        score_tensor = torch.zeros(
            (num_windows, self.num_voices, window_length), dtype=dtype)
        metadata_tensor = torch.zeros(
            (num_windows, self.num_voices, window_length,
             len(self.metadatas) + 1), dtype=dtype)

        index = 0
        for chorale_id, chorale in tqdm(enumerate(self.iterator_gen())):
            index = self.fill_chorale(chorale, chorale_id,
                                      score_tensor, metadata_tensor, index)

        if index != num_windows:
            raise RuntimeError(
                f'window layout predicted {num_windows} windows but the fill '
                f'pass wrote {index}; a per-window KeyError (printed above) '
                f'leaves a gap and is the usual cause')
        print(f'Sizes: {score_tensor.size()}, {metadata_tensor.size()}')
        return TensorDataset(score_tensor, metadata_tensor)

    def check_vocab_fits_storage(self):
        """The cache stores indices, so every vocabulary has to fit the dtype."""
        limit = _DTYPE_MAX[self.storage_dtype]
        for part_id, note2index in enumerate(self.note2index_dicts):
            if len(note2index) > limit:
                raise ValueError(
                    f'voice {part_id} has {len(note2index)} entries, more than '
                    f'{self.storage_dtype} can hold (max {limit})')

    def cached_metadata_channels(self, chorale):
        """
        The metadata channels that are unchanged by transposition and therefore
        evaluated once per movement.

        :return: {metadata_index: np.ndarray}

        Only metadata that itself declares `is_transposition_invariant` is
        cached; the rest (such as KeyMetadata) is still recomputed per
        transposition.
        """
        cached = {}
        for index, metadata in enumerate(self.metadatas):
            if getattr(metadata, 'is_transposition_invariant', False):
                cached[index] = metadata.evaluate(chorale, self.subdivision)
        return cached

    def fill_chorale(self, chorale, chorale_id, score_tensor, metadata_tensor,
                     index):
        """Write one movement's windows into the preallocated buffers, return the
        new index."""
        flat = self.flat_parts(chorale)
        lowest, highest = self.movement_bounds(chorale)
        # Keyed by semitone: the transposed score and its metadata are reused by
        # every window position sharing that transposition.
        transposed_cache = {}
        cached_metadata = self.cached_metadata_channels(chorale)

        for offset_start in self.window_positions(lowest, highest):
            offset_end = offset_start + self.sequences_size
            ranges = self.voice_range_in_subsequence(flat, offset_start,
                                                     offset_end)
            min_t, max_t = self.min_max_transposition(ranges)
            start_tick = int(offset_start * self.subdivision)
            end_tick = int(offset_end * self.subdivision)

            for semi_tone in self.transposition_offsets(min_t, max_t):
                try:
                    if semi_tone not in transposed_cache:
                        transposed_cache[semi_tone] = \
                            self.transposed_score_and_metadata_tensors(
                                chorale, semi_tone=semi_tone,
                                cached_metadata=cached_metadata)
                    ct, mt = transposed_cache[semi_tone]
                    local_ct = self.extract_score_tensor_with_padding(
                        ct, start_tick, end_tick)
                    local_mt = self.extract_metadata_with_padding(
                        mt, start_tick, end_tick)
                except KeyError as e:
                    # Skipping without advancing index leaves the buffers short,
                    # which the caller's count check turns into a hard error
                    # rather than a silently truncated cache.
                    print(f'KeyError with chorale {chorale_id}: {e}')
                    continue
                score_tensor[index] = local_ct
                metadata_tensor[index] = local_mt
                index += 1
        return index

    def transposed_score_and_metadata_tensors(self, score, semi_tone,
                                              cached_metadata=None):
        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(semi_tone)
        transposition_interval = interval.Interval(str(interval_nature) + str(interval_type))
        chorale_tranposed = score.transpose(transposition_interval)
        chorale_tensor = self.get_score_tensor(
            # MODERNIZE: .flat → .flatten()
            chorale_tranposed, offsetStart=0., offsetEnd=chorale_tranposed.flatten().highestTime)
        metadatas_transposed = self.get_metadata_tensor(
            chorale_tranposed, cached=cached_metadata)
        return chorale_tensor, metadatas_transposed

    def get_metadata_tensor(self, score, cached=None):
        """
        Returns tensor (num_voices, chorale_length, len(self.metadatas) + 1)

        :param cached: {metadata_index: np.ndarray}, channels already evaluated
            on the untransposed score, for transposition-invariant metadata.
            Transposition does not change durations, so the cached array has the
            same length as the transposed score.
        """
        cached = cached or {}
        md = []
        if self.metadatas:
            for metadata_index, metadata in enumerate(self.metadatas):
                if metadata_index in cached:
                    sequence_metadata = torch.from_numpy(
                        cached[metadata_index]).long().clone()
                else:
                    sequence_metadata = torch.from_numpy(
                        metadata.evaluate(score, self.subdivision)).long().clone()
                square_metadata = sequence_metadata.repeat(self.num_voices, 1)
                md.append(square_metadata[:, :, None])
        chorale_length = int(score.duration.quarterLength * self.subdivision)
        voice_id_metadata = torch.from_numpy(np.arange(self.num_voices)).long().clone()
        square_metadata = torch.transpose(voice_id_metadata.repeat(chorale_length, 1), 0, 1)
        md.append(square_metadata[:, :, None])
        return torch.cat(md, 2)

    def set_fermatas(self, metadata_tensor, fermata_tensor):
        if self.metadatas:
            for metadata_index, metadata in enumerate(self.metadatas):
                if isinstance(metadata, FermataMetadata):
                    metadata_tensor[:, :, metadata_index] = fermata_tensor
                    break
        return metadata_tensor

    def add_fermata(self, metadata_tensor, time_index_start, time_index_stop):
        fermata_tensor = torch.zeros(self.sequences_size)
        fermata_tensor[time_index_start:time_index_stop] = 1
        return self.set_fermatas(metadata_tensor, fermata_tensor)

    def min_max_transposition(self, current_subseq_ranges):
        if current_subseq_ranges is None:
            return (0, 0)
        transpositions = [
            (min_pitch_corpus - min_pitch_current,
             max_pitch_corpus - max_pitch_current)
            for ((min_pitch_corpus, max_pitch_corpus),
                 (min_pitch_current, max_pitch_current))
            in zip(self.voice_ranges, current_subseq_ranges)
        ]
        transpositions = list(zip(*transpositions))
        return [max(transpositions[0]), min(transpositions[1])]

    def get_score_tensor(self, score, offsetStart, offsetEnd):
        chorale_tensor = []
        for part_id, part in enumerate(score.parts[:self.num_voices]):
            part_tensor = self.part_to_tensor(part, part_id,
                                              offsetStart=offsetStart,
                                              offsetEnd=offsetEnd)
            chorale_tensor.append(part_tensor)
        return torch.cat(chorale_tensor, 0)

    def part_to_tensor(self, part, part_id, offsetStart, offsetEnd):
        """
        :return: torch LongTensor (1, length)
        """
        # MODERNIZE: .flat → .flatten()
        list_notes_and_rests = list(part.flatten().getElementsByOffset(
            offsetStart=offsetStart,
            offsetEnd=offsetEnd,
            classList=[music21.note.Note, music21.note.Rest]))
        list_note_strings_and_pitches = [
            (n.nameWithOctave, n.pitch.midi) for n in list_notes_and_rests if n.isNote]
        length = int((offsetEnd - offsetStart) * self.subdivision)

        note2index = self.note2index_dicts[part_id]
        index2note = self.index2note_dicts[part_id]
        voice_range = self.voice_ranges[part_id]
        min_pitch, max_pitch = voice_range

        for note_name, pitch in list_note_strings_and_pitches:
            if pitch < min_pitch or pitch > max_pitch:
                note_name = OUT_OF_RANGE

            if note_name not in note2index:
                new_index = len(note2index)
                index2note[new_index] = note_name
                note2index[note_name] = new_index
                print(f'Warning: Entry {{{new_index}: {note_name!r}}} added to dictionaries')

        j = 0
        i = 0
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(list_notes_and_rests)
        while i < length:
            if j < num_notes - 1:
                if list_notes_and_rests[j + 1].offset > i / self.subdivision + offsetStart:
                    t[i, :] = [note2index[standard_name(list_notes_and_rests[j],
                                                        voice_range=voice_range)],
                               is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(list_notes_and_rests[j],
                                                    voice_range=voice_range)],
                           is_articulated]
                i += 1
                is_articulated = False

        seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        return torch.from_numpy(seq).long()[None, :]

    def voice_range_in_subsequence(self, flat_parts, offsetStart, offsetEnd):
        """Returns None as soon as a voice has no note in the window (which
        forbids transposition).

        :param flat_parts: the already flattened voices, in voice order.
        """
        voice_ranges = []
        for part in flat_parts:
            vr = self.voice_range_in_part(part, offsetStart=offsetStart, offsetEnd=offsetEnd)
            if vr is None:
                return None
            voice_ranges.append(vr)
        return voice_ranges

    def voice_range_in_part(self, flat_part, offsetStart, offsetEnd):
        """:param flat_part: a part that has already been flattened."""
        notes_in_subsequence = flat_part.getElementsByOffset(
            offsetStart, offsetEnd,
            includeEndBoundary=False,
            mustBeginInSpan=True,
            mustFinishInSpan=False,
            classList=[music21.note.Note, music21.note.Rest])
        midi_pitches_part = [n.pitch.midi for n in notes_in_subsequence if n.isNote]
        if midi_pitches_part:
            return min(midi_pitches_part), max(midi_pitches_part)
        return None

    def compute_index_dicts(self):
        """
        Build the note <-> index mapping in both directions for each voice.

        The collected note strings are **sorted()** before enumerating, which
        guarantees the same indices across runs, Python versions and platforms --
        an unsorted set gives a different result every time.
        """
        print('Computing index dicts')
        self.index2note_dicts = [{} for _ in range(self.num_voices)]
        self.note2index_dicts = [{} for _ in range(self.num_voices)]

        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.update([SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL])

        for chorale in tqdm(self.iterator_gen()):
            for part_id, part in enumerate(chorale.parts[:self.num_voices]):
                # MODERNIZE: .flat → .flatten()
                #
                # It has to use the **same** elements as `part_to_tensor`'s
                # `classList=[Note, Rest]`, otherwise the vocabulary takes in
                # things other than Note/Rest. `notesAndRests` also yields
                # ChordSymbol / NoChord (both GeneralNote), and `standard_name`
                # returns their `.figure` -- the MusicXML `<kind text="IV">`
                # roman numeral, not a pitch name; afterwards
                # `standard_note('IV')` raises `PitchException` in
                # `compute_voice_ranges` and kills the whole build; the
                # vocabulary also gains keys the fill pass never looks up.
                for n in part.flatten().notesAndRests:
                    if not isinstance(n, (music21.note.Note,
                                          music21.note.Rest)):
                        continue
                    note_sets[part_id].add(standard_name(n))

        for note_set, index2note, note2index in zip(
                note_sets, self.index2note_dicts, self.note2index_dicts):
            # sorted() makes the index assignment reproducible
            for note_index, note in enumerate(sorted(note_set)):
                index2note[note_index] = note
                note2index[note] = note_index

    def is_valid(self, chorale):
        """
        The four corpus filter rules, all implemented in
        `DatasetManager.chorale_filter`.

        R4 **modifies the score in place**, and it has to do so here:
        `iterator_gen` calls this predicate **before** `normalize_chorale`, and a
        part removal done any later misses R3. That is why this call is written
        out explicitly instead of being folded into `is_kept`: the rule set is
        pure, the reduction is not -- a predicate that mutates its own argument
        is a trap.

        `num_voices` must be passed the corpus constant `CORPUS_NUM_VOICES`, not
        `self.num_voices` (that is `len(voice_ids)`). R3 asks whether the *piece*
        has four voices, independently of which voices are taken this time.

        :return: True means this piece goes into the cache.
        """
        chorale_filter.reduce_parts(chorale)
        return chorale_filter.is_kept(
            chorale, num_voices=chorale_filter.CORPUS_NUM_VOICES)

    def compute_voice_ranges(self):
        assert self.index2note_dicts is not None
        assert self.note2index_dicts is not None
        self.voice_ranges = []
        print('Computing voice ranges')
        for voice_index, note2index in tqdm(enumerate(self.note2index_dicts)):
            notes = [standard_note(ns) for ns in note2index]
            midi_pitches = [n.pitch.midi for n in notes if n.isNote]
            self.voice_ranges.append((min(midi_pitches), max(midi_pitches)))

    def extract_score_tensor_with_padding(self, tensor_score, start_tick, end_tick):
        """
        Returns tensor_score[:, start_tick:end_tick], padded with START / END at
        either end as needed.
        """
        assert start_tick < end_tick
        assert end_tick > 0
        length = tensor_score.size()[1]
        padded = []

        if start_tick < 0:
            s = np.array([n2i[START_SYMBOL] for n2i in self.note2index_dicts])
            s = torch.from_numpy(s).long().repeat(-start_tick, 1).transpose(0, 1)
            padded.append(s)

        slice_start = max(start_tick, 0)
        slice_end = min(end_tick, length)
        padded.append(tensor_score[:, slice_start:slice_end])

        if end_tick > length:
            e = np.array([n2i[END_SYMBOL] for n2i in self.note2index_dicts])
            e = torch.from_numpy(e).long().repeat(end_tick - length, 1).transpose(0, 1)
            padded.append(e)

        return torch.cat(padded, 1)

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        """
        :param tensor_metadata: (num_voices, length, num_metadatas)
        """
        assert start_tick < end_tick
        assert end_tick > 0
        num_voices, length, num_metadatas = tensor_metadata.size()
        padded = []

        if start_tick < 0:
            s = np.zeros((self.num_voices, -start_tick, num_metadatas))
            padded.append(torch.from_numpy(s).long())

        slice_start = max(start_tick, 0)
        slice_end = min(end_tick, length)
        padded.append(tensor_metadata[:, slice_start:slice_end, :])

        if end_tick > length:
            e = np.zeros((self.num_voices, end_tick - length, num_metadatas))
            padded.append(torch.from_numpy(e).long())

        return torch.cat(padded, 1)

    def empty_score_tensor(self, score_length):
        s = np.array([n2i[START_SYMBOL] for n2i in self.note2index_dicts])
        return torch.from_numpy(s).long().repeat(score_length, 1).transpose(0, 1)

    def random_score_tensor(self, score_length):
        t = np.array([np.random.randint(len(n2i), size=score_length)
                      for n2i in self.note2index_dicts])
        return torch.from_numpy(t).long()

    def tensor_to_score(self, tensor_score, fermata_tensor=None):
        """
        :param tensor_score: (num_voices, length)
        :return: music21 Score
        """
        slur_indexes = [n2i[SLUR_SYMBOL] for n2i in self.note2index_dicts]
        score = music21.stream.Score()
        num_voices = tensor_score.size(0)
        name_parts = (num_voices == 4)
        part_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

        for voice_index, (voice, index2note, slur_index) in enumerate(
                zip(tensor_score, self.index2note_dicts, slur_indexes)):
            add_fermata = False
            if name_parts:
                part = stream.Part(
                    id=part_names[voice_index],
                    partName=part_names[voice_index],
                    partAbbreviation=part_names[voice_index],
                    instrumentName=part_names[voice_index])
            else:
                part = stream.Part(id='part' + str(voice_index))

            dur = 0
            total_duration = 0
            f = music21.note.Rest()
            for note_index in [n.item() for n in voice]:
                if note_index != slur_indexes[voice_index]:
                    if dur > 0:
                        f.duration = music21.duration.Duration(dur / self.subdivision)
                        if add_fermata:
                            f.expressions.append(music21.expressions.Fermata())
                            add_fermata = False
                        part.append(f)
                    dur = 1
                    f = standard_note(index2note[note_index])
                    if fermata_tensor is not None and voice_index == 0:
                        add_fermata = (fermata_tensor[0, total_duration] == 1)
                    total_duration += 1
                else:
                    dur += 1
                    total_duration += 1

            f.duration = music21.duration.Duration(dur / self.subdivision)
            if add_fermata:
                f.expressions.append(music21.expressions.Fermata())
            part.append(f)
            score.insert(part)

        return score


class ChoraleBeatsDataset(ChoraleDataset):
    """
    Beat-level variant of ChoraleDataset: one window per beat instead of one per tick.

    The parent builder is parameterized by `window_step()` alone and knows
    nothing about tick size, so the whole difference between the two datasets is
    that single override -- no separate fill loop and no separate repr;
    preallocation, the window-count check, the transposition and metadata caches,
    and the cache filename are all shared.
    """

    def window_step(self):
        """One beat, not one tick."""
        return 1.0

    def make_tensor_dataset(self):
        """The implementation is entirely in the parent; this override is kept so
        the class owns its build contract."""
        return super().make_tensor_dataset()
