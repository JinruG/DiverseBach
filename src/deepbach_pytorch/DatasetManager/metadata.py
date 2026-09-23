"""
Definition of the metadata channels
"""
import numpy as np
from music21 import analysis, stream, meter, expressions
from DatasetManager.helpers import SLUR_SYMBOL, \
    PAD_SYMBOL


class Metadata:
    def __init__(self):
        self.num_values = None
        self.is_global = None
        self.name = None
        #: True means `evaluate()` returns the same array for a piece and for any
        #: of its transpositions. The cache builder computes such channels once
        #: per movement instead of once per transposition.
        self.is_transposition_invariant = False

    def get_index(self, value):
        # the value 0 is handled specially
        raise NotImplementedError

    def get_value(self, index):
        raise NotImplementedError

    def evaluate(self, chorale, subdivision):
        """
        Input: a music21 chorale + the number of subdivisions per beat
        """
        raise NotImplementedError

    def generate(self, length):
        raise NotImplementedError


class IsPlayingMetadata(Metadata):
    def __init__(self, voice_index, min_num_ticks):
        """
        Indicates whether a voice is playing. A voice is considered silenced when
        it rests for more than `min_num_ticks` consecutive ticks.

        :param voice_index: index of the voice to look at
        :param min_num_ticks: minimum rest length (in ticks) to enter the metadata
        """
        super(IsPlayingMetadata, self).__init__()
        self.min_num_ticks = min_num_ticks
        self.voice_index = voice_index
        self.is_global = False
        self.num_values = 2
        self.name = 'isplaying'
        # rests and their lengths are unaffected by transposition.
        self.is_transposition_invariant = True

    def get_index(self, value):
        return int(value)

    def get_value(self, index):
        return bool(index)

    def evaluate(self, chorale, subdivision):
        """
        Input: a music21 chorale
        """
        length = int(chorale.duration.quarterLength * subdivision)
        metadatas = np.ones(shape=(length,))
        part = chorale.parts[self.voice_index]

        for note_or_rest in part.notesAndRests:
            is_playing = True
            if note_or_rest.isRest:
                if note_or_rest.quarterLength * subdivision >= self.min_num_ticks:
                    is_playing = False
            # these are expected to be integer values
            start_tick = note_or_rest.offset * subdivision
            end_tick = start_tick + note_or_rest.quarterLength * subdivision
            metadatas[start_tick:end_tick] = self.get_index(is_playing)
        return metadatas

    def generate(self, length):
        return np.ones(shape=(length,))


class TickMetadata(Metadata):
    """
    Metadata recording which subdivision within the beat the current tick falls on
    """

    def __init__(self, subdivision):
        super(TickMetadata, self).__init__()
        self.is_global = False
        self.num_values = subdivision
        self.name = 'tick'
        # depends only on the tick index.
        self.is_transposition_invariant = True

    def get_index(self, value):
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale, subdivision):
        assert subdivision == self.num_values
        # assumes every piece starts on a whole beat
        length = int(chorale.duration.quarterLength * subdivision)
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))

    def generate(self, length):
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))


class ModeMetadata(Metadata):
    """
    Indicates the current mode of the melody: major, minor or other
    """

    def __init__(self):
        super(ModeMetadata, self).__init__()
        self.is_global = False
        self.num_values = 3  # major, minor or other
        self.name = 'mode'

    def get_index(self, value):
        if value == 'major':
            return 1
        if value == 'minor':
            return 2
        return 0

    def get_value(self, index):
        if index == 1:
            return 'major'
        if index == 2:
            return 'minor'
        return 'other'

    def evaluate(self, chorale, subdivision):
        # todo measures have to be filled in when parsing midi
        # initialize the key analyser
        ka = analysis.floatingKey.KeyAnalyzer(chorale)
        res = ka.run()

        measure_offset_map = chorale.parts[0].measureOffsetMap()
        length = int(chorale.duration.quarterLength * subdivision)  # the unit is a 16th note

        modes = np.zeros((length,))

        measure_index = -1
        for time_index in range(length):
            beat_index = time_index / subdivision
            if beat_index in measure_offset_map:
                measure_index += 1
                modes[time_index] = self.get_index(res[measure_index].mode)

        return np.array(modes, dtype=np.int32)

    def generate(self, length):
        return np.full((length,), self.get_index('major'))


class KeyMetadata(Metadata):
    """
    Indicates which key we are in: only the number of sharps/flats is returned,
    without distinguishing a key from its relative key
    """

    def __init__(self, window_size=4):
        super(KeyMetadata, self).__init__()
        self.window_size = window_size
        self.is_global = False
        self.num_max_sharps = 7
        self.num_values = 16
        self.name = 'key'

    def get_index(self, value):
        """

        :param value: number of sharps, ranging -7..+7
        :return: the index of the representation
        """
        return value + self.num_max_sharps + 1

    def get_value(self, index):
        """

        :param index: index, ranging 0..self.num_values; 0 is unused (no constraint)
        :return: the actual number of sharps, ranging -7..7
        """
        return index - 1 - self.num_max_sharps

    def evaluate(self, chorale, subdivision):
        # initialize the key analyser
        # on the midi parsing path the measures have to be filled in by hand
        chorale_with_measures = stream.Score()
        for part in chorale.parts:
            chorale_with_measures.append(part.makeMeasures())

        ka = analysis.floatingKey.KeyAnalyzer(chorale_with_measures)
        ka.windowSize = self.window_size
        res = ka.run()

        measure_offset_map = chorale_with_measures.parts.measureOffsetMap()
        length = int(chorale.duration.quarterLength * subdivision)  # the unit is a 16th note

        key_signatures = np.zeros((length,))

        measure_index = -1
        for time_index in range(length):
            beat_index = time_index / subdivision
            if beat_index in measure_offset_map:
                measure_index += 1
                if measure_index == len(res):
                    measure_index -= 1

            key_signatures[time_index] = self.get_index(res[measure_index].sharps)
        return np.array(key_signatures, dtype=np.int32)

    def generate(self, length):
        return np.full((length,), self.get_index(0))


def _has_fermata(note_or_rest):
    """Whether this element carries a Fermata mark.

    Checking `len(note.expressions) == 1` alone is not enough: any single
    expression (an imported dynamic, an ornament) would be misread as a fermata,
    while a real fermata is missed as soon as it coexists with another
    expression.

    :return: bool
    """
    return any(isinstance(e, expressions.Fermata) for e in note_or_rest.expressions)


class FermataMetadata(Metadata):
    """
    Indicates whether the current note has a fermata
    """

    def __init__(self):
        super(FermataMetadata, self).__init__()
        self.is_global = False
        self.num_values = 2
        self.name = 'fermata'
        # a fermata is an expression; transposition changes pitch, not expressions.
        self.is_transposition_invariant = True

    def get_index(self, value):
        # the values are only 1 and 0, so value is the index
        return value

    def get_value(self, index):
        # the values are only 1 and 0, so value is the index
        return index

    def evaluate(self, chorale, subdivision):
        """
        Whether each tick has a fermata; the array is sized to the **whole score**.

        The length comes from `chorale.duration`, not `parts[0].duration`:
        `get_metadata_tensor` `torch.cat`s the channels together, so a fermata
        channel one stretch short makes the build fail on a size mismatch. The
        array is score-sized, but the forward-fill loop stops where the part
        itself ends and the tail stays 0.
        """
        part = chorale.parts[0]
        length = int(chorale.duration.quarterLength * subdivision)  # the unit is a 16th note
        part_length = int(part.duration.quarterLength * subdivision)
        # MODERNIZE: .flat → .flatten()
        list_notes = part.flatten().notes
        num_notes = len(list_notes)
        fermatas = np.zeros((length,))
        if num_notes == 0:
            # a part with no notes (an empty placeholder staff, or the melody not in
            # parts[0]) carries no fermata information.
            return np.array(fermatas, dtype=np.int32)
        j = 0
        i = 0
        while i < part_length:
            if j < num_notes - 1:
                if list_notes[j + 1].offset > i / subdivision:
                    fermatas[i] = _has_fermata(list_notes[j])
                    i += 1
                else:
                    j += 1
            else:
                fermatas[i] = _has_fermata(list_notes[j])
                i += 1
        return np.array(fermatas, dtype=np.int32)

    def generate(self, length):
        # one fermata every two measures
        return np.array([1 if i % 32 >= 28 else 0
                         for i in range(length)])
