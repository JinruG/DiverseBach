"""
Corpus filtering: five rules, all hung off the single entry point of corpus
iteration (`chorale_dataset.iterator_gen`).

    R1  no instrumental voice    -- drop the whole piece
    R2  one time point, top pitch only -- delete the rest
    R3  must be exactly 4 voices -- drop the whole piece
    R4  hardcoded part removal   -- 8 pieces, see `PART_REMOVALS`
    R5  duplicate pieces dedupe  -- keep one row per spelling

`PART_SOURCES` is not a rule: nothing is deleted or changed, it only decides
which corpus file is read.

It is a module of its own because `chorale_dataset.py` is co-written by several
sessions and has no version control: that side references only two lines, so
deleting this file is a complete rollback.
"""

import hashlib

from music21 import chord, corpus as m21corpus, note as m21note

__all__ = [
    'CORPUS_NUM_VOICES',
    'VOCAL_PART_PREFIXES',
    'PART_REMOVALS',
    'PART_SOURCES',
    'DEDUPE_ROWS',
    'part_name_is_vocal',
    'instrumental_part_names',
    'piece_number',
    'with_source',
    'reduce_parts',
    'rejection_reasons',
    'is_kept',
    'normalize_simultaneities',
    'normalize_chorale',
    'content_key',
    'dedupe',
]

#: How many voices a chorale has (R3). A corpus constant, **not** a dataset
#: instance attribute: `self.num_voices` is `len(voice_ids)`, and using it would
#: turn into "this piece has exactly 1 voice". What is asked is whether the
#: *piece* has four voices, independently of which voices are taken this time.
CORPUS_NUM_VOICES = 4

#: Vocal part names, matched by **prefix**, so 'Soprano Oboe 1 Violin1' counts as
#: vocal. 'Continuo' / 'Organ' / 'Timpani' / 'Violin' match no prefix and count
#: as instrumental.
VOCAL_PART_PREFIXES = ('soprano', 'alto', 'tenor', 'bass',
                       'cantus', 'contratenor', 'quintus')

#: Single-letter abbreviations ('s'/'a'/'t'/'b') are matched **exactly** only,
#: otherwise 'b' would swallow 'Bassoon'.
VOCAL_SINGLE_LETTERS = ('s', 'a', 't', 'b')

#: The R5 switch. False takes the whole corpus (duplicate rows included), True dedupes.
DEDUPE_ROWS = False

#: R4: pieces that are four-voice only after these parts are removed. The key is
#: the **Riemenschneider** number (in this corpus `metadata.number` is the Riem
#: number, not the BWV number), the value is the exact `partName` -- exact
#: rather than a prefix, so 'Horn 1' does not match 'Horn 1,2'.
#:
#: Exactly `CORPUS_NUM_VOICES` parts must remain after the removal; `reduce_parts`
#: asserts it, so a typo in the table is a crash rather than a piece quietly
#: going missing. The top voice Riem 51 leaves is instrumental (the horn of
#: BWV 91.6 plays in unison with the cantus), so that piece is exempt from R1.
PART_REMOVALS = {
    43:  ('Continuo',),
    51:  ('Horn 2', 'Timpani', 'Soprano'),
    150: ('Soprano 2',),
    313: ('Cornet 1', 'Cornet 2', 'Soprano'),
    323: ('Violin',),
    329: ('Cornet 1', 'Cornet 2'),
    347: ('Horn 1', 'Horn 2'),
    362: ('Continuo',),
}

#: Pieces that are always read from a fixed **file**: the key is the Riem number,
#: the value is an explicit path under the music21 corpus. An explicit path
#: rather than a bare name is used to get around music21's **prefix matching** --
#: a bare name walks back into it. Not a rule: nothing is deleted or changed.
#: Riem 327 is a pure substitution (the same music, with eleven parts to remove);
#: Riem 353 is the only file carrying that piece's four-voice spelling, and
#: Riem 313 (a different chorale) used to read it away.
PART_SOURCES = {
    327: 'bach/bwv190.7.mxl',
    353: 'bach/bwv112.5.mxl',
}


def part_name_is_vocal(name):
    """Is this part name vocal? An empty name counts as non-instrumental.

    All four parts of Riem 6/15/161 have no partName, yet they are ordinary
    four-voice chorales.
    """
    n = (name or '').strip().lower()
    if not n:
        return True
    if n in VOCAL_SINGLE_LETTERS:
        return True
    return any(n.startswith(p) for p in VOCAL_PART_PREFIXES)


def instrumental_part_names(chorale):
    """Names of the instrumental parts of this piece (empty names count as vocal)."""
    return [p.partName or '' for p in chorale.parts
            if not part_name_is_vocal(p.partName)]


def piece_number(chorale):
    """The Riemenschneider number (int); None when it cannot be read.

    `metadata.number` is a string, and it is the Riem number, not the BWV
    number; a few are compound values (such as `'248.9-1'`), so the part before
    the decimal point is taken.
    """
    num = chorale.metadata.number
    try:
        return int(str(num).split('.')[0])
    except (TypeError, ValueError):
        return None


def with_source(chorale):
    """Re-parse this piece according to `PART_SOURCES`; pieces that are not
    pinned are returned as they are.

    Most rows (369/371) take the return-as-is path; this is the only step in the
    filter that returns a **different object**, so it must come before
    `iterator_gen`'s `is_valid`.

    `metadata.number` and `title` have to be carried over by hand: a score parsed
    by explicit path has both set to None, and `number` is the key of every table
    in this module (this one included).
    """
    path = PART_SOURCES.get(piece_number(chorale))
    if path is None:
        return chorale

    fresh = m21corpus.parse(path)
    fresh.metadata.number = chorale.metadata.number
    fresh.metadata.title = chorale.metadata.title
    return fresh


def reduce_parts(chorale, num_voices=CORPUS_NUM_VOICES):
    """R4: **in place**, delete this piece's hardcoded parts and return the names
    actually removed (an empty tuple for pieces not in the table).

    It must run **before** the rules: `iterator_gen` does `is_valid` and only then
    normalizes, and a piece whose parts are deleted any later is dropped early by
    R3 as "seven voices". Modifying in place is safe for the same reason as R2:
    `chorales.Iterator()` does not cache, so changes do not accumulate across
    several passes.
    Idempotent: a second call only re-checks the postcondition -- exactly
    `num_voices` parts; a name mismatch (a revised corpus, a typo in the table)
    leaves the piece long and raises here.
    """
    want = PART_REMOVALS.get(piece_number(chorale))
    if not want:
        return ()

    removed = []
    for name in want:
        hits = [p for p in chorale.parts if (p.partName or '') == name]
        if not hits:
            continue                      # already gone: second call, or corpus drift
        for p in hits:
            chorale.remove(p)
        removed.append(name)

    if len(chorale.parts) != num_voices:
        raise RuntimeError(
            f'Riem {piece_number(chorale)}: PART_REMOVALS {want} left '
            f'{len(chorale.parts)} parts, expected {num_voices} -- '
            f'remaining {[p.partName for p in chorale.parts]}. '
            f'A name in the table no longer matches the corpus.')
    return tuple(removed)


def rejection_reasons(chorale, num_voices=CORPUS_NUM_VOICES):
    """**All** the rules that are violated; an empty list means it is kept.

    All are listed, not just the first (Riem 43 violates two at once).
    It must be called **after** `reduce_parts`, otherwise R3 counts the parts R4
    is supposed to remove. Pieces in `PART_REMOVALS` are **exempt from R1**: the
    table already says which parts to remove, so the instrumental parts that
    remain are a decision, not an oversight. Pieces outside the table are checked
    as usual.
    """
    reasons = []
    n = len(chorale.parts)
    if n != num_voices:
        reasons.append(f'声部数 {n} != {num_voices}')
    if piece_number(chorale) not in PART_REMOVALS:
        inst = instrumental_part_names(chorale)
        if inst:
            reasons.append(f'含乐器声部 {inst}')
    return reasons


def is_kept(chorale, num_voices=CORPUS_NUM_VOICES):
    return not rejection_reasons(chorale, num_voices)


def _remove(part, el, where):
    """Remove `el` from `part`; if it is still there, raise with context.

    `el.activeSite.remove()` is not used: some elements have `activeSite is None`
    (Riem 209's two duplicate notes hang directly off the Part).
    `part.remove(el, recurse=True)` searches the whole subtree by identity,
    regardless of what the element itself records.
    """
    part.remove(el, recurse=True)
    if any(x is el for x in part.recurse().notes):
        raise RuntimeError(
            f'{where}: part.remove did not delete this note '
            f'(activeSite={type(el.activeSite).__name__ if el.activeSite else None}, '
            f'offset={el.offset}, el={el!r})')


def normalize_simultaneities(chorale, num_voices=CORPUS_NUM_VOICES):
    """R2: keep only the **highest** note at each time point of each voice,
    returning how many notes were deleted or replaced.

    Only the first `num_voices` parts are handled, the same range as
    `get_score_tensor`'s `score.parts[:self.num_voices]`; the rest could not
    reach the tensor anyway.

    Two cases:
      * several `Note`s at the same offset -> keep the highest, delete the rest
      * a `Chord` at the same offset -> replace it with its top note
        (`part_to_tensor`'s `classList=[Note, Rest]` skips a Chord wholesale)

    Offsets are compared as absolute values after `flatten()`, as in
    `part_to_tensor`.
    """
    removed = 0
    for part in list(chorale.parts)[:num_voices]:
        by_onset = {}
        for el in part.flatten().notes:
            by_onset.setdefault(round(float(el.offset), 6), []).append(el)

        for onset, els in sorted(by_onset.items()):
            if len(els) == 1 and isinstance(els[0], m21note.Note):
                continue                      # clean, leave it alone

            def top_midi(e):
                return max(p.midi for p in (e.pitches
                                            if isinstance(e, chord.Chord)
                                            else [e.pitch]))

            els_sorted = sorted(els, key=top_midi)
            winner = els_sorted[-1]
            losers = els_sorted[:-1]

            where = (f'{chorale.metadata.number} / {part.partName!r} / '
                     f'onset {onset}')

            # the winning chord must also become a single note, otherwise
            # `part_to_tensor` still skips it wholesale.
            if isinstance(winner, chord.Chord):
                keep = m21note.Note(
                    max(winner.pitches, key=lambda p: p.midi))
                keep.duration = winner.duration
                _remove(part, winner, where + ' (chord)')
                part.insert(onset, keep)
                removed += 1

            for loser in losers:
                _remove(part, loser, where)
                removed += 1
    return removed


def normalize_chorale(chorale, num_voices=CORPUS_NUM_VOICES):
    """Judge the rules first, then normalize; a discarded piece is not normalized."""
    normalize_simultaneities(chorale, num_voices=num_voices)
    return chorale


def content_key(chorale):
    """The **music**'s fingerprint: pitch, onset and duration of each voice.

    R5 uses it to dedupe. The filename is deliberately not used, and neither is
    the `bwv` field of the table.

    It is taken from the score **after** `normalize_chorale`: the same-onset
    conflicts are gone by then, so the fingerprint describes exactly what
    `part_to_tensor` will really see. Offsets are rounded with `%.6f`, because
    the float values arrive by different paths (the pinned `corpus.parse` and the
    iterator).
    """
    digest = hashlib.md5()
    for part in chorale.parts:
        for el in part.flatten().notes:
            if el.isNote:
                pitches = str(el.pitch.midi)
            else:                       # a chord (if any survives R2)
                pitches = ','.join(str(m) for m in
                                   sorted(p.midi for p in el.pitches))
            digest.update(('%s|%.6f|%g;' % (pitches, float(el.offset),
                                            float(el.duration.quarterLength))
                           ).encode())
        digest.update(b'#')             # voice boundary: '12'+'3' must not equal '1'+'23'
    return digest.hexdigest()


def dedupe(chorales, key=content_key):
    """R5 entry point: keep one row per distinct spelling, or return `chorales`
    unchanged.

    With `DEDUPE_ROWS` False it returns `chorales` directly. A plain `return` is
    used rather than a branch inside the generator, because a `return` inside a
    generator ends the iteration instead of returning a value, and the caller
    would silently get an empty corpus.
    """
    if not DEDUPE_ROWS:
        return chorales
    return _dedupe(chorales, key)


def _dedupe(chorales, key):
    """Yield only the first row of each identical spelling (R5).

    The representative is the row that arrives first, i.e. the smallest Riem
    number in the group (`chorales.Iterator` walks the table in Riem order).
    `seen` is **local**, so every pass over the corpus starts from scratch.
    """
    seen = set()
    for chorale in chorales:
        k = key(chorale)
        if k in seen:
            continue
        seen.add(k)
        yield chorale


# ---------------------------------------------------------------------------
#  self-test: run the rules over the whole corpus and print the tally
# ---------------------------------------------------------------------------

def _self_test():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    from music21.corpus import chorales

    scores = list(chorales.Iterator())
    print(f"\n语料总数: {len(scores)}")

    kept, dropped = [], []
    for s in scores:
        # with_source before R4, the same order as `iterator_gen`:
        # the rules must see the pinned file, and must see the reduced piece.
        src = PART_SOURCES.get(piece_number(s))
        s = with_source(s)
        if src:
            print(f"  源  Riem {piece_number(s):>3}: 换用 {src} -> "
                  f"{[p.partName for p in s.parts]}")
        removed = reduce_parts(s)
        reasons = rejection_reasons(s)
        if removed:
            print(f"  R4  Riem {piece_number(s):>3}: 去掉 {list(removed)} -> "
                  f"{[p.partName for p in s.parts]}")
        (dropped if reasons else kept).append((s, reasons))

    print(f"保留: {len(kept)}    排除: {len(dropped)}")
    print(f"\n排除清单（Riem 号）:")
    nums = []
    for s, reasons in dropped:
        num = s.metadata.number
        try:
            num = int(str(num).split('.')[0])
        except (TypeError, ValueError):
            pass
        nums.append(num)
        print(f"  Riem {num:>3}  {'; '.join(reasons)}")
    print(f"\n排除号的集合: {sorted(nums)}")

    # R2
    total = 0
    touched = []
    for s, _ in kept:
        n = normalize_simultaneities(s)
        if n:
            total += n
            num = s.metadata.number
            touched.append(num)
    print(f"\nR2 归一化: 命中 {len(touched)} 首 {touched}，"
          f"共删/换 {total} 个音")

    # idempotence: a second pass must change nothing
    again = sum(normalize_simultaneities(s) for s, _ in kept)
    print(f"R2 幂等性: 第二遍删了 {again} 个（必须是 0）")
    assert again == 0, 'normalize_simultaneities is not idempotent!'

    # no same-onset multi-note may survive normalization
    leftover = 0
    for s, _ in kept:
        for part in s.parts[:4]:
            by_onset = {}
            for el in part.flatten().notes:
                by_onset.setdefault(round(float(el.offset), 6), []).append(el)
            leftover += sum(1 for els in by_onset.values()
                            if len(els) > 1 or isinstance(
                                els[0], chord.Chord))
    print(f"R2 结果: 归一化后仍有同 onset 多音的 time-point 数 = {leftover}"
          f"（必须是 0）")
    assert leftover == 0, 'simultaneities survived normalisation'

    # R5 runs on the same scores R2 has just normalized, the same order as
    # `iterator_gen`.
    groups = {}
    for s, _ in kept:
        groups.setdefault(content_key(s), []).append(piece_number(s))
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    redundant = sum(len(v) - 1 for v in dupes.values())
    by_riem = sorted(dupes.values(),
                     key=lambda ns: min(n for n in ns if n is not None))
    # the two tallies must come from the same source: with R5 on the corpus yields
    # `unique` pieces, with it off `len(kept)`. If a rule alters the corpus this
    # fails here instead of being dragged out to build time.
    assert len(groups) == len(kept) - redundant, 'dedupe accounting is off'

    print(f"\nR5: {len(dupes)} 组同曲异号，冗余 {redundant} 行；"
          f"DEDUPE_ROWS = {DEDUPE_ROWS}")
    for ns in by_riem:
        nums = sorted(n for n in ns if n is not None)
        other = nums[1:] + [n for n in ns if n is None]
        print(f"  留 Riem {nums[0]:>3}，{'去' if DEDUPE_ROWS else '也留'} {other}")
    corpus = len(groups) if DEDUPE_ROWS else len(kept)
    print(f"  **进缓存的语料: {corpus} 首**（保留 {len(kept)} 行，"
          f"{'去重后' if DEDUPE_ROWS else '重复照收'}）")

    print("\nself-test 通过。")
    return 0


if __name__ == '__main__':
    raise SystemExit(_self_test())
