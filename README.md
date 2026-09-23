
Training is Bach-only. The OpenScore String Quartets corpus has been removed,
along with the `corpus` / `corpus_dir` / `limit` / `extra_argv` arguments of
`train_from_scratch()`. No exported symbol was removed — the quartet corpus was
never one, only a value of the `corpus` argument — and `arch`
(`'baseline'` / `'shared_trunk'`) still selects between four independent
VoiceModels and scheme A's shared trunk, both on the same Bach tensor cache.

`harmonize()` gained two keyword arguments, both with defaults, so existing
calls behave as before.

**`arch` on every entry point.** `arch` used to be a `train_from_scratch()`
argument only, which left scheme A unreachable through the public API: driving
`harmonize()` or `generate_from_scratch()` with a shared trunk meant replacing
`_get_default_model` and `_load_pretrained_weights` with local stand-ins, and
repeating the seven hyperparameters at every call site. `harmonize()`,
`generate_from_scratch()`, `build_model()`, `create_model()`, `load_model()` and
`check_pretrained_weights()` now take `arch` (`'baseline'` / `'shared_trunk'`),
plus `role_conditioned` and `lstm_hidden_size` where they apply. An unknown
`arch` raises rather than falling back to the baseline, and each architecture's
default `lstm_hidden_size` comes from one table (`BASELINE_HYPERS` / the shared
trunk's `HYPERS`) instead of from a literal at each call site.

One consequence worth knowing: `data/models_<snapshot>/<arch>/` keeps the two
architectures apart — the loader recognises a checkpoint by
`endswith(repr(model))` and both reprs end in `)`, so one directory holding both
makes the model choice an `os.listdir` accident. `lstm_hidden_size` is no longer
one of them: both tables now read 256.

**Training follows upstream's code, not its paper.** The two disagree on four
points, and this tree takes the code every time (upstream is
`DeepBach-master/`; the paper is `_2018_paper.txt`):

| | paper | code, adopted here |
|---|---|---|
| transposition | whole chorale, must fit the range | judged per window (`chorale_dataset.py`) |
| split | train 80 / val 20 | 85 / 10 / 5 (`music_dataset.py`) |
| capacity | MLP 200 + 2xLSTM 200 | `--lstm_hidden_size 256 --linear_hidden_size 256` |
| input dropout | 20% in + 50% per layer | no `nn.Dropout` anywhere; only the LSTM's own inter-layer 50% |

So the data layer is untouched and the corpus cache is reproducible byte for
byte. One deliberate deviation remains: a single trunk serving four roles has a
quarter of the baseline's per-voice recurrent capacity at 256. Strict parity
with upstream is therefore the single-voice route (`VoiceModel` with
`num_voices == 1`, driven by `_retrain.py --stage voice0`), not the shared
trunk.

**Training budget is counted in optimizer steps**, not epochs:
`steps_per_epoch` defaults to `len(train_loader)` and `max_steps` to
`DEFAULT_TOTAL_PASSES` passes over it. `max_steps` is rounded *up* to a whole
number of passes so that every save point lands on a pass boundary; the effective
numbers are printed at the top of training. `num_epochs` is gone rather than aliased — a stale call site
raises `TypeError` instead of quietly training a different amount.

The two numbers do different jobs, and shrinking only one does not make a run
smaller: `max_steps` decides *how many* passes run, `steps_per_epoch` decides
*how long* each one is. The training pass is capped at `steps_per_epoch` batches
(`loss_and_acc(..., steps=steps_per_epoch)`); the validation pass is **not**
capped, because val loss is both the best-checkpoint criterion and the
`ReduceLROnPlateau` input, and a truncated one would be a different quantity
being compared across passes. A budget of fewer than `DEFAULT_TOTAL_PASSES`
passes leaves the later snapshot directories empty — that is the schedule
working, not a failure.

**Twenty-one checkpoint sets, twenty-one sibling directories.** Every training
run writes `data/models_best/<arch>/` (the pass with the best validation loss)
plus `data/models_epoch01/<arch>/` … `data/models_epoch20/<arch>`, one per pass.
Inference loads `epoch20`, which is `DEFAULT_SNAPSHOT` in
`DatasetManager.helpers` — one constant to flip. The full ladder is kept rather
than a few points because validation loss bottoms out well before a run ends
(measured at passes 5-8 on the old 15-pass budget), so with only a handful of
saved points the true minimum can fall between two of them and be invisible.
Note that the default is the *end* of the run, not `best` — the opposite
tradeoff from the old `epoch15` default, which existed because `best` measured
0.7-1.0 point better. All 21 are on disk either way, so switching is a constant,
not a retrain.

**Measurement and self-check modules.** `db.analysis` holds the pure measurement
code (teacher-forced accuracy with a constant-prediction baseline, the binned
metrics, score density/range statistics, melody preservation, file and
weight-directory digests, checkpoint export). It takes tensors and paths, returns
dicts, and prints nothing. `db.selfcheck` asserts this machine's invariants:
that the import resolves to the source tree rather than to an installed copy,
that the dataset cache was loaded rather than silently rebuilt, that the weight
directory holds exactly the expected files, and that the shared trunk really is
one trunk with four heads — including that `RoleView.forward` and
`forward_all(...)[r]` agree elementwise on real windows. Both are reachable as
attributes on the package (`db.analysis.flat_metrics(...)`) and neither does I/O
at import time. `db.selfcheck` runs the lightweight battery; the teacher-forced
accuracy report is available on demand and is deliberately not part of the run.

**Training always starts from scratch.** There is no resume and no
`skip_voices`; both parameters were removed rather than deprecated, so a stale
call site raises `TypeError` instead of quietly training a different amount. The
reason is in the weights: a run that was interrupted and resumed carries a
measurable fingerprint in its parameters (~1e-3, from the nondeterministic cuDNN
LSTM backward pass amplified by Adam) that cannot be told apart from the effect a
training variable is supposed to have. A comparison between arms is only
meaningful if none of them has one, so an interrupted arm is rerun from the
beginning. The cost is that an interruption costs the whole run rather than the
pass in progress.

What an interruption cannot cost is a checkpoint. Checkpoints are written
atomically (temp file, `fsync`, `os.replace`), so a kill during a save leaves the
previous one intact rather than a truncated file that exists and still cannot be
loaded. And `_stash()` renames a colliding output to `*_old` — it never deletes —
so rerunning an arm cannot destroy the previous one's weights. Snapshot sets are
siblings precisely so that no two of them can land in one directory and make the
checkpoint choice an `os.listdir` accident.

Every pass's metrics are written to `<models_dir>/_history/<model repr>`: train
loss, validation loss, validation accuracy and the learning rate, one row per
pass. They live in a `_history/` subdirectory rather than at the top level
because `selfcheck.check_weights_dir` asserts an exact top-level file count. That
payload is the artefact of record for the curves; `_retrain.py` additionally
copies it into `_retrain/<tree>_<stage>.json`, so rerunning a stage cannot lose
the previous curve.

**Input robustness.** `harmonize()` now accepts a melody-only file. It picks
the part that actually carries notes rather than trusting the part count, so an
exported lead sheet that ships one real staff plus three empty ones no longer
fixes silence and regenerates the wrong voice. A single-part input used to
produce a `(1, length)` tensor and crash the sampler; the melody is now encoded
with the trained range and vocabulary of the slot it will occupy and the other
voices are filled with rests. Metadata is padded or truncated to the melody
length, which fixes the assertion failure on imported scores with pickups or
trailing measures, and empty or zero-length input now raises a named error
instead of failing deep inside the sampler.

**Fixing an inner voice.** `melody_voice` was documented as 0/1/2/3 but the
generation span was built with `[min(others), max(others) + 1]`, which for
`melody_voice=1` or `2` expanded to all four voices and silently regenerated
the melody it was supposed to hold fixed. The voices to regenerate are now
enumerated explicitly through a new `voice_indices` argument on
`DeepBach.generation()`, so any voice can be held fixed.

**Cadential cue.** An imported melody normally carries no fermata, which left
the fermata metadata channel — the model's only explicit phrase-ending cue — at
zero for the whole piece. `derive_fermatas=True` (default) fills it with the
convention the model was trained on: the last beat of every two bars plus the
final note. Worth roughly +3.3 points of note-exact reconstruction accuracy on
Bach chorales (70.2% -> 67.9% of the gap to a true channel; leaving it zeroed
is 64.6%). Pass `False` for the previous behaviour. The channel is a
conditioning input, not part of the generated music, so `fermata_marks`
controls how it is written out: `"none"` (default) writes the XML with no
fermata marks, `"input"` keeps only the input file's own, `"channel"` draws the
channel as-is. Purely notational — a music21 Fermata carries no duration, so
removing one changes no note's offset, length or pitch.

**Range diagnostic.** DeepBach encodes any note outside the trained range of
the voice it occupies as a single out-of-range token, losing the pitch. A
melody well below C4 therefore reaches the model as a nearly constant line.
`harmonize()` now reports how many notes fall outside the fixed voice's range
and suggests the voice that fits better, instead of failing silently. The
melody is never transposed for you.

**Two metadata fixes.** `FermataMetadata.generate()` tested
`len(note.expressions) == 1`, which reports any single expression as a fermata
and misses a real fermata sharing a note with another expression; it now tests
for `music21.expressions.Fermata` directly. It also indexed `list_notes[0]` on
a part with no notes and raised `IndexError`. The MuseScore server no longer
returns the conditioning channel painted onto the soprano of its output.




# DeepBach
This repository contains implementations of the DeepBach model described in

*DeepBach: a Steerable Model for Bach chorales generation*<br/>
Gaëtan Hadjeres, François Pachet, Frank Nielsen<br/>
*ICML 2017 [arXiv:1612.01010](http://proceedings.mlr.press/v70/hadjeres17a.html)*


The code uses python 3.9 together with [PyTorch v2.0](https://pytorch.org/) and
 [music21](http://web.mit.edu/music21/) libraries.

For the original Keras version, please checkout the `original_keras` branch.

Examples of music generated by DeepBach are available on [this website](https://sites.google.com/site/deepbachexamples/)

Models, Dataset caches and Deployment script are available on [Google Drive](https://drive.google.com/drive/folders/1ZbZiDmX3yShaelS3p_xdGBEQsPoHi6tb?usp=drive_link)

