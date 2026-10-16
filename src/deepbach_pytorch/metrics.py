"""
Training-run metrics: parameter accounting, top-k accuracy, wall time, peak VRAM.

Pure measurement, same contract as `analysis.py`: it reads a model (or logits and
labels) and returns dicts, does no I/O at import and prints nothing. The two training
loops in `voice_model.py` / `shared_trunk_model.py` are what call it and what write the
result into the `_history/` payload (see `training_state.save_history`).

Why the numbers are measured here rather than re-derived later: everything in
`run_metrics_block` is only observable **while the run is happening**. A parameter count
can be recovered from a checkpoint afterwards, but the wall time, the peak VRAM and the
test-split evaluation at the end of that particular run cannot.
"""

import time

import torch

# Top-k accuracies reported alongside the top-1 the trainer already computed. The table
# this feeds names them `val_top3` / `val_top5`; `topk_accuracy` returns those keys.
TOP_K = (1, 3, 5)

# Top-level module name -> the group its parameters count as. Both architectures have
# exactly these six as `nn.Module` children (`VoiceModel.__init__`,
# `SharedTrunkModel.__init__`), so the table is total for the shapes this repo builds.
# Anything outside it is counted under `other` **and reported** (`ungrouped_modules`)
# rather than dropped, so a new module shows up as an anomaly instead of as a total
# that quietly disagrees with `model.parameters()`.
PARAM_GROUPS = (
    ('embedding', ('note_embeddings', 'meta_embeddings')),
    ('recurrent', ('lstm_left', 'lstm_right')),
    ('head',      ('mlp_center', 'mlp_predictions')),
)


def param_counts(model):
    """
    Scalar counts of `model`'s parameters, total and by group.

    Classified by the **first segment of the parameter's dotted name**, from
    `named_parameters()`. That is deliberate rather than walking `named_children()` and
    summing each child: `named_parameters()` yields every parameter exactly once (a
    parameter shared by two children would be counted twice by the per-child route,
    and the groups would then not add up to `total`).

    :return: dict with `total`, `trainable`, one key per group in `PARAM_GROUPS`, an
        `other` group, and `ungrouped_modules` (the module names that landed in
        `other`; empty is the healthy value)
    """
    group_of = {name: group for group, names in PARAM_GROUPS for name in names}
    counts = {group: 0 for group, _ in PARAM_GROUPS}
    counts['other'] = 0
    total = 0
    trainable = 0
    ungrouped = set()

    for param_name, param in model.named_parameters():
        n = int(param.numel())
        head = param_name.split('.', 1)[0]
        group = group_of.get(head)
        if group is None:
            group = 'other'
            ungrouped.add(head)
        counts[group] += n
        total += n
        if param.requires_grad:
            trainable += n

    counts['total'] = total
    counts['trainable'] = trainable
    counts['ungrouped_modules'] = sorted(ungrouped)
    return counts


def topk_accuracy(logits, target, ks=TOP_K):
    """
    Top-k accuracies in percent, over the **note** vocabulary.

    `logits` are the model's output for one voice (or one role), whose last dimension is
    that voice's note vocabulary — the same tensor `VoiceModel.accuracy` takes the
    argmax of, so `top1` and the trainer's `acc` are the same quantity computed twice
    and must agree.

    k greater than the number of classes raises: a "top-5" over three classes is 100%
    by construction, and silently reporting that as an accuracy would be a fabricated
    number, not a measured one.

    :return: dict keyed `top1` / `top3` / `top5` (only the requested `ks`)
    """
    n_classes = int(logits.size(1))
    wanted = sorted({int(k) for k in ks})
    largest = max(wanted)
    if largest > n_classes:
        raise ValueError(
            f'top-{largest} requested but the output has only {n_classes} '
            f'class(es); a top-k accuracy above the class count is trivially 100%')

    _, top = logits.topk(largest, dim=1)
    target = target.unsqueeze(1)
    return {f'top{k}': float((top[:, :k] == target).any(dim=1).float().mean().item() * 100)
            for k in wanted}


def mean_topk(per_batch):
    """
    Mean of a list of `topk_accuracy` dicts, key by key.

    Per-batch means averaged over batches, matching how the two trainers average loss and
    accuracy (`loss_and_acc`'s denominator is the number of batches actually run). An
    empty list raises rather than returning zeros: that would read as a real 0%.
    """
    if not per_batch:
        raise ValueError('mean_topk got no batches; there is no mean of nothing')
    keys = per_batch[0].keys()
    return {k: sum(d[k] for d in per_batch) / len(per_batch) for k in keys}


def eval_metrics(loss, accuracy, topk, batches, windows):
    """One split's whole evaluation result, as the flat dict the payload stores."""
    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'top1': float(topk['top1']),
        'top3': float(topk['top3']),
        'top5': float(topk['top5']),
        'batches': int(batches),
        'windows': int(windows),
    }


class Stopwatch:
    """
    Wall-clock seconds of the `with` body, as `.seconds` once it has exited.

    `perf_counter`, not `time()`: a duration is what is being measured, and the wall
    clock can step. Deliberately not a `timeit`-style repeat-and-average: one pass over a
    dataset is the measurement, and repeating it would triple the cost of the run it is
    measuring.
    """

    def __enter__(self):
        self.seconds = None
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.seconds = time.perf_counter() - self._t0
        return False


def _mb(n_bytes):
    return round(n_bytes / (1 << 20), 1)


def device_info():
    """
    The card this run is on, and how big it is.

    Recorded next to the peak memory because a peak without the device's capacity is not
    interpretable: 1.2 GB is comfortable on an 8 GB card and 90% of a 1.3 GB one.
    """
    if not torch.cuda.is_available():
        return {'device_name': 'cpu', 'device_total_memory_mb': None}
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {'device_name': props.name,
            'device_total_memory_mb': _mb(props.total_memory)}


def reset_peak_gpu_memory():
    """
    Zero the peak counters, so the next `peak_gpu_memory_mb` covers only what follows.

    Called at the **start of training** (not at the start of the process): before that,
    the peak would include dataset construction, and this number is meant to answer "what
    does training this model need".
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb():
    """
    Peak CUDA memory **of this process** since the last reset, in MB.

    `allocated` is live tensor memory, `reserved` is what the caching allocator holds onto
    (`reserved >= allocated`); the pair is what tells a genuine capacity problem apart
    from fragmentation. Both are None without CUDA — not 0, which would read as a
    measurement.
    """
    out = dict(device_info())
    if not torch.cuda.is_available():
        out.update({'allocated': None, 'reserved': None})
        return out
    out.update({'allocated': _mb(torch.cuda.max_memory_allocated()),
                'reserved': _mb(torch.cuda.max_memory_reserved())})
    return out


def run_metrics_block(model, *, seed, batch_size, steps_per_epoch, max_steps,
                      total_passes, train_seconds, pass_seconds, evals,
                      inference_seconds=None, inference_kind=None):
    """
    The run-level metrics for one trained model, as the dict written into the payload.

    Assembled from already-measured values; the only thing it measures itself is the
    peak memory, which is why it is called **after** the last device work of the run and
    before anything else touches CUDA. `evals` maps a label (`'test'`, `'test_final'`)
    to an `eval_metrics` dict.

    `params_*` keys come from `param_counts`; `other` and `ungrouped_modules` are carried
    through rather than dropped so a total that does not decompose is visible in the
    payload instead of only in a parameter count nobody re-checks.

    `inference_seconds` is the wall time of the forward-pass evaluation named by
    `inference_kind` (a **string**, because "inference time" of a teacher-forced pass over
    the test split and of a 500-iteration Gibbs generation differ by orders of magnitude
    and a bare number would be read as the wrong one).

    :param pass_seconds: per-pass wall times; `epoch_time_s` is their mean, and
        `training_time_s` their sum, so both describe the passes alone — the
        end-of-training test evaluation is neither
    """
    params = param_counts(model)
    peak = peak_gpu_memory_mb()

    block = {
        'model': type(model).__name__,
        'model_repr': model.__repr__(),
        'rnn_type': getattr(model, 'rnn_type', 'lstm'),
        'hidden_size': int(getattr(model, 'lstm_hidden_size')),
        'seed': None if seed is None else int(seed),
        'batch_size': int(batch_size),
        'steps_per_epoch': int(steps_per_epoch),
        'max_steps': int(max_steps),
        'total_passes': int(total_passes),
        # Two decimals, not one: a smoke run's passes are fractions of a second, and
        # rounding those to 0.0 would make `epoch_time_s` read as "not measured".
        'training_time_s': round(float(sum(pass_seconds)), 2),
        'epoch_time_s': (round(sum(pass_seconds) / len(pass_seconds), 2)
                         if pass_seconds else None),
        'pass_time_s': [round(float(s), 2) for s in pass_seconds],
        'inference_time_s': (None if inference_seconds is None
                             else round(float(inference_seconds), 2)),
        'inference_kind': inference_kind,
        'peak_gpu_memory_mb': peak['allocated'],
        'peak_gpu_memory_reserved_mb': peak['reserved'],
        'device_name': peak['device_name'],
        'device_total_memory_mb': peak['device_total_memory_mb'],
        'evals': dict(evals),
    }
    for group, _ in PARAM_GROUPS:
        block[f'params_{group}'] = params[group]
    block['params_other'] = params['other']
    block['params_other_modules'] = params['ungrouped_modules']
    block['trainable_params'] = params['trainable']
    block['total_params'] = params['total']
    return block


# `metrics` runs inside the training loop, whose stdout may be redirected to a file that
# the locale (cp936 here) has to encode. Every label below is plain ASCII, and the model
# repr -- the one field this module did not choose -- goes through `_ascii_printable`.
def _ascii_printable(text):
    """Squeeze arbitrary text to ASCII; a repr with a non-GBK character must not raise here."""
    return str(text).encode('ascii', 'backslashreplace').decode('ascii')


def summary_lines(block):
    """
    `run_metrics_block`'s result as printable ASCII lines.

    The same numbers the payload gets, so the training log alone answers "what did this run
    cost and score" without loading a file afterwards.
    """
    lines = ['  --- run metrics ({}) ---'.format(_ascii_printable(block['model_repr']))]
    if block['params_other_modules']:
        # Warn first and loudly: the parameter groups below do not add up to `total`.
        lines.append('  [warn] parameters outside the known groups: {}'.format(
            ', '.join(block['params_other_modules'])))
    lines.append('  params         : {} trainable of {} total '
                 '(emb {} + rnn {} + head {} + other {})'.format(
                     block['trainable_params'], block['total_params'],
                     block['params_embedding'], block['params_recurrent'],
                     block['params_head'], block['params_other']))
    lines.append('  hidden_size    : {}   cell: {}   seed: {}'.format(
        block['hidden_size'], block['rnn_type'],
        'unset' if block['seed'] is None else block['seed']))
    lines.append('  training_time_s: {}  ({} pass(es), mean {} s/pass)'.format(
        block['training_time_s'], block['total_passes'], block['epoch_time_s']))
    if block['inference_time_s'] is not None:
        lines.append('  inference_time_s: {}  ({})'.format(
            block['inference_time_s'], block['inference_kind']))
    lines.append('  peak_gpu_memory: {} MB allocated, {} MB reserved '
                 '(of {} MB on {})'.format(
                     block['peak_gpu_memory_mb'], block['peak_gpu_memory_reserved_mb'],
                     block['device_total_memory_mb'],
                     _ascii_printable(block['device_name'])))
    for label, ev in sorted(block['evals'].items()):
        if ev is None:
            lines.append('  {:<14}: not evaluated (empty split)'.format(label))
            continue
        lines.append('  {:<14}: loss {:.4f}  acc {:.2f}%  top3 {:.2f}%  top5 {:.2f}% '
                     '({} window(s))'.format(
                         label, ev['loss'], ev['accuracy'], ev['top3'], ev['top5'],
                         ev['windows']))
    return lines
