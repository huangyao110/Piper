"""Train through the frozen split index without materializing source payloads.

Only sampling/row-location metadata is resident. Each optimizer batch groups
source reads by row group, then restores the exact sampled order (including
repeated rows). Sequence axes, contact coordinates and valid masks are unchanged.
"""
from bisect import bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .indexed_data import IndexedSplit, file_sha256


POLICY = 'balanced_ppi_sequence_ddi_structure_weighted_v1'
RF2_SAMPLING_POLICY = 'balanced_sequence_clusters_source_quota_v3'
RF2_SOURCE_QUOTA = {'ddi/conglab': 0.5, 'ppi/conglab': 0.25, 'ppi/distilled': 0.25}
DISTILLED_SOURCES = frozenset(('conglab_pred', 'afdb_complexs_pred', 'complex_atlas_pred',
                              'huintaf2_pred', 'yeast_core_pred', 'endomap_pred',
                              'predictomes_pred', 'gwyre_pred'))
METADATA_COLUMNS = ['sample_id', 'shard_id', 'row_index', 'kind', 'structure_source',
                    'legacy_train', 'seq1_cluster', 'seq2_cluster', 'within_pair_weight',
                    'structure_cluster_id', 'structure_cluster_scope']
PAYLOAD_COLUMNS = ['pair_id', 'seq1', 'seq2', 'contacts', 'valid1', 'valid2']


def verify_indexed_inputs(config):
    """Validate the completed index, fixed holdouts and approved protocol lock."""
    from .data_protocol import (read_lock, PROTOCOL_ID, LOCK_SHA256,
                                RF2_PROTOCOL_ID, RF2_LOCK_SHA256)
    directory = Path(config['indexed_split'])
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (directory / 'FILTER_PROTOCOL_RETIRED.json').exists():
        raise ValueError('Cannot train on a retired filtering protocol')
    if manifest['state'] != 'complete' or manifest.get('audit_status') != 'passed':
        raise ValueError('Indexed split has not passed its independent audit')
    if manifest['filter_protocol'] == dict(id=RF2_PROTOCOL_ID, lock_sha256=RF2_LOCK_SHA256):
        return _verify_rf2_inputs(config, directory, manifest)
    if manifest['filter_protocol'] != dict(id=PROTOCOL_ID, lock_sha256=LOCK_SHA256):
        raise ValueError('Indexed training requires the locked original filtering protocol')
    if config['indexed_sampling'] != POLICY or not config.get('balanced_ppi_ddi'):
        raise ValueError('Indexed training requires the explicit balanced group sampling policy')
    if config.get('epochs'):
        raise ValueError('Grouped replacement sampling uses an explicit step budget, not dataset epochs')
    if config['eligible_train_rows'] != manifest['splits']['train']['rows']:
        raise ValueError('Indexed population size changed')
    if config['cluster_multiplier'] != manifest['cluster_multiplier']:
        raise ValueError('Expanded known-positive cluster namespace must be used')
    entry = manifest['splits']['train']
    if Path(config['train']).resolve() != (directory / entry['file']).resolve():
        raise ValueError('Train path does not match the selected index')
    expected = {str(manifest_path): file_sha256(manifest_path),
                str(directory / entry['file']): entry['sha256'],
                str(directory / 'sources.json'): manifest['source_registry_sha256']}
    for name in ('known_positive_cluster_pairs.parquet', 'verification.json',
                 'independent_audit.json', 'protocol_acceptance.json', 'filter_protocol.lock.json'):
        expected[str(directory / name)] = manifest['artifacts'][name]['sha256']
    for name in ('verification.json', 'independent_audit.json', 'protocol_acceptance.json'):
        if json.loads((directory / name).read_text())['state'] != 'passed':
            raise ValueError('Index acceptance failed: ' + name)
    lock = read_lock()
    for key, name in (('validation_structures', 'validation.parquet'),
                      ('validation_interactions', 'interaction.parquet')):
        expected[config[key]] = lock['frozen_outputs'][name]['sha256']
    expected[config['test']] = lock['test_sha256']
    return expected


def _verify_rf2_inputs(config, directory, manifest):
    """Accept v3 explicitly without relaxing the historical v1 checks."""
    from .data_protocol import read_strict_lock, RF2_PROTOCOL_ID, RF2_LOCK_SHA256
    lock = read_strict_lock(RF2_PROTOCOL_ID)
    if config.get('indexed_sampling') != RF2_SAMPLING_POLICY or not config.get('balanced_ppi_ddi'):
        raise ValueError('RF2 v3 requires sequence-cluster sampling for both PPI and DDI')
    if config.get('indexed_source_quota') != RF2_SOURCE_QUOTA:
        raise ValueError('RF2 training requires the approved 50/25/25 source quota')
    if config.get('batch_size', 0) < 4 or config['batch_size'] % 4:
        raise ValueError('Exact source quotas require a global batch divisible by four')
    if config.get('validation_structures') or config.get('validation_interactions'):
        raise ValueError('RF2 v3 has no validation split')
    if config.get('evaluation_split') != 'test':
        raise ValueError('RF2 v3 evaluates the explicitly requested fixed test')
    if set(manifest['splits']) != {'train', 'test'}:
        raise ValueError('RF2 v3 must contain only train and test indexes')
    train = manifest['splits']['train']
    if config['eligible_train_rows'] != train['rows']:
        raise ValueError('Indexed population size changed')
    if config['cluster_multiplier'] != manifest['cluster_multiplier']:
        raise ValueError('Known-positive cluster namespace changed')
    if Path(config['train']).resolve() != (directory / train['file']).resolve():
        raise ValueError('Train path does not match the selected index')
    epochs = config.get('epochs')
    if (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
            or config['steps'] != epochs * math.ceil(train['rows'] / config['batch_size'])
            or config.get('phase_start_step', 0) != 0):
        raise ValueError('RF2 epochs must specify complete replacement draw budgets from step zero')
    expected = {str(directory / 'manifest.json'): file_sha256(directory / 'manifest.json'),
                str(directory / 'sources.json'): manifest['source_registry_sha256'],
                config['test']: lock['required_config']['test_sha256']}
    for entry in manifest['splits'].values():
        expected[str(directory / entry['file'])] = entry['sha256']
    for name in ('known_positive_cluster_pairs.parquet', 'verification.json',
                 'independent_audit.json', 'filter_protocol.lock.json'):
        expected[str(directory / name)] = manifest['artifacts'][name]['sha256']
    if expected[str(directory / 'filter_protocol.lock.json')] != RF2_LOCK_SHA256:
        raise ValueError('RF2 filtering lock changed')
    for name in ('verification.json', 'independent_audit.json'):
        if json.loads((directory / name).read_text())['state'] != 'passed':
            raise ValueError('Index acceptance failed: ' + name)
    return expected


class RowBatch:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


class IndexedTrainingPopulation:
    def __init__(self, directory):
        self.reader = IndexedSplit(directory, 'train', cache_row_groups=8)
        self.metadata = self.reader.index.read(columns=METADATA_COLUMNS)
        self.files = {}
        self.signatures = {}

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, column):
        return self.metadata[column]

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self)).any():
            raise IndexError('Training index outside the selected split')
        annotations = self.metadata.take(indices).to_pylist()
        groups = defaultdict(list)
        for position, annotation in enumerate(annotations):
            sid, row = int(annotation['shard_id']), int(annotation['row_index'])
            source = self.reader.registry['shards'][str(sid)]
            group = bisect_right(source['row_group_starts'], row) - 1
            groups[sid, group].append((position, row - source['row_group_starts'][group]))
        rows = [None] * len(indices)
        for (sid, group), locations in groups.items():
            source = self.reader.registry['shards'][str(sid)]
            path = self.reader.roots[source['root']] / source['path']
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            if sid not in self.reader.verified:
                if file_sha256(path) != source['sha256']:
                    raise ValueError('Source checksum mismatch: ' + str(path))
                self.reader.verified.add(sid)
                self.signatures[sid] = signature
                self.files[sid] = pq.ParquetFile(path)
            elif signature != self.signatures[sid]:
                raise ValueError('Source changed during training: ' + str(path))
            table = self.files[sid].read_row_group(group, columns=PAYLOAD_COLUMNS)
            selected = table.take([offset for _, offset in locations]).to_pylist()
            for (position, _), row in zip(locations, selected):
                annotation = annotations[position]
                if row['pair_id'] != annotation['sample_id']:
                    raise ValueError('Index sample identity does not match source row')
                row.update(seq1_cluster=annotation['seq1_cluster'],
                           seq2_cluster=annotation['seq2_cluster'], kind=annotation['kind'],
                           structure_source=annotation['structure_source'],
                           legacy_train=annotation['legacy_train'])
                rows[position] = row
        return RowBatch(rows)


class IndexedKindSampler:
    """Exact 1:1 types, uniform PPI sequence / DDI structure clusters.

The index already stores the conditional weight 1/(supervisions * members).
    Its sum is one per sequence pair. PPI chooses a protein sequence cluster,
    then weights interactions containing that cluster; a same-cluster pair is
    counted once. DDI chooses a structure cluster, then uses the same base
    weights within it. The code marginalizes the first choice into a row CDF.
    There is no extra source quota. Per-step RNG makes schedule prefixes stable.
"""
    def __init__(self, kinds, weights, batch_size, seed, structure_clusters, structure_scopes,
                 seq1_clusters, seq2_clusters):
        kinds, weights = np.asarray(kinds), np.asarray(weights, dtype=np.float64)
        if batch_size < 2 or batch_size % 2:
            raise ValueError('Exact 1:1 sampling requires an even global batch')
        if set(kinds) != {'ppi', 'ddi'} or len(kinds) != len(weights):
            raise ValueError('Both types and one weight per training row are required')
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Every retained training row needs a finite positive weight')
        if len(structure_clusters) != len(kinds) or len(structure_scopes) != len(kinds):
            raise ValueError('Missing structure cluster annotations')
        if len(seq1_clusters) != len(kinds) or len(seq2_clusters) != len(kinds):
            raise ValueError('Missing sequence cluster annotations')
        ppi = np.flatnonzero(kinds == 'ppi')
        a, b = np.asarray(seq1_clusters)[ppi], np.asarray(seq2_clusters)[ppi]
        if (not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or
                (b < 0).any() or (a != np.floor(a)).any() or (b != np.floor(b)).any()):
            raise ValueError('Every retained PPI needs integer sequence clusters')
        sequence_clusters, inverse_ppi = np.unique(np.r_[a, b], return_inverse=True)
        ia, ib = inverse_ppi[:len(ppi)], inverse_ppi[len(ppi):]
        different = ia != ib
        totals_ppi = np.bincount(ia, weights=weights[ppi], minlength=len(sequence_clusters))
        totals_ppi += np.bincount(ib[different], weights=weights[ppi][different],
                                  minlength=len(sequence_clusters))
        ppi_factor = 1.0 / totals_ppi[ia] + different / totals_ppi[ib]
        ddi = np.flatnonzero(kinds == 'ddi')
        ids = np.asarray(structure_clusters)[ddi]
        scopes = np.asarray(structure_scopes)[ddi]
        if set(scopes) != {'ddi_teddymer_extension'}:
            raise ValueError('DDI structure cluster namespace is missing or changed')
        if not np.isfinite(ids).all() or (ids < 0).any() or (ids != np.floor(ids)).any():
            raise ValueError('Every retained DDI needs an integer structure cluster')
        clusters, inverse = np.unique(ids.astype(np.int64), return_inverse=True)
        totals = np.bincount(inverse, weights=weights[ddi])
        weights = weights.copy()
        weights[ppi] *= ppi_factor
        weights[ddi] /= totals[inverse]
        self.ppi_sequence_clusters = len(sequence_clusters)
        self.ddi_structure_clusters = len(clusters)
        self.effective_weights = weights
        self.pools, self.cdf = {}, {}
        for kind in ('ppi', 'ddi'):
            pool = np.flatnonzero(kinds == kind)
            cumulative = np.cumsum(weights[pool], dtype=np.float64)
            cumulative /= cumulative[-1]
            cumulative[-1] = 1.0
            self.pools[kind], self.cdf[kind] = pool, cumulative
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.num_samples = 0

    def batch(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step)]))
        half = self.batch_size // 2
        selected = np.concatenate([self.pools[k][np.searchsorted(
            self.cdf[k], rng.random(half), side='right')] for k in ('ppi', 'ddi')])
        rng.shuffle(selected)
        return selected

    def __iter__(self):
        if self.num_samples < 1 or self.num_samples % self.batch_size:
            raise ValueError('Sampling requires whole optimizer batches')
        for step in range(self.num_samples // self.batch_size):
            yield from self.batch(step).tolist()


class IndexedSequenceQuotaSampler:
    """Fixed source groups, then uniform endpoint clusters within each group.

    Within a selected cluster, rows retain their deduplication weights. A pair
    whose endpoints share a cluster contributes only once to that cluster.
    The eight distilled PPI sources form ONE pool, not eight equal quotas.
    This balances clusters conditionally on source group, not across groups.
    """
    def __init__(self, kinds, sources, weights, batch_size, seed, seq1_clusters,
                 seq2_clusters, source_quota):
        if source_quota != RF2_SOURCE_QUOTA:
            raise ValueError('Expected the approved 50/25/25 source quota')
        if batch_size < 4 or batch_size % 4:
            raise ValueError('Exact source quotas require a global batch divisible by four')
        kinds, sources = np.asarray(kinds), np.asarray(sources)
        weights = np.asarray(weights, dtype=np.float64)
        a, b = np.asarray(seq1_clusters), np.asarray(seq2_clusters)
        if any(x.ndim != 1 or len(x) != len(kinds) for x in (sources, weights, a, b)):
            raise ValueError('One source, weight and two sequence clusters per row are required')
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Every retained row needs a finite positive weight')
        for values in (a, b):
            if (not np.isfinite(values).all() or (values < 0).any()
                    or (values != np.floor(values)).any()):
                raise ValueError('Every retained row needs integer sequence clusters')
        allowed = ((kinds == 'ddi') & (sources == 'conglab')) | (
            (kinds == 'ppi') & np.isin(sources, ['conglab', *DISTILLED_SOURCES]))
        if not allowed.all():
            raise ValueError('Unexpected source in the RF2 training population')
        groups = np.where(kinds == 'ddi', 'ddi/conglab',
                          np.where(sources == 'conglab', 'ppi/conglab', 'ppi/distilled'))
        self.pools, self.cdf, self.counts, self.group_cluster_counts = {}, {}, {}, {}
        self.effective_weights = np.zeros(len(kinds), dtype=np.float64)
        for group, fraction in RF2_SOURCE_QUOTA.items():
            pool = np.flatnonzero(groups == group)
            if not len(pool):
                raise ValueError('Source quota has no eligible members: ' + group)
            clusters, inverse = np.unique(np.r_[a[pool], b[pool]], return_inverse=True)
            ia, ib = inverse[:len(pool)], inverse[len(pool):]
            different = ia != ib
            total = np.bincount(ia, weights=weights[pool], minlength=len(clusters))
            total += np.bincount(ib[different], weights=weights[pool][different],
                                 minlength=len(clusters))
            marginal = weights[pool] * (1.0 / total[ia] + different / total[ib])
            marginal /= marginal.sum()
            self.effective_weights[pool] = fraction * marginal
            cumulative = np.cumsum(marginal, dtype=np.float64)
            cumulative[-1] = 1.0
            self.pools[group], self.cdf[group] = pool, cumulative
            self.counts[group] = int(batch_size * fraction)
            self.group_cluster_counts[group] = len(clusters)
        self.ppi_sequence_clusters = len(np.unique(np.r_[a[kinds == 'ppi'], b[kinds == 'ppi']]))
        self.ddi_sequence_clusters = len(np.unique(np.r_[a[kinds == 'ddi'], b[kinds == 'ddi']]))
        self.batch_size, self.seed, self.num_samples = int(batch_size), int(seed), 0

    def batch(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step)]))
        selected = np.concatenate([self.pools[group][np.searchsorted(
            self.cdf[group], rng.random(count), side='right')] for group, count in self.counts.items()])
        rng.shuffle(selected)
        return selected

    def __iter__(self):
        if self.num_samples < 1 or self.num_samples % self.batch_size:
            raise ValueError('Sampling requires whole optimizer batches')
        for step in range(self.num_samples // self.batch_size):
            yield from self.batch(step).tolist()


def build_indexed_sampler(config, table):
    if config['indexed_sampling'] == RF2_SAMPLING_POLICY:
        return IndexedSequenceQuotaSampler(
            table['kind'].to_numpy(), table['structure_source'].to_numpy(),
            table['within_pair_weight'].to_numpy(), config['batch_size'], config['seed'],
            table['seq1_cluster'].to_numpy(), table['seq2_cluster'].to_numpy(),
            config['indexed_source_quota'])
    if config['indexed_sampling'] != POLICY:
        raise ValueError('Unknown indexed sampling policy')
    return IndexedKindSampler(table['kind'].to_numpy(), table['within_pair_weight'].to_numpy(),
                              config['batch_size'], config['seed'],
                              table['structure_cluster_id'].to_numpy(),
                              table['structure_cluster_scope'].to_numpy(),
                              table['seq1_cluster'].to_numpy(), table['seq2_cluster'].to_numpy())


def validate_continuation(old, new, checkpoint_step):
    """Data expansion is explicit; model, objective and Adam hyperparameters stay fixed."""
    allowed = {'train', 'indexed_split', 'indexed_sampling', 'eligible_train_rows',
               'cluster_multiplier', 'steps', 'epochs', 'phase_start_step',
               'checkpoint_retention', 'sampling_block_steps'}
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if changed - allowed:
        raise ValueError('Continuation changes model/training semantics: ' + ', '.join(sorted(changed - allowed)))
    if not new.get('indexed_split') or new.get('indexed_sampling') != POLICY:
        raise ValueError('Continuation requires an explicitly selected indexed population')
    if new.get('phase_start_step') != checkpoint_step or new['steps'] <= checkpoint_step:
        raise ValueError('Continuation budget must begin at the saved optimizer step')
    if old.get('learning_rate_schedule') != 'constant':
        raise ValueError('Data continuation currently requires an unchanged constant LR schedule')
    return sorted(changed)


def prune_checkpoints(output, best, keep_last=2, keep_steps=()):
    """Prune only complete checkpoints owned by this run, retaining recovery/best."""
    import shutil
    output = Path(output).resolve()
    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 2:
        raise ValueError('Keep at least two recovery checkpoints')
    root = output / 'checkpoints'
    complete = sorted(p for p in root.glob('step_*') if p.name[5:].isdigit()
                      and p.is_dir() and not p.is_symlink() and (p / 'manifest.json').is_file())
    protected = set(complete[-keep_last:])
    protected.update((output / record['checkpoint']).resolve() for record in best.values())
    protected.update(root / f'step_{step:06d}' for step in keep_steps)
    removed = []
    for path in complete:
        if path not in protected:
            record = json.loads((path / 'manifest.json').read_text())
            if path.resolve().parent != root or record['step'] != int(path.name[5:]):
                raise ValueError('Checkpoint ownership mismatch')
            with (output / 'checkpoint_retention.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(removed=path.name, manifest=record)) + '\n')
            shutil.rmtree(path)
            removed.append(path.name)
    return removed
