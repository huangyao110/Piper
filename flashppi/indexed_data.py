"""Read immutable data through small, versioned split indexes.

The index contains row locations and sampling annotations, never sequences or
contact matrices. Source hashes are checked once per opened shard. Cluster IDs
are supplied by the index because PPI and DDI source namespaces differ.
"""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pyarrow.parquet as pq


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sequence_id(sequence):
    return 'seq_' + hashlib.sha256(sequence.encode('ascii')).hexdigest()


def contact_fingerprint(seq1, seq2, valid1, valid2, contacts, kind='ppi', canonical=True):
    """Hash exact binary supervision, including unknown-position masks.

    Chain swaps transpose contacts. Equal-chain (homomer) examples also compare
    both orientations. Different conformations are never unioned. The ordered
    variant verifies that legacy sequence axes and payloads have not changed.
    """
    valid1, valid2 = np.asarray(valid1, dtype=np.bool_), np.asarray(valid2, dtype=np.bool_)
    contacts = np.asarray(contacts, dtype=np.int32).reshape(-1, 2)
    if len(valid1) != len(seq1) or len(valid2) != len(seq2):
        raise ValueError('Sequence and residue-mask lengths disagree')
    if len(contacts) and ((contacts < 0).any() or
                         (contacts[:, 0] >= len(seq1)).any() or
                         (contacts[:, 1] >= len(seq2)).any()):
        raise ValueError('Contact lies outside its sequence axes')

    def digest(a, b, va, vb, xy):
        if len(xy):
            order = np.lexsort((xy[:, 1], xy[:, 0]))
            xy = np.asarray(xy[order], dtype='<i4')
            # Labels are binary: repeated coordinates do not change supervision.
            keep = np.r_[True, np.any(xy[1:] != xy[:-1], axis=1)]
            xy = xy[keep]
        h = hashlib.sha256(b'flashppi-contact-supervision-v1\0' + kind.encode() + b'\0')
        for sequence, valid in ((a, va), (b, vb)):
            h.update(struct.pack('<Q', len(sequence)))
            h.update(sequence.encode('ascii'))
            h.update(np.packbits(valid, bitorder='little').tobytes())
        h.update(struct.pack('<Q', len(xy)))
        h.update(np.asarray(xy, dtype='<i4').tobytes())
        return h.digest()

    forward = digest(seq1, seq2, valid1, valid2, contacts)
    if not canonical:
        return forward
    if seq1 > seq2:
        return digest(seq2, seq1, valid2, valid1, contacts[:, ::-1])
    if seq1 == seq2:
        return min(forward, digest(seq2, seq1, valid2, valid1, contacts[:, ::-1]))
    return forward


class IndexedSplit:
    """Bounded-memory random access to a split and its original data shards.

    Roots may be relocated by passing root_overrides. Physical source identity
    still comes from the manifest hashes, not from absolute filesystem paths.
    No CUDA, model weights, or DuckDB are needed by this reader.
    """
    def __init__(self, directory, split='train', root_overrides=None,
                 cache_row_groups=8, verify_hashes=True, allow_incomplete=False):
        self.directory = Path(directory).resolve()
        if (self.directory / 'FILTER_PROTOCOL_RETIRED.json').exists():
            raise ValueError('Index used a retired filtering protocol; use the locked original-protocol index')
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        # Import here: the protocol module uses this module's hashing utilities.
        from .data_protocol import (PROTOCOL_ID, LOCK_SHA256, STRICT_PROTOCOL_ID, STRICT_LOCK_SHA256,
                                    RF2_PROTOCOL_ID, RF2_LOCK_SHA256, FRESH_PROTOCOL_IDS, read_strict_lock)
        protocols = [dict(id=PROTOCOL_ID, lock_sha256=LOCK_SHA256),
                     dict(id=STRICT_PROTOCOL_ID, lock_sha256=STRICT_LOCK_SHA256),
                     dict(id=RF2_PROTOCOL_ID, lock_sha256=RF2_LOCK_SHA256)]
        if self.manifest.get('filter_protocol') not in protocols:
            raise ValueError('Index does not use an approved locked filtering protocol')
        if self.manifest['filter_protocol']['id'] in FRESH_PROTOCOL_IDS:
            read_strict_lock(self.manifest['filter_protocol']['id'])
            if file_sha256(self.directory / 'filter_protocol.lock.json') != self.manifest['filter_protocol']['lock_sha256']:
                raise ValueError('Indexed strict protocol lock checksum mismatch')
            if self.manifest['state'] == 'complete' and self.manifest.get('audit_status') != 'passed':
                raise ValueError('Strict split has not passed independent acceptance')
        if self.manifest['state'] != 'complete' and not (
                allow_incomplete and self.manifest['state'] == 'verifying'):
            raise ValueError('Split build has not passed verification')
        registry_path = self.directory / 'sources.json'
        if verify_hashes and file_sha256(registry_path) != self.manifest['source_registry_sha256']:
            raise ValueError('Source registry checksum mismatch')
        self.registry = json.loads(registry_path.read_text())
        self.roots = {k: Path(v) for k, v in self.registry['roots'].items()}
        self.roots.update({k: Path(v) for k, v in (root_overrides or {}).items()})
        entry = self.manifest['splits'][split]
        path = self.directory / entry['file']
        if verify_hashes and file_sha256(path) != entry['sha256']:
            raise ValueError('Split index checksum mismatch')
        self.index = pq.ParquetFile(path)
        if self.index.metadata.num_rows != entry['rows']:
            raise ValueError('Split index row count changed')
        self.starts = np.r_[0, np.cumsum([self.index.metadata.row_group(i).num_rows
                                       for i in range(self.index.metadata.num_row_groups)])]
        self.length = int(self.starts[-1])
        self.capacity = max(1, int(cache_row_groups))
        self.verify_hashes = verify_hashes
        self.verified = set()
        self.index_cache, self.data_cache = OrderedDict(), OrderedDict()

    def __len__(self):
        return self.length

    @staticmethod
    def _cached(cache, key, loader, capacity):
        if key not in cache:
            cache[key] = loader()
            while len(cache) > capacity:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        group = bisect_right(self.starts, index) - 1
        table = self._cached(self.index_cache, group, lambda: self.index.read_row_group(group), 2)
        annotation = table.slice(index - int(self.starts[group]), 1).to_pylist()[0]
        source_id = int(annotation['shard_id'])
        source = self.registry['shards'][str(source_id)]
        path = self.roots[source['root']] / source['path']
        if self.verify_hashes and source_id not in self.verified:
            if file_sha256(path) != source['sha256']:
                raise ValueError('Source checksum mismatch: ' + str(path))
            self.verified.add(source_id)
        row = int(annotation['row_index'])
        starts = source['row_group_starts']
        data_group = bisect_right(starts, row) - 1
        key = (source_id, data_group)
        data = self._cached(self.data_cache, key,
                            lambda: pq.ParquetFile(path).read_row_group(data_group), self.capacity)
        result = data.slice(row - starts[data_group], 1).to_pylist()[0]
        if annotation['kind'] != 'interaction' and result.get('pair_id') != annotation['sample_id']:
            raise ValueError('Index sample identity does not match source row')
        result['split_annotation'] = annotation
        result['sample_id'] = annotation['sample_id']
        for field in ('seq1_cluster', 'seq2_cluster', 'sequence_cluster_scope',
                      'structure_cluster_id', 'structure_cluster_scope'):
            if field in annotation and annotation[field] is not None:
                result[field] = annotation[field]
        return result

    def __getitems__(self, indices):
        return [self[i] for i in indices]
