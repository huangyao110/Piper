# Data protocol and isolation

The maintained entry point is `tools/train_test.py`. Do not create an alternative builder or alter historical locks to obtain a desired retained count. Active protocol: **conglab_distilled_ecoli_rf2ppi_v3**. Its configuration and `.lock.json` are kept alongside the earlier versions in `config/data/`.

## Inputs and units

The source release is `ppidb-2026.10.3`, with CongLab PPI/DDI and eight selected distilled PPI sources. PINDER and Teddymer are excluded from this fresh run. The locked source list, full-file hashes and test reference hash are in the v3 lock. Original paired FASTA entries are used before training crops; some entries are domains or structure fragments rather than canonical full proteins.

The reference contains the union of sequence endpoints in all 650 positive and 65,000 negative historical E. coli test pairs. It is not restricted to positive test proteins. There is no validation split.

## Exact v3 hit predicate

| Condition | Rule |
|---|---|
| Identity | `nident / alnlen > 0.30` |
| Significance | Reported MMseqs E-value `< 1e-5` |
| Coverage | `qcov > 0.50 OR tcov > 0.50` |
| Endpoint hit | Any reference endpoint satisfying all three conditions |
| Pair exclusion | Both training endpoints independently hit the reference |

All inequality boundaries are strict. PPI and DDI share this predicate. The two hits need not correspond to the same test pair. An exact sequence match does not bypass the significance condition. A single hit is allowed: **this is not C3**.

The paper evidence was reviewed from RF2-PPI supplementary M3.1 (pp. 20–21) and M3.3 (p. 27); the local review is preserved in [RF2PPI_ISOLATION_REVIEW_20261024.md](RF2PPI_ISOLATION_REVIEW_20261024.md). The full author MMseqs command/version was not published. Our reproducible implementation pins **18.8cc5c**, binary SHA256, arguments, deterministic input ordering, bidirectional k=6 searches and bidirectional direct searches, with no coverage prefilter (`-c 0`). These implementation details must not be attributed to the authors as their published command.

## Build and independent check

```bash
python tools/train_test.py --mode build --config config/data/conglab_distilled_ecoli_rf2ppi_v3.json
python tools/train_test.py --mode check --config config/data/conglab_distilled_ecoli_rf2ppi_v3.json
```

`build` produces indexes, evidence, manifests and an audit. `check` independently reparses archived raw alignments and significance evidence, then verifies source hashes, endpoint decisions, retained members, test content, deduplication and weights. It does not change membership, rerun MMseqs or refresh a protocol lock. It updates check reports and acceptance metadata. A failed check exits nonzero and marks the check/manifest as failed; an older passing report cannot override that failure.

Outputs include:

| Artifact | Purpose |
|---|---|
| `train.index.parquet` | Training row references into original source shards |
| `test.index.parquet` | References to the frozen test rows |
| `duplicate_map.parquet` | Aliases from discarded duplicate records to retained records |
| `manifest.json` | Source identity, protocol, file hashes, membership counts and audit state |
| `independent_audit.json`, `check_status.json` | Audit details and latest independent check state |
| Output `README.md` | Human-readable split provenance and usage |

The source datasets remain read-only. A passing mechanical audit establishes compliance with this fixed implementation, not exhaustive biological homology exclusion.

The 0.2 source consolidation moves the shared reader and protocol modules into `piper/`; the builder and independent checker remain at their existing paths. Predicate logic and protocol locks are unchanged. An unfinished build records its code hashes in `build_identity.json`: continue it using its original source version, rather than rewriting that identity to accept a different layout. Completed splits can still be checked through the maintained entry point.

Version 0.2.1 moves shared configurations and locks into `config/data/` without modifying their contents. The v1 lock's historical `configs_data/...` baseline keys are resolved to `config/data/...` before checking their original hashes. Missing or altered canonical files still fail verification, even if an old-path copy exists. The four locked scripts retain their original bytes; historical standalone utilities must receive an explicit configuration path as described in [the configuration guide](../config/README.md).

## Sampling and labels

The supplied 70/70 sequence cluster TSVs are sampling metadata for both PPI and DDI. They are distinct from the 30% isolation threshold. Draws are allocated 50% CongLab DDI, 25% CongLab PPI and 25% distilled PPI, then balanced over sequence clusters with within-cluster deduplication weights. The source mixture is not proportional to raw row count. Four epochs mean four budgets of balanced draws, not four exhaustive traversals.

Contacts remain residue labels at Cβ distance strictly below 12 Å (Cα for glycine), with missing structural positions masked. The organizational cleanup changes no source quality admission, contact labels, masks, cluster TSVs or split membership.

## Another machine

The historical configuration intentionally records the original absolute paths. Copy it to an ignored `config/data/rf2ppi_v3.local.json` and change **only storage locations** (`release`, `legacy_directory`, `test`, `output`, `mmseqs`). Keep the protocol id, seed, source scope, parameters and every `.lock.json` unchanged. The historical v1 train file is needed for before/after annotations, not as an exemption from v3 filtering. Copy all source shards and archived alignment evidence when checking an existing index split; indexes alone are not a dataset.

The builder verifies the exact MMseqs executable hash and required source hashes. If the binary or assets differ, stop and resolve their provenance rather than refreshing the lock. Moving completed split files may also require preserving the absolute locations encoded in their manifests; do not rewrite a completed manifest simply to bypass verification.

v1 and v2 have different predicates. Keep their small historical configurations, locks and comparisons for interpretation; do not apply the v3 rules retroactively to their stored outputs. The retired v2 output was deleted under separate cleanup authorization; [the cleanup record](DATA_CLEANUP_20261024.json) remains.
