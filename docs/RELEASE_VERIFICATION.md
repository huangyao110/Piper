# Piper source release verification — 2026-10-26

This report verifies the source organization and interfaces. It does not report a new training run or a real-model predictive benchmark.

## Checks completed

| Check | Result |
|---|---|
| Offline CPU regression suite | **204 passed, 2 skipped, 11 deselected** |
| Independent CLIP-gate research tests | **24 passed** |
| Clean exported source tree | Both suites passed without local experiments, model weights, data shards or vendor snapshots |
| CLI | Top-level help/version and train/evaluate/split/predict help passed |
| Package metadata/build | `piper_contact-0.1.0` wheel built with no dependencies installed or production environment changes |
| Reference environment | `pip check`: no broken requirements |
| Shell scripts / diff | `bash -n` and `git diff --cached --check` passed |
| Protected source/protocol/data records | **55 files checked, zero hash changes** |
| Portable training example | Identical scientific settings to the original config; filesystem paths only differ |
| Publication inventory | No runtime data/weights/log directories staged; no file over 5 MB; token/private-key/literal-password scan had no findings |

The two skipped cases require CUDA offload/restore. The eleven deselected cases are ten archived experiment-controller checks and one historical MMseqs evidence integration. They remain available under the `integration` marker. The standalone upstream GPU padding script was not run. SentencePiece emitted two existing SWIG deprecation warnings; these did not fail the suite.

## Test environment and isolation

Python 3.12, the current pinned PyTorch 2.8.0 + CUDA 12.8 environment, with CUDA hidden for verification. Missing test-only DuckDB 1.4.3 and Gemmi 0.7.5 packages were installed to a temporary directory, not the running training environment. Tests used offline Hugging Face settings, one OpenMP/OpenBLAS thread, temporary synthetic inputs, and an exported copy of the staged repository. The gate-study tests use two small historical gate fixtures with recorded hashes rather than depending on excluded vendor snapshots.

Commands from the clean source export:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -m 'not integration' -rs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m unittest discover -s resercher/tests -v
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/piper-package .
```

## Preservation and limits

The cleanup removed **43 retired files** from active locations and moved old documentation/upstream examples into explicit directories. It retained local compressed pre-cleanup archives. See [the per-file cleanup manifest](PIPER_CLEANUP_20261026.json).

No data split was rebuilt, no MMseqs threshold/command/lock changed, and no checkpoint was deleted. The running trainer and its supervisor continued using their independent runtime snapshot. The 55 protected hashes include the trainer's source-file set, original locked filters, data configurations/locks, selected historical membership records and active run controller/configuration metadata.

No new full-model GPU prediction, fresh training or throughput experiment was performed, because the existing job occupied the GPU and the core training implementation was preserved. CPU fixtures validate behavior and portability, not biological performance. A GitHub CPU workflow is included; its remote execution status is separate from the local results above.

## Follow-up: unified dependencies and cloud portability

The six requirements files were consolidated into one root `requirements.txt`, with 21 direct dependencies covering training, prediction, data preparation and tests. Existing direct version pins were retained. All 109 previously recorded installed versions are preserved in [ENVIRONMENT_REFERENCE.json](ENVIRONMENT_REFERENCE.json) as historical provenance. Installation scripts, package metadata, CI and current documentation use the single list. Pip's installation dry-run succeeded without modifying the running environment; the built wheel's dependency metadata exactly matches the unified list.

The initial GitHub runs exposed two real-MMseqs end-to-end tests incorrectly included in the offline test selection, and a one-ULP FP32 loss reduction difference across CPU builds. The two external-binary cases are now explicitly marked `integration`, with `PIPER_MMSEQS` selecting the test executable. The loss comparison uses `rtol=1e-7, atol=1e-7`; exact token/mask replay and the existing gradient assertions remain in place. No production numerical or filtering implementation was changed.

After these corrections, the local offline suite passed **202 tests**, with **2 GPU cases skipped** and **13 integration cases deselected**. The two real-MMseqs integrations were run separately with the pinned local executable: **2 passed**. This accounts for the change in the offline count from the initial 204. The 55 protected source/data hashes remain unchanged. The only data-builder edit updates the dependency filename in its generated README text; its executable logic is unchanged.
