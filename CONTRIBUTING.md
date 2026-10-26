# Contributing to Piper

Read [AGENTS.md](AGENTS.md) and the [data protocol](docs/DATA_PROTOCOL.md) before changes to filtering, labels, sampling or training. Keep changes focused and include the configuration and validation used.

The maintained entry points are documented in [tools/README.md](tools/README.md). Place exploratory scripts under a local `experiments/<name>/`; do not add competing builders to `tools/`. The `resercher/` CLIP-gate study remains isolated from production training.

Use four-space indentation, snake_case functions and PascalCase classes. Preserve tensor-shape and masking explanations. Do not rename locked source paths or rewrite filtering code for style: historical hashes and checkpoint recovery depend on them.

Before submitting, run the offline CPU suite and independent research tests described in README. Mark tests that need historical datasets, runtime files or external binaries as `integration`; do not silently weaken checks when an artifact is missing. Real GPU checks should identify hardware and pretrained artifacts and run only when the GPU is available.

Keep raw datasets, model weights, prediction outputs, credentials, machine crash reports and local environments out of commits. Use the pinned source/model manifests and small reproducibility records instead. A deliberate data protocol change requires a newly named version, explicit authorization, unchanged historical locks, and a before/after report.

Piper is a derivative of FlashPPI under the retained CC BY-NC 4.0 license. Preserve upstream attribution. Model and dataset licenses remain separate. Do not add unverified performance or biological interpretation claims to documentation.
