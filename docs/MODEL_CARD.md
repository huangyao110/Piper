# Piper model card

## Scope and release status

Piper predicts residue contacts between protein sequences and contrastive representations for interaction scoring. Its design combines three protein language models with residue-gated contrastive pooling and residual channel gating for contact prediction. This is a research code release; a final weight set has not yet been released. Performance, calibration and biological interpretation require separate evaluation. The name Piper is provisional.

## Representation and heads

Inputs use one token per residue, with explicit right-padding masks. Frozen encoders map supported ambiguous residues as documented in `flashppi/frozen_prott5.py`; the mapping and residue alignment have regression tests.

For a residue, the fused representation is

`h = h_gLM2 + alpha_E * P_E(LN(h_ESMC)) + alpha_T * P_T(LN(h_ProtT5))`.

ESM C has dimension 960, ProtT5 dimension 1024, and the fused gLM2 space dimension 1280. Both scalar fusion coefficients initialize at zero. ESM C and ProtT5 parameters remain frozen. In the current fresh run, gLM2's last 16 transformer blocks and the new task modules are trained.

The CLIP query and key heads each learn a residue score `w_i = sigmoid(MLP(LN(h_i)))`. Pooling is `sum_i(w_i*h_i)/sum_i(w_i)`, restricted to valid residues, followed by the head projection and vector normalization. A zero-initialized final score layer gives uniform weights initially.

Before contact prediction, the residue gate applies

`h'_i = h_i + beta * sigmoid(G(LN(h_i))) * P(LN(h_i))`.

`beta` initializes at zero. Each residue retains its position. The concat head joins the two residue sequences with segment embeddings, runs its transformer, and forms cross-chain contact logits from projected features. Padding and structurally unknown positions are masked separately. There is no patch pooling in the current contact training path.

## Training and evaluation

The original gLM2 backbone is initialized from its pinned pretrained checkpoint. The released FlashPPI architecture code supplies the compatible backbone implementation and configuration; fresh initialization does not load its task-trained weights. Hyperparameters, source quotas and sampling semantics are in [README](../README.md) and the original run configuration.

The primary scientific target is Contact P@K. The differentiable training objective is the weighted contact loss plus contrastive loss; P@K itself is a ranking metric, not a directly differentiated loss. Frozen length bucketing changes floating-point execution shape and is an explicit setting in this run. It should not be advertised as bitwise-equivalent to all older runs.

For each positive structure at cutoff d (12 Å or 8 Å):

1. Form the set C of unique observed true contacts, intersecting both structural validity masks.
2. Let K = |C|. A structure with K=0 is excluded from this cutoff's P@K mean.
3. Rank the valid residue-pair contact logits and take the top K.
4. Compute `|topK ∩ C| / K`, then macro-average across eligible structures.

Different cutoffs may have different eligible sets; always retain `evaluable_positives`. Full-length test evaluation preserves original pair orientation, without training crops or retrieval filtering. Pair interaction score is `sigmoid(max(valid residue contact logits))`; AP/AUROC describe interaction classification rather than contact localization. CLIP score is the query/key cosine similarity. Optional patch AP/P@K pool predictions for diagnostics and have different label density and units.

## Limitations

- The RF2-PPI v3 isolation predicate permits one training endpoint to hit the test reference. It does not certify C3 isolation or absence of all evolutionary relationships.
- The same historical test has been inspected during development. Periodic test monitoring is not evidence from an untouched final holdout; the current fixed schedule does not use it for automatic model selection.
- Generated hard negatives are training assumptions, not experimentally established non-interactions.
- Distilled structural labels may carry source-model bias and correlated errors. Cluster weighting reduces some sampling imbalance but does not establish statistical independence.
- A gate value is a learned pooling coefficient. It is not a causal attribution, conservation score or established biological importance measure. The isolated historical study in `resercher/` is exploratory and not automatically evidence about the current fresh model.
- Long pairwise contact maps and full-length encodings can exceed device memory. The prediction CLI intentionally does not silently crop.
- Existing research checkpoints store only trainable tensors. Reproduction also requires frozen model assets, architecture code, configuration, and exact data/source manifests.

No clinical, experimental binding-validation, calibrated probability or state-of-the-art claim is made by this source release.

## Implementation provenance

Piper incorporates and adapts FlashPPI code for the residue concat contact head and contrastive training components. The three-encoder fusion and two gating paths described above are project-specific design additions. Internal module names and upstream history remain intact for checkpoint compatibility and attribution. See the [acknowledgments and license](../README.md#acknowledgments-and-license) for source credit and licensing.
