# Optional residue annotations

Only CLIP residue pooling gates are studied. Conservation and solvent exposure
are not fields in the frozen validation tables. **Real annotations were added on
2026-10-23** in `data/annotations.csv`, using the original coordinates and public
ConSurfDB conservation profiles/MSAs. See
[the completed annotation study](results/annotation_study/REPORT.md).
An absent field is missing evidence, never a score of zero.

Put annotations inside this directory, for example `data/annotations.csv`, with:

```csv
sequence_id,position,aa,conservation,relative_sasa,msa_neff,msa_coverage
```

- `sequence_id`: the full SHA256 of the exact sequence, as in `data/sequences.parquet`.
- `position`: **one-based**, full sequence position. Do not use PDB residue numbers directly.
- `aa`: exact residue at that position, checked before analysis.
- `conservation`: higher means more conserved; transform entropy beforehand and document it.
- `relative_sasa`: fraction in [0,1], with calculation method and normalization documented.
  Prefer exposure of the isolated chain when asking whether the gate recognizes
  interface residues beyond general surface exposure. Complex SASA already reflects burial.
- `msa_neff`, `msa_coverage`: optional alignment quality annotations; missing values remain missing.

Leave unavailable scores blank. Provide a sidecar `data/annotations.json`:

```json
{
  "conservation_direction": "higher_is_more_conserved",
  "method": "Actual search, alignment, scoring, SASA and residue mapping methods",
  "source": "Database versions and source structure/MSA paths",
  "date": "YYYY-MM-DD"
}
```

No example scores are supplied: synthetic values must not be confused with actual
biological annotations. Duplicate residue keys, sequence mismatches and invalid
fractions are rejected. Current analysis reports per-protein Spearman correlations,
annotation coverage, and interface enrichment after an AA + relative-SASA-bin
permutation control. It does not fit an MSA, claim a hotspot, or infer an experimental
mutation effect. Small strata may make the joint control uninformative; its movable
fraction is reported. MSA quality must be inspected before interpreting conservation
correlations; supplying an annotation alone does not establish its reliability.

The completed run records all quality rules in the annotation sidecar and
`clip_gate/annotations.py`. Observed sequence positions are mapped to the original
chains using the read-only component catalog, then to the ConSurf reference by
unique alignment. The original pair-name order need not match prepared A/B order.
ConSurf uncertainty flags, MSA effective depth and coverage are checked before
acceptance. RSA is calculated from an isolated chain in its observed bound
conformation; positions lacking any expected heavy atom are excluded. Unclipped
RSA, coordinate residue numbers, mapping details and exclusion reasons are saved
in `data/annotation_details.parquet` and the per-chain caches. These annotation
quality rules do not modify training membership, the frozen homology filter or
the original Cβ contact labels.
