# Legacy paper-reproduction evidence

These small files were copied from the validated legacy run stored at:

```text
/Volumes/T7/ProjectData/Deepship/runs/legacy/deepship_macnna_paper
```

They report `95.15%` segment-level test accuracy and macro F1 `0.95157` on the
paper-style balanced split. The split samples segments rather than recordings,
so segments from one original recording can occur in multiple partitions. This
is a paper-reproduction result, not a recording-disjoint generalization claim.

The stored run used the earlier `branch_channels=88` configuration. The current
working tree contains later architecture and learning-rate-schedule changes, so
these metrics must not be presented as measurements of the current code until
that code is retrained and evaluated.

The current 532,166-parameter V2 code has since been trained separately under
the three frozen protocols for seeds 42 and 43. Its segment-level accuracy is
`97.30 ± 0.21%`; recording-disjoint recording accuracy is
`70.19 ± 4.08%`; vessel-name-disjoint group accuracy is
`53.13 ± 4.42%`. Those runs live under the Git-ignored `runs/` directory and
must not be merged into this legacy evidence folder.
