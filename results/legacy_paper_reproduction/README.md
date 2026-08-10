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
