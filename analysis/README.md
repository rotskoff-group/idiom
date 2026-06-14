# analysis/ — downstream paper analysis & figures (not shipped)

Manuscript-reproduction code: evaluation (DisProt / CATH / AFDB-pLDDT>70 controls, DeepLoc,
W1 / kappa / motif / disorder), interpretability analyses, and figure scripts. Imports the
`idiom` library; **not** part of the installed wheel (kept repo-only, run via `bash/` scripts).

Reusable interpretability *methods* live in the library (`idiom.sae`); only paper-specific
analyses and figures live here. Heavy deps (foldseek / deeploc / metapredict / PAE tooling)
go behind the `[paper]` optional extra. Populated in P6.
