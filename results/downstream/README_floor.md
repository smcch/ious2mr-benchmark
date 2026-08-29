
## Holm correction and the FLAIR significance claim

The configuration-versus-real Wilcoxon tests are run once per configuration, so they form two
families: T2w lesion (48 tests) and FLAIR lesion (24 tests). Holm-corrected within each family
at alpha = 0.05:

  Seg-T2   lesion: 48/48 remain significant (max adjusted p = 1.1e-4)
  Seg-FLAIR lesion: 10/24 remain significant (max adjusted p = 0.113)

The manuscript's earlier claim that every FLAIR configuration is significantly below real
FLAIR held only uncorrected; it is now reported as 10 of 24.
