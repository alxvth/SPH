# Spatial Hierarchy Library Evaluations

You can re-use `sph-cache` in the eval (with some extra steps):
- set the `externalCachePath` to your local path and set cs.cacheActive `true`
- update `ihs.minNumComp`, `ihs.maxLevels`, which might be different in the eval script form your compute setting
