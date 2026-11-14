# hippylibx (bundled copy)

A copy of hIPPYlib-X, the DOLFINx port of hIPPYlib (https://hippylib.github.io), bundled so that
the pipeline runs without a separate install. The copy was taken from hIPPYlib-X commit `ade3bb8`
(January 2026), with the test suite removed, and may differ from the released versions of
hIPPYlib-X. It is distributed under the GNU General Public License version 2 (`LICENSE`), with
the copyright notices in `COPYRIGHT`; any changes made to these files are recorded in the git
history of this repository.

The pipeline uses `BiLaplacianPrior` and `Random` (Gaussian random field samples),
`MultiVector` and `doublePassG` (randomized generalized eigensolver of the stability test), and
`PETScLUSolver`.
