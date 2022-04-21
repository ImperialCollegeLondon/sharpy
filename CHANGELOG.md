# Changelog

## [1.3](https://github.com/imperialcollegelondon/sharpy/tree/1.3) (2021-11-11)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.2.1...1.3)

**Implemented enhancements:**

- Include gravity direction as input for structural solvers [\#112](https://github.com/ImperialCollegeLondon/sharpy/issues/112)
- Simulation settings check - Unrecognised settings raise Error [\#148](https://github.com/ImperialCollegeLondon/sharpy/pull/148) ([ngoiz](https://github.com/ngoiz))
- Aerodynamic forces correction enhancements [\#140](https://github.com/ImperialCollegeLondon/sharpy/pull/140) ([ngoiz](https://github.com/ngoiz))
- New feature: apply external forces to the system at runtime [\#125](https://github.com/ImperialCollegeLondon/sharpy/pull/125) ([ArturoMS13](https://github.com/ArturoMS13))
- Lift distribution post-processor [\#121](https://github.com/ImperialCollegeLondon/sharpy/pull/121) ([sduess](https://github.com/sduess))

**Fixed bugs:**

- Fix bug in computing total moments in A frame [\#177](https://github.com/ImperialCollegeLondon/sharpy/pull/177) ([ngoiz](https://github.com/ngoiz))

**Closed issues:**

- Unrecognised model in goland test case [\#143](https://github.com/ImperialCollegeLondon/sharpy/issues/143)

**Merged pull requests:**

- Implement GitHub Actions as testing suite provider [\#179](https://github.com/ImperialCollegeLondon/sharpy/pull/179) ([ngoiz](https://github.com/ngoiz))
- Update submodules and conda environments [\#161](https://github.com/ImperialCollegeLondon/sharpy/pull/161) ([sduess](https://github.com/sduess))
- Support element variables in UDP output [\#160](https://github.com/ImperialCollegeLondon/sharpy/pull/160) ([ngoiz](https://github.com/ngoiz))
- Output directory in the location specified in settings\[SHARPy\]\[log\_folder\] [\#130](https://github.com/ImperialCollegeLondon/sharpy/pull/130) ([ngoiz](https://github.com/ngoiz))
- Update and include features in multibody [\#126](https://github.com/ImperialCollegeLondon/sharpy/pull/126) ([ArturoMS13](https://github.com/ArturoMS13))
- Update linear UVLM to account for CFL not equal to one in the wake convection [\#124](https://github.com/ImperialCollegeLondon/sharpy/pull/124) ([ArturoMS13](https://github.com/ArturoMS13))
- Minor changes [\#123](https://github.com/ImperialCollegeLondon/sharpy/pull/123) ([ArturoMS13](https://github.com/ArturoMS13))

## [v1.2.1](https://github.com/imperialcollegelondon/sharpy/tree/v1.2.1) (2021-02-09)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.2...v1.2.1)

**Implemented enhancements:**

- Allow CFL != 1in shedding step [\#78](https://github.com/ImperialCollegeLondon/sharpy/issues/78)
- Vortex wake managed by SHARPy [\#77](https://github.com/ImperialCollegeLondon/sharpy/issues/77)
- Recover vortex core in UVLM [\#76](https://github.com/ImperialCollegeLondon/sharpy/issues/76)
- Include viscous drag force from airfoil properties [\#75](https://github.com/ImperialCollegeLondon/sharpy/issues/75)

**Fixed bugs:**

- Bug in beamstructure.py [\#117](https://github.com/ImperialCollegeLondon/sharpy/issues/117)
- Definition of control surfaces and impact of node ordering in mirrored wings [\#43](https://github.com/ImperialCollegeLondon/sharpy/issues/43)

**Closed issues:**

- examples refer to non-existent solver SHWUvlm [\#119](https://github.com/ImperialCollegeLondon/sharpy/issues/119)
- Potential bug in xbeam and cbeam interfaces [\#89](https://github.com/ImperialCollegeLondon/sharpy/issues/89)
- Update packages producing deprecation warnings and tackle other warnings [\#80](https://github.com/ImperialCollegeLondon/sharpy/issues/80)

**Merged pull requests:**

- Rigid coupled solver [\#120](https://github.com/ImperialCollegeLondon/sharpy/pull/120) ([ArturoMS13](https://github.com/ArturoMS13))
- Support to save ROM projector matrices [\#118](https://github.com/ImperialCollegeLondon/sharpy/pull/118) ([ngoiz](https://github.com/ngoiz))

## [v1.2](https://github.com/imperialcollegelondon/sharpy/tree/v1.2) (2020-12-03)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.1.1...v1.2)

**Implemented enhancements:**

- Quasi-steady solver in stepuvlm [\#114](https://github.com/ImperialCollegeLondon/sharpy/pull/114) ([ArturoMS13](https://github.com/ArturoMS13))
- Generator to modify structural mass at runtime [\#107](https://github.com/ImperialCollegeLondon/sharpy/pull/107) ([ngoiz](https://github.com/ngoiz))
- Sends simulation info to xplane via udp [\#104](https://github.com/ImperialCollegeLondon/sharpy/pull/104) ([wong-hl](https://github.com/wong-hl))
- Major new aerodynamic features [\#101](https://github.com/ImperialCollegeLondon/sharpy/pull/101) ([ArturoMS13](https://github.com/ArturoMS13))
- Simple post-processor to save simulation parameters for batch runs [\#91](https://github.com/ImperialCollegeLondon/sharpy/pull/91) ([ngoiz](https://github.com/ngoiz))
- SHARPy support for external inputs via UDP network [\#90](https://github.com/ImperialCollegeLondon/sharpy/pull/90) ([ngoiz](https://github.com/ngoiz))
- Vortex radius as input parameter [\#86](https://github.com/ImperialCollegeLondon/sharpy/pull/86) ([ArturoMS13](https://github.com/ArturoMS13))

**Fixed bugs:**

- libss.py disc2cont doesn't accept SISO systems [\#88](https://github.com/ImperialCollegeLondon/sharpy/issues/88)
- Dimension mismatch when assembling linear UVLM with "shortened" wake [\#71](https://github.com/ImperialCollegeLondon/sharpy/issues/71)

**Closed issues:**

- \[develop\] Documentation for postprocs not rendering in RTD [\#109](https://github.com/ImperialCollegeLondon/sharpy/issues/109)
- Numerical error in the linear UVLM output matrices when vortex radius is too small [\#105](https://github.com/ImperialCollegeLondon/sharpy/issues/105)
- fatal error: mkl.h: No such file or directory [\#70](https://github.com/ImperialCollegeLondon/sharpy/issues/70)
- lingebm.py -\> update\_matrices\_time\_scale [\#69](https://github.com/ImperialCollegeLondon/sharpy/issues/69)
- Missing node connectivities figure in case files documentation [\#68](https://github.com/ImperialCollegeLondon/sharpy/issues/68)

**Merged pull requests:**

- \[fix\] Fixed bug in Gridbox class [\#111](https://github.com/ImperialCollegeLondon/sharpy/pull/111) ([sduess](https://github.com/sduess))
- Include linear UVLM stiffening and damping terms in the UVLM D matrix [\#108](https://github.com/ImperialCollegeLondon/sharpy/pull/108) ([ngoiz](https://github.com/ngoiz))
- Fix accuracy problem in UVLMLin [\#106](https://github.com/ImperialCollegeLondon/sharpy/pull/106) ([ArturoMS13](https://github.com/ArturoMS13))
- Minor improvements [\#102](https://github.com/ImperialCollegeLondon/sharpy/pull/102) ([ngoiz](https://github.com/ngoiz))
- Linearisation of externally applied follower forces [\#100](https://github.com/ImperialCollegeLondon/sharpy/pull/100) ([ngoiz](https://github.com/ngoiz))
- Updated docs for DataStructures and Multibody [\#99](https://github.com/ImperialCollegeLondon/sharpy/pull/99) ([ArturoMS13](https://github.com/ArturoMS13))
- Support linearised all-moving control surfaces [\#97](https://github.com/ImperialCollegeLondon/sharpy/pull/97) ([ngoiz](https://github.com/ngoiz))
- Update Linux and minimal environments [\#96](https://github.com/ImperialCollegeLondon/sharpy/pull/96) ([ngoiz](https://github.com/ngoiz))
- New approach to multibody computations [\#95](https://github.com/ImperialCollegeLondon/sharpy/pull/95) ([ArturoMS13](https://github.com/ArturoMS13))
- New SHARPy examples in the documentation [\#94](https://github.com/ImperialCollegeLondon/sharpy/pull/94) ([ArturoMS13](https://github.com/ArturoMS13))
- Add support for offline use of UDPout postproc [\#93](https://github.com/ImperialCollegeLondon/sharpy/pull/93) ([ngoiz](https://github.com/ngoiz))
- Option to transform rigid modes given at A FoR to centre of gravity and aligned with principal axes of inertia [\#92](https://github.com/ImperialCollegeLondon/sharpy/pull/92) ([ngoiz](https://github.com/ngoiz))

## [v1.1.1](https://github.com/imperialcollegelondon/sharpy/tree/v1.1.1) (2020-02-03)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.1.0-2...v1.1.1)

**Implemented enhancements:**

- User-defined aerodynamic airfoil efficiency factor and constant force terms [\#59](https://github.com/ImperialCollegeLondon/sharpy/pull/59) ([ngoiz](https://github.com/ngoiz))

**Closed issues:**

- Broken link on SHARPy Installation Guide [\#67](https://github.com/ImperialCollegeLondon/sharpy/issues/67)
- Update citation instructions [\#62](https://github.com/ImperialCollegeLondon/sharpy/issues/62)
- Incorrect version tag displayed when running SHARPy [\#61](https://github.com/ImperialCollegeLondon/sharpy/issues/61)
- Clean up SHARPy linear interface with UVLM [\#48](https://github.com/ImperialCollegeLondon/sharpy/issues/48)

**Merged pull requests:**

- Documentation Improvements [\#66](https://github.com/ImperialCollegeLondon/sharpy/pull/66) ([ngoiz](https://github.com/ngoiz))
- Minor fixes and general code clean up of linear modules [\#65](https://github.com/ImperialCollegeLondon/sharpy/pull/65) ([ngoiz](https://github.com/ngoiz))
- Error log file created when program encounters exceptions [\#64](https://github.com/ImperialCollegeLondon/sharpy/pull/64) ([ngoiz](https://github.com/ngoiz))
- Update README.md [\#63](https://github.com/ImperialCollegeLondon/sharpy/pull/63) ([rafapalacios](https://github.com/rafapalacios))
- Clean up linear SHARPy's interface with UVLM [\#60](https://github.com/ImperialCollegeLondon/sharpy/pull/60) ([ngoiz](https://github.com/ngoiz))

## [v1.1.0](https://github.com/imperialcollegelondon/sharpy/tree/v1.1.0) (2019-12-12)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.0.1...v1.1.0)

**Implemented enhancements:**

- Improvements to Model Reduction [\#56](https://github.com/ImperialCollegeLondon/sharpy/pull/56) ([ngoiz](https://github.com/ngoiz))
- Submodules and cmake build tools instead of Makefiles [\#52](https://github.com/ImperialCollegeLondon/sharpy/pull/52) ([fonsocarre](https://github.com/fonsocarre))
- New settings check against valid options [\#39](https://github.com/ImperialCollegeLondon/sharpy/pull/39) ([ngoiz](https://github.com/ngoiz))
- Default settings get type specified for the setting rather than their own type [\#34](https://github.com/ImperialCollegeLondon/sharpy/pull/34) ([ngoiz](https://github.com/ngoiz))

**Fixed bugs:**

- Default value not correct in documentation when it is a numpy type. [\#32](https://github.com/ImperialCollegeLondon/sharpy/issues/32)
- Documentation for Postprocessors being skipped by Sphinx in RTD [\#30](https://github.com/ImperialCollegeLondon/sharpy/issues/30)

**Closed issues:**

- Minor documentation issues [\#53](https://github.com/ImperialCollegeLondon/sharpy/issues/53)
- WindTurbine case generation script does not produce sharpy file [\#50](https://github.com/ImperialCollegeLondon/sharpy/issues/50)
- Considerations for building SHARPy [\#47](https://github.com/ImperialCollegeLondon/sharpy/issues/47)
- Installation fails on macOS with Intel compiler [\#46](https://github.com/ImperialCollegeLondon/sharpy/issues/46)
- run\_theo\_freq.py fails in Docker container [\#37](https://github.com/ImperialCollegeLondon/sharpy/issues/37)
- Compare to other competing software in JOSS paper [\#36](https://github.com/ImperialCollegeLondon/sharpy/issues/36)

**Merged pull requests:**

- Example wind turbine [\#58](https://github.com/ImperialCollegeLondon/sharpy/pull/58) ([ArturoMS13](https://github.com/ArturoMS13))
- Small typo in README.md and updates to it [\#57](https://github.com/ImperialCollegeLondon/sharpy/pull/57) ([fonsocarre](https://github.com/fonsocarre))
- Restructuring of A Short Guide to SHARPy [\#55](https://github.com/ImperialCollegeLondon/sharpy/pull/55) ([ngoiz](https://github.com/ngoiz))
- JOSS Paper Minor typos fixed [\#54](https://github.com/ImperialCollegeLondon/sharpy/pull/54) ([ngoiz](https://github.com/ngoiz))
- Update .solver.txt extension to .sharpy [\#51](https://github.com/ImperialCollegeLondon/sharpy/pull/51) ([ArturoMS13](https://github.com/ArturoMS13))
- Fix typo in unittests using tearDowns instead of tearDown [\#49](https://github.com/ImperialCollegeLondon/sharpy/pull/49) ([ngoiz](https://github.com/ngoiz))
- Bug fixes in installation docs [\#45](https://github.com/ImperialCollegeLondon/sharpy/pull/45) ([rafmudaf](https://github.com/rafmudaf))
- Updated installation instructions [\#44](https://github.com/ImperialCollegeLondon/sharpy/pull/44) ([ngoiz](https://github.com/ngoiz))
- Travis CI now uses the minimal environment, the same as the Docker build [\#42](https://github.com/ImperialCollegeLondon/sharpy/pull/42) ([fonsocarre](https://github.com/fonsocarre))
- Remove calls to matplotlib \(or wrap in try except\) [\#41](https://github.com/ImperialCollegeLondon/sharpy/pull/41) ([ngoiz](https://github.com/ngoiz))
- Added information about competing software in JOSS paper [\#40](https://github.com/ImperialCollegeLondon/sharpy/pull/40) ([fonsocarre](https://github.com/fonsocarre))
- Removes deprecated case files from cases folder [\#38](https://github.com/ImperialCollegeLondon/sharpy/pull/38) ([ngoiz](https://github.com/ngoiz))
- Change position of --name argument in docs [\#35](https://github.com/ImperialCollegeLondon/sharpy/pull/35) ([petebachant](https://github.com/petebachant))
- Improvements in documentation  [\#31](https://github.com/ImperialCollegeLondon/sharpy/pull/31) ([ngoiz](https://github.com/ngoiz))

## [v1.0.1](https://github.com/imperialcollegelondon/sharpy/tree/v1.0.1) (2019-11-17)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/1.0.0...v1.0.1)

**Implemented enhancements:**

- New example of linearised flying wing [\#28](https://github.com/ImperialCollegeLondon/sharpy/pull/28) ([ngoiz](https://github.com/ngoiz))
- SHARPy can now be obtained from a Docker Hub container [\#27](https://github.com/ImperialCollegeLondon/sharpy/pull/27) ([fonsocarre](https://github.com/fonsocarre))
- Improved modal solver [\#26](https://github.com/ImperialCollegeLondon/sharpy/pull/26) ([fonsocarre](https://github.com/fonsocarre))
- Updated function calls for latest numpy 1.17 [\#25](https://github.com/ImperialCollegeLondon/sharpy/pull/25) ([ngoiz](https://github.com/ngoiz))
- Examples added to documentation [\#24](https://github.com/ImperialCollegeLondon/sharpy/pull/24) ([fonsocarre](https://github.com/fonsocarre))
- Improved linear solver documentation and minor Krylov ROM fixes [\#23](https://github.com/ImperialCollegeLondon/sharpy/pull/23) ([ngoiz](https://github.com/ngoiz))
- change log generator incorporated [\#22](https://github.com/ImperialCollegeLondon/sharpy/pull/22) ([ngoiz](https://github.com/ngoiz))

**Merged pull requests:**

- Version v1.0.1 released [\#29](https://github.com/ImperialCollegeLondon/sharpy/pull/29) ([fonsocarre](https://github.com/fonsocarre))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
