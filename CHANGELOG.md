# Changelog

## [2.3](https://github.com/imperialcollegelondon/sharpy/tree/2.3) (2024-05-10)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/2.2...2.3)

**Implemented enhancements:**

- Version 2.3 update [\#289](https://github.com/ImperialCollegeLondon/sharpy/pull/289) ([ben-l-p](https://github.com/ben-l-p))
- Update develop branch with main [\#284](https://github.com/ImperialCollegeLondon/sharpy/pull/284) ([ben-l-p](https://github.com/ben-l-p))
- Added pip install \(with docs\) [\#280](https://github.com/ImperialCollegeLondon/sharpy/pull/280) ([ben-l-p](https://github.com/ben-l-p))
- Update beamplot.py to have stride option, consistent with aerogridplot.py [\#279](https://github.com/ImperialCollegeLondon/sharpy/pull/279) ([kccwing](https://github.com/kccwing))

**Fixed bugs:**

- Fix Github Runner Docker build failing [\#285](https://github.com/ImperialCollegeLondon/sharpy/pull/285) ([ben-l-p](https://github.com/ben-l-p))
- Add scipy version info to env yml [\#277](https://github.com/ImperialCollegeLondon/sharpy/pull/277) ([SJ-Innovation](https://github.com/SJ-Innovation))

**Closed issues:**

- Scipy 1.12.0 Incompatible [\#276](https://github.com/ImperialCollegeLondon/sharpy/issues/276)
- BeamLoader postprocessor squishing answers [\#270](https://github.com/ImperialCollegeLondon/sharpy/issues/270)
- Solving Environment gets killed. [\#268](https://github.com/ImperialCollegeLondon/sharpy/issues/268)
- Error when running sharpy unittest: module scipy.sparse.\_sputils not found [\#227](https://github.com/ImperialCollegeLondon/sharpy/issues/227)
- Potential bug in /sharpy/structure/utils/modalutils.py [\#208](https://github.com/ImperialCollegeLondon/sharpy/issues/208)

**Merged pull requests:**

- Added ability to turn aligned grid off [\#288](https://github.com/ImperialCollegeLondon/sharpy/pull/288) ([ben-l-p](https://github.com/ben-l-p))
- Update with main for mamba fixes [\#286](https://github.com/ImperialCollegeLondon/sharpy/pull/286) ([ben-l-p](https://github.com/ben-l-p))
- Correct typos caught by Divya Sanghi [\#283](https://github.com/ImperialCollegeLondon/sharpy/pull/283) ([bbahiam](https://github.com/bbahiam))
- Develop: Update environment.yml to fix scipy version issue [\#282](https://github.com/ImperialCollegeLondon/sharpy/pull/282) ([kccwing](https://github.com/kccwing))
- Update noaero.py for consistency in function input [\#275](https://github.com/ImperialCollegeLondon/sharpy/pull/275) ([kccwing](https://github.com/kccwing))
- A few minor bug fixes [\#273](https://github.com/ImperialCollegeLondon/sharpy/pull/273) ([sduess](https://github.com/sduess))
- Update XBeam version to include compiler optimisation [\#272](https://github.com/ImperialCollegeLondon/sharpy/pull/272) ([ben-l-p](https://github.com/ben-l-p))
- Update XBeam version to include compiler optimisation [\#271](https://github.com/ImperialCollegeLondon/sharpy/pull/271) ([ben-l-p](https://github.com/ben-l-p))
- Improve docs and code of newmark\_ss [\#267](https://github.com/ImperialCollegeLondon/sharpy/pull/267) ([bbahiam](https://github.com/bbahiam))
- Changed Github runner from Conda to Mamba [\#266](https://github.com/ImperialCollegeLondon/sharpy/pull/266) ([ben-l-p](https://github.com/ben-l-p))
- Changed Github runner from Conda to Mamba [\#265](https://github.com/ImperialCollegeLondon/sharpy/pull/265) ([ben-l-p](https://github.com/ben-l-p))
- Hotfix for documentation search [\#264](https://github.com/ImperialCollegeLondon/sharpy/pull/264) ([kccwing](https://github.com/kccwing))
- Hotfix for documentation - develop [\#263](https://github.com/ImperialCollegeLondon/sharpy/pull/263) ([kccwing](https://github.com/kccwing))
- Hotfix for documentation - main [\#262](https://github.com/ImperialCollegeLondon/sharpy/pull/262) ([kccwing](https://github.com/kccwing))
- Merging v2.2 into develop [\#261](https://github.com/ImperialCollegeLondon/sharpy/pull/261) ([kccwing](https://github.com/kccwing))


## [2.2](https://github.com/imperialcollegelondon/sharpy/tree/2.2) (2023-10-18)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/2.1...2.2)

**Implemented enhancements:**

- Added environment for Apple Silicon \(ARM64\) [\#254](https://github.com/ImperialCollegeLondon/sharpy/pull/254) ([ben-l-p](https://github.com/ben-l-p))
- New Fuselage Model plus Minor Improvements [\#249](https://github.com/ImperialCollegeLondon/sharpy/pull/249) ([sduess](https://github.com/sduess))

**Fixed bugs:**

- fix polars concatenation in assembly of aeroinformation - develop [\#253](https://github.com/ImperialCollegeLondon/sharpy/pull/253) ([kccwing](https://github.com/kccwing))
- fix polars concatenation in assembly of aeroinformation - main [\#252](https://github.com/ImperialCollegeLondon/sharpy/pull/252) ([kccwing](https://github.com/kccwing))
- Minor bug fixes in aero util functions and save data postprocessor [\#247](https://github.com/ImperialCollegeLondon/sharpy/pull/247) ([sduess](https://github.com/sduess))

**Closed issues:**

- Automated tests failed : UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 47: ordinal not in range\(128\) [\#245](https://github.com/ImperialCollegeLondon/sharpy/issues/245)
- Wrong key in settings for /cases/templates/flying\_wings.py [\#205](https://github.com/ImperialCollegeLondon/sharpy/issues/205)

**Merged pull requests:**

- merging develop into main for v2.2 [\#259](https://github.com/ImperialCollegeLondon/sharpy/pull/259) ([kccwing](https://github.com/kccwing))
- fix \[docker\] use correct environment name in docker bashrc [\#257](https://github.com/ImperialCollegeLondon/sharpy/pull/257) ([sduess](https://github.com/sduess))
- Update docker environment  [\#255](https://github.com/ImperialCollegeLondon/sharpy/pull/255) ([sduess](https://github.com/sduess))
- Update README.md [\#246](https://github.com/ImperialCollegeLondon/sharpy/pull/246) ([rafapalacios](https://github.com/rafapalacios))
- bringing commits to main into develop [\#244](https://github.com/ImperialCollegeLondon/sharpy/pull/244) ([rafapalacios](https://github.com/rafapalacios))
- Updates to plotutils.py and the cantilever\_wing demo [\#242](https://github.com/ImperialCollegeLondon/sharpy/pull/242) ([boltyboi](https://github.com/boltyboi))
- Small additions to the installation guide. [\#241](https://github.com/ImperialCollegeLondon/sharpy/pull/241) ([boltyboi](https://github.com/boltyboi))
- Tutorial for closed-Loop Simulation with SHARPy as a hardware-in-the-loop system [\#240](https://github.com/ImperialCollegeLondon/sharpy/pull/240) ([sduess](https://github.com/sduess))

## [2.1](https://github.com/imperialcollegelondon/sharpy/tree/2.1) (2023-05-31)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/2.0...2.1)

**Implemented enhancements:**

- SHARPy Docker now releases to GitHub Packages  [\#218](https://github.com/ImperialCollegeLondon/sharpy/pull/218) ([ngoiz](https://github.com/ngoiz))
- Some Enhancements and Fixes for the Polar Correction, Wake Discretisation, and Gust Velocity Field Generator [\#217](https://github.com/ImperialCollegeLondon/sharpy/pull/217) ([sduess](https://github.com/sduess))
- Collective blade pitch PID control of wind turbines [\#176](https://github.com/ImperialCollegeLondon/sharpy/pull/176) ([ArturoMS13](https://github.com/ArturoMS13))

**Fixed bugs:**

- Handle absolute tolerance for structural convergence tests as input settings and increase default value [\#221](https://github.com/ImperialCollegeLondon/sharpy/pull/221) ([sduess](https://github.com/sduess))
- Fix horseshoe wake handling in UVLM [\#204](https://github.com/ImperialCollegeLondon/sharpy/pull/204) ([sduess](https://github.com/sduess))

**Closed issues:**

- No such file or directory during: source bin/sharpy\_vars.sh [\#233](https://github.com/ImperialCollegeLondon/sharpy/issues/233)
- Twist direction inconsistent [\#212](https://github.com/ImperialCollegeLondon/sharpy/issues/212)

**Merged pull requests:**

- Version number updates for 2.1 from develop [\#237](https://github.com/ImperialCollegeLondon/sharpy/pull/237) ([kccwing](https://github.com/kccwing))
- Merge to main for version 2.0.1 release [\#236](https://github.com/ImperialCollegeLondon/sharpy/pull/236) ([kccwing](https://github.com/kccwing))
- Updates to Goland wing example [\#234](https://github.com/ImperialCollegeLondon/sharpy/pull/234) ([rafapalacios](https://github.com/rafapalacios))
- A bit of clean-up of the readme file of the repo [\#231](https://github.com/ImperialCollegeLondon/sharpy/pull/231) ([rafapalacios](https://github.com/rafapalacios))
- Dev internals [\#223](https://github.com/ImperialCollegeLondon/sharpy/pull/223) ([ACea15](https://github.com/ACea15))
- updating from python 3.7 to 3.10 [\#222](https://github.com/ImperialCollegeLondon/sharpy/pull/222) ([kccwing](https://github.com/kccwing))
- Fix typo in setting [\#206](https://github.com/ImperialCollegeLondon/sharpy/pull/206) ([sduess](https://github.com/sduess))

## [2.0](https://github.com/imperialcollegelondon/sharpy/tree/2.0) (2022-07-04)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/1.3...2.0)

**Implemented enhancements:**

- Plot the aeroelastic mode shape to Paraview [\#202](https://github.com/ImperialCollegeLondon/sharpy/pull/202) ([ngoiz](https://github.com/ngoiz))
- Enhanced linear solver [\#196](https://github.com/ImperialCollegeLondon/sharpy/pull/196) ([ngoiz](https://github.com/ngoiz))
- Add pip support [\#175](https://github.com/ImperialCollegeLondon/sharpy/pull/175) ([DavidAnderegg](https://github.com/DavidAnderegg))

**Fixed bugs:**

- Flap inputs in state-space model not working in certain wing configurations [\#192](https://github.com/ImperialCollegeLondon/sharpy/issues/192)
- Fix mass matrix generation for lumped masses in case of several masses per node [\#194](https://github.com/ImperialCollegeLondon/sharpy/pull/194) ([sduess](https://github.com/sduess))
- Fix the sparse matrix in balancing ROM [\#186](https://github.com/ImperialCollegeLondon/sharpy/pull/186) ([AntonioWR](https://github.com/AntonioWR))

**Closed issues:**

- scipy used for direct balancing method [\#184](https://github.com/ImperialCollegeLondon/sharpy/issues/184)
- Potential Issue in the balancing direct method [\#183](https://github.com/ImperialCollegeLondon/sharpy/issues/183)
- Why no pip support? [\#171](https://github.com/ImperialCollegeLondon/sharpy/issues/171)

**Merged pull requests:**

- Contain write operations within with statements [\#195](https://github.com/ImperialCollegeLondon/sharpy/pull/195) ([ngoiz](https://github.com/ngoiz))
- Support loading/saving state-spaces and gains to h5 files [\#188](https://github.com/ImperialCollegeLondon/sharpy/pull/188) ([ngoiz](https://github.com/ngoiz))
- Update installation docs [\#187](https://github.com/ImperialCollegeLondon/sharpy/pull/187) ([nacho-carnicero](https://github.com/nacho-carnicero))
- Unittest for nonlinear dynamic solver [\#185](https://github.com/ImperialCollegeLondon/sharpy/pull/185) ([sduess](https://github.com/sduess))
- Change in the io-Settings to add thrust. [\#164](https://github.com/ImperialCollegeLondon/sharpy/pull/164) ([Eriklyy](https://github.com/Eriklyy))
- UDP-Inout change for multiply cs\_surface\_deflections and loads/strain [\#162](https://github.com/ImperialCollegeLondon/sharpy/pull/162) ([Eriklyy](https://github.com/Eriklyy))
- Update AsymptoticStability and FrequencyResponse post-processors [\#103](https://github.com/ImperialCollegeLondon/sharpy/pull/103) ([ngoiz](https://github.com/ngoiz))

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
- Enhanced Frequency Response post-processor and linear system input/output options [\#83](https://github.com/ImperialCollegeLondon/sharpy/pull/83) ([ngoiz](https://github.com/ngoiz))
- Pazi wing added to flying\_wing template [\#82](https://github.com/ImperialCollegeLondon/sharpy/pull/82) ([outoforderdev](https://github.com/outoforderdev))
- Several new aerodynamic enhancements [\#79](https://github.com/ImperialCollegeLondon/sharpy/pull/79) ([ArturoMS13](https://github.com/ArturoMS13))

**Fixed bugs:**

- libss.py disc2cont doesn't accept SISO systems [\#88](https://github.com/ImperialCollegeLondon/sharpy/issues/88)
- Dimension mismatch when assembling linear UVLM with "shortened" wake [\#71](https://github.com/ImperialCollegeLondon/sharpy/issues/71)
- Fix bug wake shape generator StaticCoupled [\#85](https://github.com/ImperialCollegeLondon/sharpy/pull/85) ([ArturoMS13](https://github.com/ArturoMS13))
- Rework of direct balancing [\#74](https://github.com/ImperialCollegeLondon/sharpy/pull/74) ([outoforderdev](https://github.com/outoforderdev))

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
- Pazy wing modified to include the tip weight [\#87](https://github.com/ImperialCollegeLondon/sharpy/pull/87) ([outoforderdev](https://github.com/outoforderdev))
- Minor output clean up [\#81](https://github.com/ImperialCollegeLondon/sharpy/pull/81) ([ngoiz](https://github.com/ngoiz))
- Fixes assembly of linUVLM after plotting wake with minus m\_star  [\#72](https://github.com/ImperialCollegeLondon/sharpy/pull/72) ([ngoiz](https://github.com/ngoiz))

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

## [v1.1.0-2](https://github.com/imperialcollegelondon/sharpy/tree/v1.1.0-2) (2019-12-12)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.1.0...v1.1.0-2)

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

## [1.0.0](https://github.com/imperialcollegelondon/sharpy/tree/1.0.0) (2019-11-07)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v1.0.0-rc...1.0.0)

**Implemented enhancements:**

- WriteVariablesTime output global beam variables and consistent out dir [\#19](https://github.com/ImperialCollegeLondon/sharpy/pull/19) ([ngoiz](https://github.com/ngoiz))
- Autodocumenter [\#16](https://github.com/ImperialCollegeLondon/sharpy/pull/16) ([ngoiz](https://github.com/ngoiz))

**Closed issues:**

- Tests not passing due to them being outdated + test refactoring. [\#11](https://github.com/ImperialCollegeLondon/sharpy/issues/11)

**Merged pull requests:**

- Release of v1.0.0!!! [\#20](https://github.com/ImperialCollegeLondon/sharpy/pull/20) ([fonsocarre](https://github.com/fonsocarre))
- Documentation fixes/updates [\#18](https://github.com/ImperialCollegeLondon/sharpy/pull/18) ([ngoiz](https://github.com/ngoiz))
- Fix dynamic control surface and settings for aerogridloader [\#15](https://github.com/ImperialCollegeLondon/sharpy/pull/15) ([ngoiz](https://github.com/ngoiz))

## [v1.0.0-rc](https://github.com/imperialcollegelondon/sharpy/tree/v1.0.0-rc) (2019-08-22)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/V0.2.1...v1.0.0-rc)

**Closed issues:**

- Output table [\#10](https://github.com/ImperialCollegeLondon/sharpy/issues/10)

**Merged pull requests:**

- Remove H5pyDeprecationWarning [\#14](https://github.com/ImperialCollegeLondon/sharpy/pull/14) ([ArturoMS13](https://github.com/ArturoMS13))
- Lagrange multipliers for Catapult Take Off works + clean tests [\#13](https://github.com/ImperialCollegeLondon/sharpy/pull/13) ([fonsocarre](https://github.com/fonsocarre))

## [V0.2.1](https://github.com/imperialcollegelondon/sharpy/tree/V0.2.1) (2019-03-14)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v0.2...V0.2.1)

## [v0.2](https://github.com/imperialcollegelondon/sharpy/tree/v0.2) (2019-03-14)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/v0.1...v0.2)

**Closed issues:**

- Add recovery options [\#9](https://github.com/ImperialCollegeLondon/sharpy/issues/9)

## [v0.1](https://github.com/imperialcollegelondon/sharpy/tree/v0.1) (2018-09-03)

[Full Changelog](https://github.com/imperialcollegelondon/sharpy/compare/bd2b65974d57d2d6486ea90cdb68ef6324efbac8...v0.1)

**Implemented enhancements:**

- Hinge definition for control surface [\#8](https://github.com/ImperialCollegeLondon/sharpy/issues/8)
- sharpy\_main.main does not return output [\#5](https://github.com/ImperialCollegeLondon/sharpy/issues/5)

**Fixed bugs:**

- Aerofoil data associated to the nodes instead of the elements [\#6](https://github.com/ImperialCollegeLondon/sharpy/issues/6)

**Merged pull requests:**

- Trimming routine working [\#4](https://github.com/ImperialCollegeLondon/sharpy/pull/4) ([fonsocarre](https://github.com/fonsocarre))
- Feature coupled dynamic [\#3](https://github.com/ImperialCollegeLondon/sharpy/pull/3) ([fonsocarre](https://github.com/fonsocarre))
- Refactored storage finished [\#2](https://github.com/ImperialCollegeLondon/sharpy/pull/2) ([fonsocarre](https://github.com/fonsocarre))
- Settings files are ConfigObjs now, not ConfigParser anymore [\#1](https://github.com/ImperialCollegeLondon/sharpy/pull/1) ([fonsocarre](https://github.com/fonsocarre))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
