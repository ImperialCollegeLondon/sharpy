# SHARPy GitHub Workflows

There are 4 automated workflows for SHARPy's CI/CD.

## SHARPy Tests

The related to the SHARPy tests that run the `SHARPy Tests` job are:

    * `sharpy_tests.yaml`: when Python or the submodules files are edited
    * `sharpy_no_test_needed.yaml`: otherwise

This avoids running the full set of tests for changes in the documentation etc.
Since the merge to `develop` and `main` is protected by the tests passing, the 
second workflow ensures a positive result for those PRs that did not change the 
Python code, hence allowing the merge.

## Docker

Two nearly identical workflows, the only difference is that one pushes the Docker 
image to the SHARPy packages. Therefore:

    * `docker_build.yaml`: Builds the Docker image but does not push. Runs on changes to the `docker*` workflows, changes to the `utils/` directory (environments) and changes to the `Dockerfile`. Required test for PRs to merge to `develop` and `main`.
    * `docker_build.yaml`: Builds and pushes the Docker image. Runs on pushes to `develop`, `main` and annotated tags.
