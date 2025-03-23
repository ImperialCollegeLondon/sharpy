"""
Avoid git URLs breaking PyPI uploads.
Similar problem to:
https://stackoverflow.com/questions/54887301/how-can-i-use-git-repos-as-dependencies-for-my-pypi-package
"""

from io import StringIO
from typing import TextIO

import setuptools
from packaging.metadata import Metadata
from setuptools._core_metadata import _write_requirements  # type: ignore[import-not-found]
from setuptools.build_meta import *  # noqa: F403


def write_pypi_compatible_requirements(self: Metadata, final_file: TextIO) -> None:
    """Mark requirements with URLs as external."""
    print("abcd")
    initial_file = StringIO()
    _write_requirements(self, initial_file)
    initial_file.seek(0)
    for initial_line in initial_file:
        final_line = initial_line
        metadata = Metadata.from_email(initial_line, validate=False)
        if metadata.requires_dist and metadata.requires_dist[0].url:
            final_line = initial_line.replace('Requires-Dist:', 'Requires-External:')
        final_file.write(final_line)

setuptools._core_metadata._write_requirements = write_pypi_compatible_requirements



from setuptools.build_meta import build_wheel as setuptools_build_wheel
from setuptools.build_meta import build_sdist as setuptools_build_sdist

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    # Custom behavior before building the wheel
    result = setuptools_build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
    # Custom behavior after building the wheel
    return result

def build_sdist(sdist_directory, config_settings=None):
    # Custom behavior before building the sdist
    result = setuptools_build_sdist(sdist_directory, config_settings)
    # Custom behavior after building the sdist
    return result
