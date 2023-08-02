import os
import shutil
from pathlib import Path

import pytest

from jetto_mobo.simulation import compress_jetto_dir, create_config, run, run_many


def test_create_config(tmp_path):
    template = Path("tests/data/example_results")
    directory = tmp_path / "config"
    config = create_config(template, directory)
    assert Path(config.exfile) == (tmp_path / "config" / "jetto.ex").resolve()


def test_compress_jetto_dir(tmp_path):
    shutil.copytree("tests/data/example_results", tmp_path / "example_results")
    compress_jetto_dir(tmp_path / "example_results", delete=True)

    files = os.listdir(tmp_path / "example_results")
    assert "jetto_results.tar.bz2" in files

    file_extensions = [os.path.splitext(f)[1] for f in files]
    permitted_file_extensions = [".CDF", ".log", ".bz2"]
    assert all([ext in permitted_file_extensions for ext in file_extensions])


@pytest.mark.jetto
def test_run(tmp_path):
    pass


@pytest.mark.jetto
def test_run_many(tmp_path):
    pass
