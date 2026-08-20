"""The app has to be importable from somewhere other than the repo root.

Everything used to resolve against the working directory: "./frontend" for the
static mount, "./data/raw" for uploads, "./data/models/model.joblib" for the
model, "sqlite:///./drift_radar.db" for the database. StaticFiles raises at
import time if its directory is missing, so `import app.main` from any other
directory ended in RuntimeError. A worker started elsewhere would have written
its uploads and its model where the API was not looking.

Nothing else in the suite can catch this, because pytest runs from the root,
where a relative path happens to resolve.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_from(cwd, code):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )


def test_the_app_imports_from_an_unrelated_directory(tmp_path):
    result = run_from(tmp_path, "import app.main; print('ok')")

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_paths_do_not_depend_on_the_working_directory(tmp_path):
    code = (
        "from app.config import settings;"
        "print('frontend_dir', settings.frontend_dir);"
        "print('data_dir', settings.data_dir);"
        "print('artifact_path', settings.artifact_path);"
        "print('database_url', settings.database_url)"
    )
    here = run_from(REPO_ROOT, code)
    there = run_from(tmp_path, code)

    assert here.returncode == 0, here.stderr
    assert there.returncode == 0, there.stderr
    assert here.stdout == there.stdout

    for line in there.stdout.splitlines():
        name, value = line.split(" ", 1)
        # The database is a URL, not a bare path, and "sqlite:///./drift_radar.db"
        # is relative while still not starting with a dot. Strip the scheme
        # before deciding, or this check passes on exactly the bug it is for.
        path = value.split(":///", 1)[1] if ":///" in value else value
        assert Path(path).is_absolute(), f"{name}={value} follows the working directory"
