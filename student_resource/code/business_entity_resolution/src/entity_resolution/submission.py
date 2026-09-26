"""Build <team_name>_submission.zip in the layout the challenge requires.

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       ├── requirements.txt      # pinned to the versions installed here
    │       └── run_full_test.ipynb
    └── Documentation_template.md
"""
import os
import re
import subprocess
import sys
import zipfile
from importlib import metadata

from .config import BASE_DIR, OUTPUT_DIR, SRC_DIR

CODE_DIR = os.path.dirname(SRC_DIR)                    # code/business_entity_resolution
DOC_PATH = os.path.join(BASE_DIR, "Documentation_template.md")
PLACEHOLDER_TEAM_NAMES = {"", "your_team_name", "team_name", "<team_name>"}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def pinned_requirements(path):
    """requirements.txt with each package pinned (==) to its installed version.

    Lines for packages that are not installed are kept unchanged.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*([<>=!~].*)?$", line)
            if not m or line.lstrip().startswith("#"):
                out.append(line)
                continue
            name, extras = m.group(1), m.group(2) or ""
            try:
                out.append(f"{name}{extras}=={metadata.version(name)}")
            except metadata.PackageNotFoundError:
                out.append(line)
    return "\n".join(out) + "\n"


def validate_outputs():
    """Run the challenge validator on output/; raise unless it prints PASS."""
    validator = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    result = subprocess.run(
        [sys.executable, validator,
         "--matching", os.path.join(OUTPUT_DIR, "matching_results.tsv"),
         "--candidate", os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
         "--test-dir", os.path.join(BASE_DIR, "dataset", "test")],
        cwd=BASE_DIR, capture_output=True, text=True,
    )
    print(result.stdout, result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Validator did not PASS — fix output/ before zipping.")


def build_submission_zip(team_name, validate=True):
    """Write <BASE_DIR>/<team_name>_submission.zip and return its path."""
    team_name = (team_name or "").strip()
    if team_name.lower() in PLACEHOLDER_TEAM_NAMES:
        raise ValueError("Set TEAM_NAME to your registered team name before building the zip.")

    outputs = [os.path.join(OUTPUT_DIR, n)
               for n in ("matching_results.tsv", "candidate_pairs.tsv")]
    missing = [p for p in outputs + [DOC_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing files for the submission: {missing}")
    if validate:
        validate_outputs()

    zip_path = os.path.join(BASE_DIR, f"{team_name}_submission.zip")
    code_arc = "code/business_entity_resolution"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in outputs:
            zf.write(p, f"output/{os.path.basename(p)}")

        for root, dirs, files in os.walk(SRC_DIR):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, SRC_DIR).replace(os.sep, "/")
                zf.write(full, f"{code_arc}/src/{rel}")

        zf.write(os.path.join(CODE_DIR, "README.md"), f"{code_arc}/README.md")
        notebook = os.path.join(CODE_DIR, "run_full_test.ipynb")
        if os.path.exists(notebook):
            zf.write(notebook, f"{code_arc}/run_full_test.ipynb")
        zf.writestr(f"{code_arc}/requirements.txt",
                    pinned_requirements(os.path.join(CODE_DIR, "requirements.txt")))

        zf.write(DOC_PATH, "Documentation_template.md")

    print(f"[Zip] {zip_path} ({os.path.getsize(zip_path) / 2**20:,.1f} MB)")
    return zip_path
