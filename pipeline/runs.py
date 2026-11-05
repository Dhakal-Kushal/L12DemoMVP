import hashlib
import re
from datetime import datetime
from pathlib import Path


def new_run_id(pair_id):
    return f"pair{pair_id}_{datetime.now():%Y%m%d_%H%M%S_%f}"


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"pair[0-9]+_[0-9]{8}_[0-9]{6}_[0-9]{6}", run_id):
        raise ValueError("Missing or invalid run_id. Regenerate the Stage 2 export.")
    return run_id


def source_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
