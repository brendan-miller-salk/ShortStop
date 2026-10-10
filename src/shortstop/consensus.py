"""Locate, download and verify the ShortStop 100-model consensus bundle.

The bundle is a GitHub release asset rather than package data, so `pip install` stays small.
It is fetched once, checked against a pinned sha256, and cached in the package directory (or
~/.cache/shortstop if that is not writable). Set SHORTSTOP_CONSENSUS_URL to use a mirror.
"""
import hashlib
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
import urllib.request

CONSENSUS_NAME = "shortstop_consensus_v1.1"
CONSENSUS_URL = ("https://github.com/brendan-miller-salk/ShortStop/releases/download/"
                f"v1.1.0/{CONSENSUS_NAME}.tar.gz")
CONSENSUS_SHA256 = "2c027c8dde89e05dca3795383046b04c0cf16973653f57dbceed99c75e76c7a0"
MANIFEST = "consensus_manifest.csv"
STALE_SECONDS = 6 * 3600  # temporary files older than this are from killed runs

BASE_DIR = pathlib.Path(__file__).resolve().parent
CACHE_ROOTS = [BASE_DIR / "consensus_models", pathlib.Path.home() / ".cache" / "shortstop"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _writable(root):
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(dir=root, prefix=".write_test.")
        os.close(fd)
        os.remove(probe)
        return True
    except OSError:
        return False


def _remove_stale_temps(root):
    """Remove temporary files left in root by runs that were killed (e.g. at a SLURM time limit)."""
    now = time.time()
    for path in root.glob(f".{CONSENSUS_NAME}.*"):
        try:
            if now - path.stat().st_mtime > STALE_SECONDS:  # old enough not to belong to a live run
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except OSError:
            pass


def _download(root):
    """Download, verify and unpack the bundle into root/CONSENSUS_NAME.

    Everything happens under private temporary names in root; the finished folder is moved into
    place in one rename, so an interrupted run never leaves a half-unpacked bundle that looks complete.
    """
    url = os.environ.get("SHORTSTOP_CONSENSUS_URL", CONSENSUS_URL)
    final = root / CONSENSUS_NAME
    if (final / MANIFEST).exists():  # a concurrent run finished first
        return final
    if final.exists():
        raise RuntimeError(f"{final} exists but has no {MANIFEST}, so it is incomplete. "
                           "Delete that folder and run again.")
    _remove_stale_temps(root)
    print(f"⬇️ Downloading the ShortStop consensus models (~36 MB) to {root} ...")
    tarball = staging = None
    try:
        try:
            fd, tarball = tempfile.mkstemp(dir=root, prefix=f".{CONSENSUS_NAME}.", suffix=".tar.gz.part")
            os.close(fd)
            staging = tempfile.mkdtemp(dir=root, prefix=f".{CONSENSUS_NAME}.", suffix=".unpack")
        except OSError as e:
            raise RuntimeError(f"Cannot write to {root} ({e}). Pass --consensus_dir with a writable folder.") from e
        try:
            response = urllib.request.urlopen(url, timeout=60)
        except (OSError, ValueError) as e:  # URLError and timeouts are OSErrors; a malformed URL is a ValueError
            raise RuntimeError(
                f"Could not download the consensus models from {url} ({e}). Download that file on a "
                "machine with internet access, unpack it, and pass --consensus_dir <unpacked folder>; "
                "or run with --no_consensus.") from e
        try:
            with response, open(tarball, "wb") as f:
                shutil.copyfileobj(response, f)
        except OSError as e:
            raise RuntimeError(f"Downloading {url} into {root} failed ({e}). Check free disk space and quota, "
                               "or pass --consensus_dir to use another folder.") from e
        got = _sha256(tarball)
        if got != CONSENSUS_SHA256:
            raise RuntimeError(f"Consensus model download failed sha256 check (expected {CONSENSUS_SHA256}, got {got}).")
        try:
            with tarfile.open(tarball) as tar:
                for m in tar.getmembers():
                    inside = m.name == CONSENSUS_NAME or m.name.startswith(f"{CONSENSUS_NAME}/")
                    if not (m.isdir() or m.isfile()) or not inside or ".." in m.name.split("/"):
                        raise RuntimeError(f"Unexpected entry in consensus model bundle: {m.name}")
                tar.extractall(staging)
            try:
                os.rename(os.path.join(staging, CONSENSUS_NAME), final)
            except OSError:
                if not (final / MANIFEST).exists():  # unless a concurrent run finished first
                    raise
        except (OSError, tarfile.TarError) as e:
            raise RuntimeError(f"Could not unpack the consensus models into {root} ({e}). Check free disk space "
                               "and quota, or pass --consensus_dir to use another folder.") from e
    finally:
        if tarball and os.path.exists(tarball):
            os.remove(tarball)
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
    print("✅ Consensus models downloaded and verified.")
    return final


def ensure_consensus_models(consensus_dir=None):
    """Return the directory holding consensus_manifest.csv, downloading the bundle if needed.

    consensus_dir may point at an extracted bundle, or at a folder to download it into.
    """
    roots = CACHE_ROOTS
    if consensus_dir:
        given = pathlib.Path(consensus_dir).expanduser().resolve()
        if (given / MANIFEST).exists():
            return given
        roots = [given]
    for root in roots:
        if (root / CONSENSUS_NAME / MANIFEST).exists():
            return root / CONSENSUS_NAME
    for root in roots:
        if _writable(root):
            return _download(root)
    raise RuntimeError(f"No writable location for the consensus models (tried {', '.join(map(str, roots))}). "
                       "Pass --consensus_dir, or --no_consensus to skip.")
