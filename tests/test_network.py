"""The network surface, pinned.

`docs/NETWORK.md` tells a security team which flows to open and promises that a decision needs none
of them. A promise is worth what a test makes of it, so this file checks the shape of the claim:
only the Hugging Face client may reach out, from a known list of modules, and the code that runs per
decision may not reach out at all.

No model, no data, no network: it reads the source with `ast`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("lib", "cli")
DOC = ROOT / "docs" / "NETWORK.md"

#: Names that reach the network, and the modules allowed to use them. Adding a module here is a
#: deliberate act: it widens what an air-gapped install has to account for.
ALLOWED: dict[str, set[str]] = {
    # the Hugging Face Hub client: weights, decision.json, the calibration datasets
    "snapshot_download": {"lib/jul/backbone.py", "lib/jul/presets.py", "cli/jul_cli/setup.py"},
    "hf_hub_download": {"lib/jul/backends/mlx.py", "lib/jul/decision.py"},
    "from_pretrained": {"lib/jul/backends/torch.py"},
    "load_dataset": {"lib/jul/calibrate.py"},
    # cache lookups: same names, but they never leave the machine
    "try_to_load_from_cache": {"lib/jul/backends/mlx.py"},
    "scan_cache_dir": {"cli/jul_cli/main.py"},
}

#: Never acceptable in the library or the CLI: JuL has no server of its own to talk to. Only names
#: that mean the network and nothing else. `get` is a dict method far more often than an HTTP verb,
#: so raw clients are caught by their import below rather than by their call sites.
FORBIDDEN = {"urlopen", "urlretrieve", "socketpair", "create_connection"}
FORBIDDEN_MODULES = {"requests", "httpx", "urllib", "urllib.request", "socket", "http.client", "aiohttp"}

#: The decision path. Nothing reachable from these may download anything.
DECISION_PATH = ("lib/jul/engine.py", "lib/jul/client.py", "lib/jul/context.py", "lib/jul/types.py",
                 "lib/jul/tuning.py", "lib/jul/calibration.py", "lib/jul/synth.py")


def sources() -> list[Path]:
    return sorted(p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py"))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def called_names(tree: ast.AST) -> set[str]:
    """Every name that appears in a call position, attribute calls included."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def imported_modules(tree: ast.AST) -> set[str]:
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


@pytest.fixture(scope="module")
def parsed() -> dict[str, ast.AST]:
    return {relative(p): ast.parse(p.read_text()) for p in sources()}


def test_only_known_modules_download_anything(parsed):
    """A download from a module nobody expected is the surprise an audit is there to find."""
    unexpected = []
    for path, tree in parsed.items():
        for name in called_names(tree) & ALLOWED.keys():
            if path not in ALLOWED[name]:
                unexpected.append(f"{path} calls {name}()")
    assert not unexpected, (
        "undocumented network call sites:\n  " + "\n  ".join(unexpected)
        + "\n\nAdd them to ALLOWED in this test, and check docs/NETWORK.md still describes what "
          "goes over the wire.")


def test_the_library_never_speaks_http_itself(parsed):
    """Only the Hub client may reach out: no raw HTTP anywhere in lib/ or cli/."""
    offenders = []
    for path, tree in parsed.items():
        for module in imported_modules(tree) & FORBIDDEN_MODULES:
            offenders.append(f"{path} imports {module}")
    assert not offenders, (
        "raw network clients in the library:\n  " + "\n  ".join(offenders)
        + "\n\nJuL has no service of its own. If this is deliberate, it needs its own review.")


def test_the_decision_path_downloads_nothing(parsed):
    """What runs per decision must not be able to touch the network, cached or not."""
    offenders = []
    for path in DECISION_PATH:
        tree = parsed.get(path)
        assert tree is not None, f"{path} moved: DECISION_PATH in this test is now wrong"
        for name in called_names(tree) & (ALLOWED.keys() | FORBIDDEN):
            offenders.append(f"{path} calls {name}()")
        for module in imported_modules(tree) & FORBIDDEN_MODULES:
            offenders.append(f"{path} imports {module}")
    assert not offenders, (
        "the decision path reaches the network:\n  " + "\n  ".join(offenders)
        + "\n\nA decision is offline by design; this breaks the promise in docs/NETWORK.md.")


def test_the_document_lists_the_hosts_to_open():
    """The reason the page exists: someone has to put these in a firewall rule."""
    assert DOC.exists(), "docs/NETWORK.md is the answer given to security teams; it must exist"
    text = DOC.read_text()
    for host in ("huggingface.co", "cdn.hf.co", "cas-server.xethub.hf.co", "pypi.org"):
        assert host in text, f"docs/NETWORK.md does not name {host}"


def test_offline_switches_are_documented():
    """The escape hatches an air-gapped or proxied install needs."""
    text = DOC.read_text()
    for variable in ("HF_HUB_OFFLINE", "HF_ENDPOINT", "HF_HOME", "HF_HUB_DISABLE_XET",
                     "HF_HUB_DISABLE_TELEMETRY"):
        assert variable in text, f"docs/NETWORK.md does not mention {variable}"
