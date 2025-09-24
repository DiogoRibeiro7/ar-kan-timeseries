#!/usr/bin/env python
"""check_imports_vs_pyproject.py

Scan Python sources for imports, compare to pyproject.toml, and optionally FIX:
- Adds missing dependencies to the chosen group with a selectable spec strategy.
- Preserves formatting/comments using tomlkit.
- Supports Poetry ([tool.poetry]) and PEP 621 ([project]).

Examples:
--------
# Detect only (text)
python scripts/check_imports_vs_pyproject.py

# Detect and fail CI on missing
python scripts/check_imports_vs_pyproject.py --fail-on missing

# Fix: add missing to main group using caret spec (^X.Y.Z), resolving latest from PyPI
python scripts/check_imports_vs_pyproject.py --fix --strategy caret --resolve-latest

# Fix into 'dev' group (Poetry) or optional group 'dev' (PEP 621)
python scripts/check_imports_vs_pyproject.py --fix --fix-to dev --strategy floor

# Dry run (print unified diff; do not write)
python scripts/check_imports_vs_pyproject.py --fix --check

Exit codes
----------
0 OK, 1 Missing only, 2 Unused only, 3 Missing+Unused (when --fail-on=both)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import tomlkit
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

# --- Optional helpers for stdlib / env mapping ---
try:
    _STDLIB: set[str] = set(sys.stdlib_module_names)  # py>=3.10
except Exception:
    _STDLIB = set()

try:
    from importlib.metadata import packages_distributions

    _MODULE_TO_DISTS: dict[str, list[str]] = packages_distributions()
except Exception:
    _MODULE_TO_DISTS = {}

_COMMON_MODULE_TO_DIST: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
    "importlib_metadata": "importlib-metadata",
    "pkg_resources": "setuptools",
}

# ---------------- Models ----------------


@dataclass(frozen=True)
class Config:
    root: Path
    pyproject: Path
    groups: list[str]  # Which groups count as "declared" when checking
    include_optional: bool
    fail_on: str  # missing|unused|both|none
    fmt: str  # text|json
    exclude_dirs: list[str]
    use_env_map: bool
    src_hints: list[str]
    # Fix-related:
    fix: bool
    fix_to: str  # main|dev|<group>
    strategy: str  # caret|tilde|exact|floor
    resolve_latest: bool
    include_prerelease: bool
    check: bool  # dry-run (print diff; don't write)
    timeout: float


@dataclass
class Report:
    missing: dict[str, set[str]]  # dist -> {modules}
    unused: set[str]
    ambiguous: dict[str, set[str]]
    used_dists: set[str]
    declared_dists: set[str]


# ---------------- Utilities ----------------


def pep503(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def is_stdlib(top: str) -> bool:
    return top in _STDLIB


def iter_py_files(root: Path, exclude_dirs: list[str]) -> Iterable[Path]:
    default = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "build",
        "dist",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
    }
    excl = set(exclude_dirs)
    for p in root.rglob("*.py"):
        ps = set(p.parts)
        if ps & default:
            continue
        if any(e in ps for e in excl):
            continue
        yield p


def parse_imports(pyfile: Path) -> set[str]:
    text = pyfile.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(pyfile))
    except SyntaxError:
        return set()
    tops: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name:
                    tops.add(a.name.split(".", 1)[0])
        elif isinstance(n, ast.ImportFrom):
            if n.level and n.level > 0:
                continue
            if n.module:
                tops.add(n.module.split(".", 1)[0])
    return tops


def discover_local_tops(root: Path, hints: list[str]) -> set[str]:
    locals_: set[str] = set()

    def scan(base: Path) -> None:
        if not base.exists() or not base.is_dir():
            return
        for c in base.iterdir():
            if c.is_dir():
                if (c / "__init__.py").exists():
                    locals_.add(c.name)
            elif c.suffix == ".py":
                locals_.add(c.stem)

    for h in hints:
        scan(root / h)
    scan(root)
    locals_ -= {"tests", "test"}
    return locals_


def map_module_to_dists(mod: str, use_env: bool = True) -> list[str]:
    if use_env and _MODULE_TO_DISTS:
        d = _MODULE_TO_DISTS.get(mod, [])
        if d:
            return [pep503(x) for x in d]
    if mod in _COMMON_MODULE_TO_DIST:
        return [pep503(_COMMON_MODULE_TO_DIST[mod])]
    return [pep503(mod)]


# ------------- PyPI lookup + strategy -------------


def fetch_latest(name: str, timeout: float, include_prerelease: bool) -> Version | None:
    url = f"https://pypi.org/pypi/{pep503(name)}/json"
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return None
    releases = data.get("releases", {}) or {}
    best: Version | None = None
    for vstr, files in releases.items():
        if not files:
            continue
        if all(bool(f.get("yanked", False)) for f in files if isinstance(f, dict)):
            continue
        try:
            v = Version(vstr)
        except InvalidVersion:
            continue
        if not include_prerelease and v.is_prerelease:
            continue
        if best is None or v > best:
            best = v
    return best


def pep621_spec_for(v: Version, strategy: str) -> str:
    if strategy == "exact":
        return f"=={v}"
    if strategy == "floor":
        return f">={v}"
    if strategy == "caret":
        upper = f"{v.major + 1}.0.0"
        return f">={v},<{upper}"
    if strategy == "tilde":
        upper = f"{v.major}.{v.minor + 1}.0"
        return f">={v},<{upper}"
    raise ValueError(f"Unknown strategy: {strategy}")


def poetry_spec_for(v: Version, strategy: str) -> str:
    if strategy == "exact":
        return f"=={v}"
    if strategy == "floor":
        return f">={v}"
    if strategy == "caret":
        return f"^{v}"
    if strategy == "tilde":
        return f"~{v}"
    raise ValueError(f"Unknown strategy: {strategy}")


# ------------- Read/Write pyproject (tomlkit) -------------


def load_doc(path: Path) -> tuple[str, tomlkit.TOMLDocument]:
    text = path.read_text(encoding="utf-8")
    return text, tomlkit.parse(text)


def dump_doc(doc: tomlkit.TOMLDocument) -> str:
    return tomlkit.dumps(doc)


def layout(doc: tomlkit.TOMLDocument) -> str:
    if "tool" in doc and isinstance(doc["tool"], dict) and "poetry" in doc["tool"]:
        return "poetry"
    if "project" in doc:
        return "pep621"
    raise ValueError("Unsupported pyproject: neither [tool.poetry] nor [project] found.")


def collect_declared(
    doc: tomlkit.TOMLDocument, groups: list[str], include_optional: bool
) -> set[str]:
    wanted = set(groups)
    dec: set[str] = set()

    # Poetry
    tool = doc.get("tool", {})
    poetry = tool.get("poetry") if isinstance(tool, dict) else None
    if poetry:
        if not wanted or "main" in wanted:
            for name in (poetry.get("dependencies", {}) or {}).keys():
                if name != "python":
                    dec.add(pep503(str(name)))
        if "dev" in wanted:
            for name in (poetry.get("dev-dependencies", {}) or {}).keys():
                dec.add(pep503(str(name)))
        grp = poetry.get("group", {}) or {}
        if isinstance(grp, dict):
            for gname, gtbl in grp.items():
                if wanted and gname not in wanted:
                    continue
                deps = (gtbl or {}).get("dependencies", {}) or {}
                for name in deps.keys():
                    dec.add(pep503(str(name)))

    # PEP 621
    project = doc.get("project")
    if project:
        if not wanted or "main" in wanted:
            for item in project.get("dependencies", []) or []:
                try:
                    req = Requirement(str(item))
                    dec.add(pep503(req.name))
                except Exception:
                    continue
        opt = project.get("optional-dependencies", {}) or {}
        if isinstance(opt, dict):
            for gname, arr in opt.items():
                if (not include_optional) and (wanted and gname not in wanted):
                    continue
                for item in arr or []:
                    try:
                        req = Requirement(str(item))
                        dec.add(pep503(req.name))
                    except Exception:
                        continue

    return dec


def add_dep_to_doc(doc: tomlkit.TOMLDocument, dist: str, spec: str, fix_to: str) -> None:
    """Add dependency to doc in the selected group.
    - Poetry: under [tool.poetry.dependencies] or group.<g>.dependencies
    - PEP 621: in project.dependencies (main) or optional-dependencies.<g>
    """
    lay = layout(doc)
    if lay == "poetry":
        tool = doc.setdefault("tool", tomlkit.table())
        poetry = tool.setdefault("poetry", tomlkit.table())
        if fix_to == "main":
            deps = poetry.setdefault("dependencies", tomlkit.table())
        elif fix_to == "dev":
            deps = poetry.setdefault("dev-dependencies", tomlkit.table())
        else:
            grp = poetry.setdefault("group", tomlkit.table())
            gtbl = grp.setdefault(fix_to, tomlkit.table())
            deps = gtbl.setdefault("dependencies", tomlkit.table())
        deps[dist] = spec  # as string spec
        return

    # PEP 621
    project = doc.setdefault("project", tomlkit.table())
    if fix_to == "main":
        arr = project.setdefault("dependencies", tomlkit.array())
    else:
        opt = project.setdefault("optional-dependencies", tomlkit.table())
        arr = opt.setdefault(fix_to, tomlkit.array())
    if not isinstance(arr, tomlkit.items.Array):
        raise TypeError("Expected an array for dependencies.")
    arr.append(f"{dist} {spec}")


def write_or_diff(path: Path, before: str, after: str, check: bool) -> int:
    if before == after:
        return 0
    if check:
        sys.stdout.write(
            "".join(
                unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
        )
        return 0
    path.write_text(after, encoding="utf-8")
    return 0


# ------------- Analysis + Fix -------------


def analyze(cfg: Config) -> Report:
    before_text, doc = load_doc(cfg.pyproject)
    declared = collect_declared(doc, cfg.groups, cfg.include_optional)

    local = discover_local_tops(cfg.root, cfg.src_hints)
    imported: set[str] = set()
    for f in iter_py_files(cfg.root, cfg.exclude_dirs):
        imported |= parse_imports(f)

    third_party = {m for m in imported if not is_stdlib(m) and m not in local}

    used_dists: set[str] = set()
    ambiguous: dict[str, set[str]] = {}
    for mod in sorted(third_party):
        dists = map_module_to_dists(mod, cfg.use_env_map)
        if len(dists) == 1:
            used_dists.add(dists[0])
        else:
            ambiguous.setdefault(mod, set()).update(dists)
            # Count it as used if any candidate is declared (avoid false missing)
            for d in dists:
                if d in declared:
                    used_dists.add(d)

    missing: dict[str, set[str]] = {}
    for mod in sorted(third_party):
        dists = map_module_to_dists(mod, cfg.use_env_map)
        if any(d in declared for d in dists):
            continue
        missing.setdefault(dists[0], set()).add(mod)

    unused = declared - used_dists
    return Report(missing, unused, ambiguous, used_dists, declared)


def apply_fix(cfg: Config, rep: Report) -> int:
    """Add missing distributions to pyproject.toml using selected strategy.
    Returns 0 (write/diff success) or raises on errors.
    """
    if not rep.missing:
        return 0

    before, doc = load_doc(cfg.pyproject)
    lay = layout(doc)

    for dist, modules in sorted(rep.missing.items()):
        # Decide version spec
        if cfg.resolve_latest:
            latest = fetch_latest(dist, cfg.timeout, cfg.include_prerelease)
            if latest is None:
                # Fallback to floor 0 if PyPI query failed
                spec = ">=0"
            else:
                spec = (
                    poetry_spec_for(latest, cfg.strategy)
                    if lay == "poetry"
                    else pep621_spec_for(latest, cfg.strategy)
                )
        else:
            # No network: choose a conservative floor
            spec = ">=0" if cfg.strategy in {"caret", "tilde", "floor"} else "==0"
        # Write into the chosen group
        add_dep_to_doc(doc, dist, spec, cfg.fix_to)

    after = dump_doc(doc)
    return write_or_diff(cfg.pyproject, before, after, cfg.check)


# ------------- CLI -------------


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        description="Check (and optionally fix) imports vs pyproject dependencies."
    )
    p.add_argument("--root", type=Path, default=Path("."), help="Project root to scan.")
    p.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml.",
    )
    p.add_argument(
        "--groups",
        default="main,dev",
        help="Comma-separated groups considered 'declared' (Poetry/PEP 621).",
    )
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="Include ALL PEP 621 optional groups as declared.",
    )
    p.add_argument(
        "--fail-on",
        choices=("missing", "unused", "both", "none"),
        default="missing",
        help="Condition for non-zero exit code.",
    )
    p.add_argument("--format", choices=("text", "json"), default="text", dest="fmt")

    p.add_argument(
        "--exclude-dirs",
        default=".venv,venv,dist,build,.git,__pycache__",
        help="Comma-separated dir names to skip.",
    )
    p.add_argument(
        "--no-use-installed",
        action="store_true",
        help="Do not use importlib.metadata module→dist mapping from current environment.",
    )
    p.add_argument(
        "--src-hints",
        default="src",
        help="Comma-separated locations of local packages (e.g. src,app).",
    )

    # Fix options
    p.add_argument("--fix", action="store_true", help="Write missing deps into pyproject.toml.")
    p.add_argument(
        "--fix-to",
        default="main",
        help="Target group for fixes: 'main', 'dev', or a specific group name.",
    )
    p.add_argument(
        "--strategy",
        choices=("caret", "tilde", "exact", "floor"),
        default="caret",
        help="Version constraint strategy for new deps.",
    )
    p.add_argument(
        "--resolve-latest",
        action="store_true",
        help="Query PyPI to select latest version and generate a proper spec.",
    )
    p.add_argument(
        "--pre",
        dest="include_prerelease",
        action="store_true",
        help="Allow pre-releases when resolving.",
    )
    p.add_argument(
        "--no-pre",
        dest="include_prerelease",
        action="store_false",
        help="Exclude pre-releases (default).",
    )
    p.set_defaults(include_prerelease=False)
    p.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: print unified diff; do not write.",
    )
    p.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout for PyPI lookups.")

    a = p.parse_args(argv)
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    excludes = [e.strip() for e in a.exclude_dirs.split(",") if e.strip()]
    src_hints = [s.strip() for s in a.src_hints.split(",") if s.strip()]
    return Config(
        root=a.root.resolve(),
        pyproject=a.pyproject.resolve(),
        groups=groups,
        include_optional=bool(a.include_optional),
        fail_on=a.fail_on,
        fmt=a.fmt,
        exclude_dirs=excludes,
        use_env_map=not a.no_use_installed,
        src_hints=src_hints,
        fix=bool(a.fix),
        fix_to=a.fix_to.strip() or "main",
        strategy=a.strategy,
        resolve_latest=bool(a.resolve_latest),
        include_prerelease=a.include_prerelease,
        check=bool(a.check),
        timeout=float(a.timeout),
    )


def format_report_text(r: Report) -> str:
    lines: list[str] = []
    if r.missing:
        lines.append("MISSING (imported but not declared):")
        for dist, modules in sorted(r.missing.items()):
            lines.append(f"  - {dist}  ← used via: {', '.join(sorted(modules))}")
    else:
        lines.append("MISSING: none")
    if r.unused:
        lines.append("\nUNUSED (declared but not imported):")
        for d in sorted(r.unused):
            lines.append(f"  - {d}")
    else:
        lines.append("\nUNUSED: none")
    if r.ambiguous:
        lines.append("\nAMBIGUOUS (module → possible distributions):")
        for mod, ds in sorted(r.ambiguous.items()):
            lines.append(f"  - {mod} → {', '.join(sorted(ds))}")
    else:
        lines.append("\nAMBIGUOUS: none")
    lines.append("")
    return "\n".join(lines)


def format_report_json(r: Report) -> str:
    obj = {
        "missing": {k: sorted(v) for k, v in r.missing.items()},
        "unused": sorted(r.unused),
        "ambiguous": {k: sorted(v) for k, v in r.ambiguous.items()},
        "used_dists": sorted(r.used_dists),
        "declared_dists": sorted(r.declared_dists),
    }
    return json.dumps(obj, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    rep = analyze(cfg)

    # Print analysis first
    out = format_report_json(rep) if cfg.fmt == "json" else format_report_text(rep)
    sys.stdout.write(out)

    # Apply fix if requested
    if cfg.fix:
        apply_fix(cfg, rep)

    # Exit code policy
    missing_flag = bool(rep.missing)
    unused_flag = bool(rep.unused)
    if cfg.fail_on == "none":
        return 0
    if cfg.fail_on == "missing":
        return 1 if missing_flag else 0
    if cfg.fail_on == "unused":
        return 2 if unused_flag else 0
    if cfg.fail_on == "both":
        code = 0
        if missing_flag:
            code |= 1
        if unused_flag:
            code |= 2
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
