from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional


def _discover_projects(root: Path) -> List[str]:
    projects_dir = root / ".reticus" / "projects"
    if not projects_dir.exists():
        return []
    return sorted(p.name for p in projects_dir.iterdir() if p.is_dir())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reticus",
        description="Serve the Reticus dashboard for an existing project's data.",
    )
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Project name. Omit to auto-pick the only project, or list available ones.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port (default 8765).")
    parser.add_argument("--dir", default=".", help="Directory containing .reticus/ (default: cwd).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    args = parser.parse_args(argv)

    root = Path(args.dir).resolve()
    projects = _discover_projects(root)

    if not projects:
        sys.stderr.write(
            "No Reticus projects found under %s/.reticus/projects/.\n"
            "Run a training script with Reticus first, or pass --dir.\n" % root
        )
        return 1

    project = args.project
    if project is None:
        if len(projects) == 1:
            project = projects[0]
        else:
            sys.stderr.write("Multiple projects found. Pick one:\n")
            for name in projects:
                sys.stderr.write("  reticus %s\n" % name)
            return 1
    elif project not in projects:
        sys.stderr.write("Project '%s' not found. Available: %s\n" % (project, ", ".join(projects)))
        return 1

    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        sys.stderr.write(
            "The dashboard needs the web stack. Install it with:\n"
            '  pip install "reticus[dashboard]"\n'
        )
        return 1

    app = create_app(root, {"project": project})
    url = "http://127.0.0.1:%d" % args.port
    print("Reticus dashboard for '%s' -> %s" % (project, url))
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        try:
            # open after a short delay so the server has bound
            import threading

            def _open() -> None:
                time.sleep(1.0)
                webbrowser.open(url)

            threading.Thread(target=_open, daemon=True).start()
        except Exception:
            pass

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
