#!/usr/bin/env python3
"""Build, clean, and view the AIAA LaTeX paper."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


LATEX_DIR = Path(__file__).resolve().parent
REPO_ROOT = LATEX_DIR.parent
OUTPUT_DIR = LATEX_DIR
DEFAULT_TEX = LATEX_DIR / "main.tex"
VERSION = "0.1.0"
COMMANDS = {"build", "clean", "view", "info", "package", "help"}
GENERATED_EXTENSIONS = {
    ".acn",
    ".acr",
    ".alg",
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".glg",
    ".glo",
    ".gls",
    ".idx",
    ".ilg",
    ".ind",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".run.xml",
    ".synctex.gz",
    ".toc",
    ".w18",
}
SVG_GENERATED_SUFFIXES = ("_svg-tex.pdf", "_svg-tex.pdf_tex")
DEFAULT_PACKAGE = REPO_ROOT / "latex-overleaf.zip"


def resolve_tex_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        candidates = [
            Path.cwd() / path,
            LATEX_DIR / path,
            REPO_ROOT / path,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[1])
    return path.resolve()


def resolve_output_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def pdf_for_tex(tex_path: Path) -> Path:
    return OUTPUT_DIR / f"{tex_path.stem}.pdf"


def build(args: argparse.Namespace) -> Path:
    tex_path = resolve_tex_path(args.tex)
    if not tex_path.exists():
        raise SystemExit(f"TeX source not found: {tex_path}")
    if tex_path.suffix != ".tex":
        raise SystemExit(f"Expected a .tex source file, got: {tex_path}")

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise SystemExit("latexmk was not found on PATH. Install latexmk or add it to PATH.")

    if args.clean_first:
        remove_generated_files(tex_path.stem, remove_pdf=True)
        remove_svg_generated_files(tex_path.parent)

    build_started_at = time.time()
    pdf_path = pdf_for_tex(tex_path)

    command = [
        latexmk,
        "-pdf",
        "-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={OUTPUT_DIR}",
    ]
    command.append(str(tex_path.name))

    print("+", " ".join(command))
    result = subprocess.run(command, cwd=tex_path.parent)

    if not pdf_path.exists():
        raise SystemExit(f"Build completed, but expected PDF was not created: {pdf_path}")
    if result.returncode != 0 and pdf_path.stat().st_mtime < build_started_at:
        raise SystemExit(result.returncode)
    if result.returncode != 0:
        print(f"latexmk exited with {result.returncode}, but produced a fresh PDF.")

    print(f"PDF: {pdf_path}")
    if args.view:
        open_pdf(pdf_path)
    return pdf_path


def clean(args: argparse.Namespace) -> None:
    tex_path = resolve_tex_path(args.tex)
    if not tex_path.exists():
        raise SystemExit(f"TeX source not found: {tex_path}")

    removed = remove_generated_files(tex_path.stem, remove_pdf=not args.keep_pdf)
    removed.extend(remove_svg_generated_files(tex_path.parent))

    if removed:
        print(f"Removed {len(removed)} generated file(s).")
    else:
        print("No generated files found.")


def view(args: argparse.Namespace) -> None:
    tex_path = resolve_tex_path(args.tex)
    pdf_path = pdf_for_tex(tex_path)
    if args.build:
        build_args = argparse.Namespace(tex=args.tex, view=False, clean_first=args.clean_first)
        pdf_path = build(build_args)
    elif not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}. Run './paper.py build' first.")

    open_pdf(pdf_path)


def info(args: argparse.Namespace) -> None:
    tex_path = resolve_tex_path(args.tex)
    pdf_path = pdf_for_tex(tex_path)
    latexmk = shutil.which("latexmk") or "not found"
    xdg_open = shutil.which("xdg-open") or "not found"

    print(f"repo:    {REPO_ROOT}")
    print(f"latex:   {LATEX_DIR}")
    print(f"output:  {OUTPUT_DIR}")
    print(f"tex:     {tex_path}")
    print(f"pdf:     {pdf_path}")
    print(f"latexmk: {latexmk}")
    if platform.system() == "Linux":
        print(f"viewer:  {xdg_open}")


def package(args: argparse.Namespace) -> None:
    tex_path = resolve_tex_path(args.tex)
    if not tex_path.exists():
        raise SystemExit(f"TeX source not found: {tex_path}")

    archive_path = resolve_output_path(args.output)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clean_first:
        clean_args = argparse.Namespace(tex=args.tex, keep_pdf=not args.exclude_pdf)
        clean(clean_args)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(LATEX_DIR.rglob("*")):
            if not path.is_file() or should_exclude_from_package(path, tex_path.stem, archive_path, args):
                continue
            archive.write(path, path.relative_to(LATEX_DIR))

    print(f"Package: {archive_path}")


def help_command(args: argparse.Namespace) -> None:
    p = parser()
    command_parsers = getattr(p, "_command_parsers")
    if args.topic:
        print(command_parsers[args.topic].format_help(), end="")
    else:
        print(p.format_help(), end="")


def is_latex_generated(path: Path, stem: str, remove_pdf: bool = False) -> bool:
    if path.parent != OUTPUT_DIR:
        return False
    if not path.name.startswith(f"{stem}."):
        return False

    suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".synctex.gz") else path.suffix
    return suffix in GENERATED_EXTENSIONS or (remove_pdf and suffix == ".pdf")


def is_svg_generated(path: Path) -> bool:
    return path.name.endswith(SVG_GENERATED_SUFFIXES)


def remove_generated_files(stem: str, remove_pdf: bool = False) -> list[Path]:
    removed: list[Path] = []

    for path in OUTPUT_DIR.glob(f"{stem}.*"):
        if is_latex_generated(path, stem, remove_pdf=remove_pdf):
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def remove_svg_generated_files(latex_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for path in latex_dir.rglob("*_svg-tex.*"):
        if is_svg_generated(path):
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def should_exclude_from_package(
    path: Path,
    stem: str,
    archive_path: Path,
    args: argparse.Namespace,
) -> bool:
    if path.resolve() == archive_path.resolve():
        return True
    if "__pycache__" in path.parts:
        return True
    if path.name.endswith((".pyc", ".pyo")):
        return True
    if path.name == ".DS_Store":
        return True
    if path.parent == LATEX_DIR and path.name == ".gitignore":
        return True
    if path.parent == LATEX_DIR and path.suffix == ".zip":
        return True
    if path.parent.name == "svg-inkscape":
        return True
    if is_svg_generated(path):
        return True
    if is_latex_generated(path, stem, remove_pdf=args.exclude_pdf):
        return True
    return False


def open_pdf(pdf_path: Path) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            opener = ["open", str(pdf_path)]
        elif system == "Windows":
            os.startfile(pdf_path)  # type: ignore[attr-defined]
            return
        else:
            opener_path = shutil.which("xdg-open")
            if opener_path is None:
                raise RuntimeError("xdg-open was not found on PATH")
            opener = [opener_path, str(pdf_path)]

        subprocess.Popen(
            opener,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        raise SystemExit(f"PDF built, but could not open viewer: {exc}") from exc


def parser() -> argparse.ArgumentParser:
    description = "Build, clean, and view the system identification literature review paper."
    epilog = f"""examples:
  cd latex
  ./paper.py
  ./paper.py build
  ./paper.py build --view
  ./paper.py --view
  ./paper.py clean
  ./paper.py clean --keep-pdf
  ./paper.py package
  ./paper.py view
  ./paper.py info

defaults:
  TeX root: {DEFAULT_TEX.relative_to(LATEX_DIR)}
  PDF out:  {OUTPUT_DIR.relative_to(LATEX_DIR)}/
"""
    p = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subparsers = p.add_subparsers(dest="command")
    command_parsers: dict[str, argparse.ArgumentParser] = {}

    build_parser = subparsers.add_parser(
        "build",
        help="build the paper PDF",
        description="Build the LaTeX paper with latexmk.",
    )
    command_parsers["build"] = build_parser
    build_parser.add_argument(
        "tex",
        nargs="?",
        default=str(DEFAULT_TEX.relative_to(LATEX_DIR)),
        help="TeX root to build, relative to the current directory or latex/ "
        f"(default: {DEFAULT_TEX.relative_to(LATEX_DIR)})",
    )
    build_parser.add_argument("-v", "--view", action="store_true", help="open the PDF after building")
    build_parser.add_argument(
        "--clean-first",
        action="store_true",
        help="remove generated build files before compiling",
    )
    build_parser.set_defaults(func=build)

    clean_parser = subparsers.add_parser(
        "clean",
        help="remove generated LaTeX build files",
        description="Remove generated PDF build files and SVG conversion helpers.",
    )
    command_parsers["clean"] = clean_parser
    clean_parser.add_argument(
        "tex",
        nargs="?",
        default=str(DEFAULT_TEX.relative_to(LATEX_DIR)),
        help="TeX root to clean, relative to the current directory or latex/ "
        f"(default: {DEFAULT_TEX.relative_to(LATEX_DIR)})",
    )
    clean_parser.add_argument("--keep-pdf", action="store_true", help="leave the generated PDF in place")
    clean_parser.set_defaults(func=clean)

    view_parser = subparsers.add_parser(
        "view",
        help="open the generated PDF",
        description="Open the generated PDF in the platform viewer.",
    )
    command_parsers["view"] = view_parser
    view_parser.add_argument(
        "tex",
        nargs="?",
        default=str(DEFAULT_TEX.relative_to(LATEX_DIR)),
        help="TeX root whose PDF should be opened "
        f"(default: {DEFAULT_TEX.relative_to(LATEX_DIR)})",
    )
    view_parser.add_argument("-b", "--build", action="store_true", help="build before opening the PDF")
    view_parser.add_argument(
        "--clean-first",
        action="store_true",
        help="when used with --build, remove generated build files before compiling",
    )
    view_parser.set_defaults(func=view)

    info_parser = subparsers.add_parser(
        "info",
        help="show configured paths and tool availability",
        description="Print the resolved paper paths and external tools used by the CLI.",
    )
    command_parsers["info"] = info_parser
    info_parser.add_argument(
        "tex",
        nargs="?",
        default=str(DEFAULT_TEX.relative_to(LATEX_DIR)),
        help="TeX root to inspect "
        f"(default: {DEFAULT_TEX.relative_to(LATEX_DIR)})",
    )
    info_parser.set_defaults(func=info)

    package_parser = subparsers.add_parser(
        "package",
        help="create a clean Overleaf upload zip",
        description="Zip the latex/ source tree while excluding generated build files.",
    )
    command_parsers["package"] = package_parser
    package_parser.add_argument(
        "tex",
        nargs="?",
        default=str(DEFAULT_TEX.relative_to(LATEX_DIR)),
        help="TeX root used to identify generated build files "
        f"(default: {DEFAULT_TEX.relative_to(LATEX_DIR)})",
    )
    package_parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_PACKAGE),
        help=f"zip file to write (default: {DEFAULT_PACKAGE})",
    )
    package_parser.add_argument(
        "--clean-first",
        action="store_true",
        help="remove generated files from latex/ before creating the zip",
    )
    package_parser.add_argument(
        "--include-pdf",
        dest="exclude_pdf",
        action="store_false",
        help="include the generated root PDF in the zip",
    )
    package_parser.set_defaults(func=package, exclude_pdf=True)

    help_parser = subparsers.add_parser(
        "help",
        help="show general help or command help",
        description="Show help for the CLI or a specific command.",
    )
    command_parsers["help"] = help_parser
    help_parser.add_argument("topic", nargs="?", choices=sorted(command_parsers), help="command to explain")
    help_parser.set_defaults(func=help_command)

    setattr(p, "_command_parsers", command_parsers)
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["build"]
    elif argv[0] not in COMMANDS and argv[0] not in {"-h", "--help", "--version"}:
        argv = ["build", *argv]

    p = parser()
    args = p.parse_args(argv)

    try:
        args.func(args)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
