#!/usr/bin/env python3
"""Markdown から PDF を生成するスクリプト (TinyTeX + XeLaTeX)"""

import subprocess
import sys
from pathlib import Path

def main():
    docs_dir = Path(__file__).parent
    md_file = docs_dir / "algorithm.md"
    pdf_file = docs_dir / "algorithm.pdf"

    # TinyTeX の pandoc を探す（ユーザーディレクトリにインストール）
    tinytex_bin = Path.home() / "Library/TinyTeX/bin/universal-darwin"

    # 環境変数にTinyTeXのパスを追加
    import os
    env = os.environ.copy()
    env["PATH"] = f"{tinytex_bin}:{env.get('PATH', '')}"

    print("Markdown → PDF (pandoc + xelatex)")
    result = subprocess.run([
        "pandoc",
        str(md_file),
        "-o", str(pdf_file),
        "--pdf-engine=xelatex",
        "-V", "documentclass=ltjarticle",
        "-V", "geometry:margin=2.5cm",
        "-V", "fontsize=10pt",
        "--highlight-style=tango"
    ], capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    if result.stderr:
        # 警告があっても表示
        print(f"Warnings: {result.stderr[:500]}")

    print(f"完了: {pdf_file}")

if __name__ == "__main__":
    main()
