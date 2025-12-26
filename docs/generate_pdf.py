#!/usr/bin/env python3
"""Markdown から PDF を生成するスクリプト

使い方:
    python generate_pdf.py          # デフォルト（WeasyPrint）
    python generate_pdf.py --html   # WeasyPrint (HTML経由、綺麗)
    python generate_pdf.py --latex  # LaTeX (数式向け)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def generate_with_weasyprint(md_file: Path, pdf_file: Path):
    """WeasyPrint方式: Markdown → HTML → PDF（Chromium風の綺麗な出力）"""
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        print("Error: weasyprint がインストールされていません")
        print("  pip install weasyprint")
        sys.exit(1)

    html_file = md_file.with_suffix(".html")

    # Step 1: Markdown → HTML (pandoc)
    print("Step 1: Markdown → HTML (pandoc)")
    result = subprocess.run([
        "pandoc",
        str(md_file),
        "-o", str(html_file),
        "--standalone",
        "--metadata", "title=意味的ワードクラウド アルゴリズム解説",
        "--toc",
        "--toc-depth=3",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    # Step 2: HTML → PDF (WeasyPrint)
    print("Step 2: HTML → PDF (WeasyPrint)")

    custom_css = CSS(string='''
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');

        body {
            font-family: "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif;
            font-size: 10pt;
            line-height: 1.7;
            max-width: none;
            margin: 0;
            padding: 2cm;
        }

        h1 {
            font-size: 18pt;
            border-bottom: 2px solid #333;
            padding-bottom: 0.3em;
            margin-top: 1.5em;
        }

        h2 {
            font-size: 14pt;
            border-bottom: 1px solid #ccc;
            padding-bottom: 0.2em;
            margin-top: 1.5em;
        }

        h3 {
            font-size: 12pt;
            margin-top: 1.2em;
        }

        h4 {
            font-size: 11pt;
            margin-top: 1em;
        }

        pre, code {
            font-family: "SF Mono", "Monaco", "Consolas", monospace;
            font-size: 9pt;
            background-color: #f6f8fa;
            border-radius: 3px;
        }

        pre {
            padding: 1em;
            overflow-x: auto;
            border: 1px solid #e1e4e8;
        }

        code {
            padding: 0.2em 0.4em;
        }

        pre code {
            padding: 0;
            background: none;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 9pt;
            margin: 1em 0;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 0.5em 0.8em;
            text-align: left;
        }

        th {
            background-color: #f6f8fa;
            font-weight: bold;
        }

        tr:nth-child(even) {
            background-color: #fafafa;
        }

        a {
            color: #0366d6;
            text-decoration: none;
        }

        blockquote {
            border-left: 4px solid #ddd;
            margin: 1em 0;
            padding-left: 1em;
            color: #666;
        }

        /* 目次のスタイル */
        nav#TOC {
            background-color: #f8f9fa;
            padding: 1em;
            margin-bottom: 2em;
            border-radius: 5px;
        }

        nav#TOC ul {
            list-style-type: none;
            padding-left: 1.5em;
        }

        nav#TOC > ul {
            padding-left: 0;
        }

        /* ページ区切り */
        h1 {
            page-break-before: auto;
        }

        pre, table {
            page-break-inside: avoid;
        }

        @page {
            size: A4;
            margin: 2cm;
        }
    ''')

    HTML(filename=str(html_file)).write_pdf(str(pdf_file), stylesheets=[custom_css])

    # 中間HTMLファイルを削除
    html_file.unlink()
    print(f"完了: {pdf_file}")


def generate_with_latex(md_file: Path, pdf_file: Path):
    """LaTeX方式: Markdown → PDF（数式に強い）"""

    # TinyTeX のパスを探す
    tinytex_bin = Path.home() / "Library/TinyTeX/bin/universal-darwin"
    if not tinytex_bin.exists():
        # Linux/Windows向けのパスも試す
        for path in [
            Path.home() / ".TinyTeX/bin/x86_64-linux",
            Path.home() / "AppData/Roaming/TinyTeX/bin/win32",
        ]:
            if path.exists():
                tinytex_bin = path
                break

    env = os.environ.copy()
    if tinytex_bin.exists():
        env["PATH"] = f"{tinytex_bin}:{env.get('PATH', '')}"

    print("Markdown → PDF (pandoc + lualatex)")
    result = subprocess.run([
        "pandoc",
        str(md_file),
        "-o", str(pdf_file),
        "--pdf-engine=lualatex",
        "-V", "documentclass=ltjarticle",
        "-V", "geometry:margin=2.5cm",
        "-V", "fontsize=10pt",
        "--toc",
        "--toc-depth=3",
        "--highlight-style=tango"
    ], capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    if result.stderr:
        # 警告は一部だけ表示
        warnings = result.stderr[:500]
        if "Missing character" in warnings or "undefined" in warnings:
            print("(一部の文字に警告がありますが、PDF は生成されました)")

    print(f"完了: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Markdown から PDF を生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
方式の違い:
  --html   WeasyPrint使用。Web風の綺麗なスタイル。日本語も問題なし。
  --latex  LaTeX使用。数式が美しい。学術文書向け。
        """
    )
    parser.add_argument("--html", action="store_true", help="WeasyPrint方式 (デフォルト)")
    parser.add_argument("--latex", action="store_true", help="LaTeX方式")
    parser.add_argument("input", nargs="?", help="入力Markdownファイル")

    args = parser.parse_args()

    docs_dir = Path(__file__).parent
    md_file = Path(args.input) if args.input else docs_dir / "algorithm.md"
    pdf_file = md_file.with_suffix(".pdf")

    if not md_file.exists():
        print(f"Error: {md_file} が見つかりません")
        sys.exit(1)

    if args.latex:
        generate_with_latex(md_file, pdf_file)
    else:
        # デフォルトは WeasyPrint（綺麗で簡単）
        generate_with_weasyprint(md_file, pdf_file)


if __name__ == "__main__":
    main()
