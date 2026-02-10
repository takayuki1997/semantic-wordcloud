# CLAUDE.md

このファイルはClaude Codeがこのリポジトリで作業する際のガイドです。

## プロジェクト概要

意味的に類似した単語を近くに配置するワードクラウド生成ツール。OpenAI Embeddingsを使用。

## 環境セットアップ

```bash
# venv環境を使用
source venv/bin/activate

# 依存関係インストール（初回のみ）
pip install -r requirements.txt
```

## 基本的な起動方法

```bash
source venv/bin/activate
python semantic_wordcloud.py "研究内容と研究キーワード.xlsx" --custom-words custom_words.txt -o WordCloud_20251228.png
```

### よく使うオプション

- `-o OUTPUT`: 出力ファイル名（日付を含めることを推奨: `WordCloud_YYYYMMDD.png`）
- `-n NUM`: 表示単語数（デフォルト: 80）
- `--custom-words FILE`: カスタム単語ファイル
- `--iterations NUM`: レイアウト反復回数（デフォルト: 500）
- `--seed NUM`: ランダムシード（再現性確保）
- `--layout-method pca|mds`: 初期配置の方法（デフォルト: pca）

## 主要ファイル

| ファイル | 説明 |
|---------|------|
| `semantic_wordcloud.py` | メインスクリプト |
| `stopwords.txt` | カスタムストップワード |
| `custom_words.txt` | カスタム単語（タブ区切り: 単語 + 正規化値） |
| `embeddings_cache.json` | 埋め込みベクトルのキャッシュ |
| `stopwords_cache.json` | 標準ストップワードのキャッシュ |

## 出力ファイル

実行すると以下が生成されます:
- `WordCloud_YYYYMMDD.png` - ワードクラウド画像（PNG）
- `WordCloud_YYYYMMDD.svg` - ワードクラウド画像（SVG、フォントはパスに変換済み）
- `WordCloud_YYYYMMDD.csv` - 単語リスト（自動出力、末尾に除外単語も記載）

## レイアウトパラメータ

`semantic_wordcloud.py` 内の主要パラメータ:

| パラメータ | 現在値 | 説明 |
|-----------|--------|------|
| `canvas_width` | 1200 | キャンバス幅 |
| `canvas_height` | 900 | キャンバス高さ |
| `repulsion_strength` | 700 | 単語間の反発力 |
| `ideal_scale` | 0.30 | 初期配置のスケール係数 |
| `padding` | 1 | 重なり検出のパディング |
| `nudge` | 3.5 | 重なり解消時の移動量 |
| 回転制限 | 4文字以上 | 長い単語は回転させない |

## API キー

OpenAI APIキーは以下のいずれかで設定:
- 環境変数: `export OPENAI_API_KEY="..."`
- ファイル: `OpenAI_API_Key.txt`
- コマンドライン: `--api-key "..."`

## ドキュメント更新

`docs/algorithm.md` を変更した場合は、PDFも再生成すること:

```bash
source venv/bin/activate
python docs/generate_pdf.py
```

## GitHub

- **リポジトリ**: https://github.com/takayuki1997/semantic-wordcloud
- **メインブランチ**: `main`

### コミット時の注意

以下のファイルはコミットしない（.gitignoreに含まれている）:
- `OpenAI_API_Key.txt` - APIキー
- `venv/` - 仮想環境
- `*.pyc`, `__pycache__/` - Pythonキャッシュ
- `custom_words.txt` - カスタム単語（大学固有の情報を含む）
- `stopwords.txt` - 除外単語（大学固有の情報を含む）
- `*.csv` - 出力CSV
- `embeddings_cache.json` - 埋め込みキャッシュ
- `stopwords_cache.json` - ストップワードキャッシュ

以下のファイルはコミットしてよい:
- `custom_words_example.txt` - カスタム単語のサンプル
- `stopwords_example.txt` - 除外単語のサンプル

※ サンプル画像は `examples/` に既存のものがあるため、新規画像はアップロード不要
