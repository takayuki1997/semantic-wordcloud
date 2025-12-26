# Semantic Word Cloud

意味的に類似した単語を近くに配置するワードクラウド生成ツール

![サンプル出力](examples/sample_output.png)

## 特徴

- **意味的配置**: OpenAI Embeddingsを使用し、類似した単語を近くに配置
- **日本語対応**: Janomeによる形態素解析で日本語テキストを処理
- **標準ストップワード**: 日本語・英語のストップワードを自動取得
- **キャッシュ機能**: 埋め込みベクトルをキャッシュしてAPI呼び出しを削減
- **異方性レイアウト**: 横長・縦長など任意のアスペクト比に対応

## インストール

```bash
git clone https://github.com/takayuki1997/semantic-wordcloud.git
cd semantic-wordcloud
pip install -r requirements.txt
```

## 使い方

### 基本

```bash
export OPENAI_API_KEY="your-api-key"
python semantic_wordcloud.py input.txt -o output.png
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-o, --output` | semantic_wordcloud.png | 出力ファイル名 |
| `-n, --num-words` | 80 | 表示する単語数 |
| `--cache-words` | 200 | キャッシュする単語数 |
| `--iterations` | 500 | レイアウト反復回数 |
| `--seed` | None | ランダムシード（再現性確保） |
| `--api-key` | 環境変数 | OpenAI APIキー |

### 例

```bash
# Excel ファイルから生成
python semantic_wordcloud.py research_data.xlsx -o wordcloud.png

# 単語数を増やして生成
python semantic_wordcloud.py input.txt -n 100 --iterations 800

# 再現可能な結果を得る
python semantic_wordcloud.py input.txt --seed 42
```

## 対応フォーマット

- テキストファイル (.txt)
- Excel ファイル (.xlsx, .xls)

## カスタマイズ

### ストップワード

`stopwords.txt` を編集して除外する単語を追加できます：

```
# コメント行
研究
可能
手法
```

## アルゴリズム

詳細は [docs/algorithm.md](docs/algorithm.md)（[PDF版](docs/algorithm.pdf)）を参照してください。

### 概要

1. **形態素解析**: Janomeで名詞を抽出
2. **埋め込み取得**: OpenAI text-embedding-3-small (1536次元)
3. **Force-directed Layout**: 高次元コサイン距離を制約として2D配置を最適化
4. **色付け**: PCAによる2D射影の角度から色相を決定

## 必要要件

- Python 3.10+
- OpenAI API キー

## ライセンス

MIT License

## 関連リンク

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Janome](https://mocobeta.github.io/janome/)
