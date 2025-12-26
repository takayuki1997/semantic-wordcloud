#!/usr/bin/env python3
"""意味的に類似した単語を近くに配置するワードクラウド生成スクリプト"""

import argparse
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI
from janome.tokenizer import Tokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text

# 日本語フォントのパス（macOS）
FONT_PATHS = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

# 除外する汎用的な単語
STOPWORDS = {
    # 日本語ストップワード
    '研究', '可能', '手法', '科学', '対象', '利用', '実現', '必要', 'たち',
    'こと', 'もの', 'ため', 'よう', 'それ', 'これ', 'ここ', 'そこ',
    # 英語ストップワード
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been',
    'jpg', 'png', 'pdf', 'http', 'https', 'www',
}


def get_font_path() -> str:
    """利用可能な日本語フォントパスを取得"""
    for fp in FONT_PATHS:
        if Path(fp).exists():
            return fp
    return None


def extract_words(text: str) -> list[str]:
    """テキストから名詞を抽出"""
    tokenizer = Tokenizer()
    words = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')
        if pos[0] == '名詞' and pos[1] not in ['非自立', '代名詞', '数']:
            word = token.surface
            if len(word) > 1 and word.lower() not in STOPWORDS and word not in STOPWORDS:
                words.append(word)
    return words


def load_text_from_excel(excel_path: str) -> str:
    """Excelファイルから全テキストを抽出"""
    xlsx = pd.ExcelFile(excel_path)
    all_text = []
    for sheet_name in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet_name)
        for col in df.columns:
            for value in df[col].dropna():
                if isinstance(value, str):
                    all_text.append(value)
    return '\n'.join(all_text)


def load_text(file_path: str) -> str:
    """ファイルからテキストを読み込む"""
    path = Path(file_path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        return load_text_from_excel(file_path)
    else:
        return path.read_text(encoding='utf-8')


def get_embeddings(words: list[str], api_key: str = None) -> np.ndarray:
    """OpenAI APIで単語の埋め込みベクトルを取得"""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    print(f"OpenAI APIで{len(words)}個の単語の埋め込みを取得中...")

    # バッチで処理（APIの制限を考慮）
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  {min(i+batch_size, len(words))}/{len(words)} 完了")

    return np.array(all_embeddings)


def reduce_dimensions(embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
    """高次元ベクトルを2次元に次元削減"""
    print(f"{method.upper()}で2次元に次元削減中...")

    if method == 'tsne':
        # t-SNEはサンプル数が少ない場合perplexityを調整
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)

    coords = reducer.fit_transform(embeddings)
    return coords


def create_semantic_wordcloud(
    words: list[str],
    freqs: dict[str, int],
    coords: np.ndarray,
    output_path: str = "semantic_wordcloud.png"
):
    """意味的配置のワードクラウドを描画"""
    font_path = get_font_path()
    if font_path:
        plt.rcParams['font.family'] = 'sans-serif'
        # フォントを設定
        import matplotlib.font_manager as fm
        font_prop = fm.FontProperties(fname=font_path)
    else:
        font_prop = None
        print("警告: 日本語フォントが見つかりません")

    # 座標を正規化
    x = coords[:, 0]
    y = coords[:, 1]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # 頻度に基づいてフォントサイズを計算
    max_freq = max(freqs.values())
    min_freq = min(freqs.values())

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    # カラーマップ
    colors = plt.cm.tab10.colors

    texts = []
    for i, word in enumerate(words):
        freq = freqs[word]
        # フォントサイズ: 頻度に応じて12〜36
        if max_freq > min_freq:
            size = 12 + 24 * ((freq - min_freq) / (max_freq - min_freq)) ** 0.4
        else:
            size = 20

        color = colors[i % len(colors)]

        text = ax.text(
            x[i], y[i], word,
            fontsize=size,
            fontproperties=font_prop,
            color=color,
            ha='center', va='center',
            alpha=0.9
        )
        texts.append(text)

    # テキストの重なりを調整
    print("テキストの配置を調整中...")
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='意味的配置ワードクラウドを生成')
    parser.add_argument('input', help='入力ファイルのパス（.txt または .xlsx）')
    parser.add_argument('-o', '--output', default='semantic_wordcloud.png', help='出力画像パス')
    parser.add_argument('-n', '--num-words', type=int, default=80, help='表示する単語数')
    parser.add_argument('-m', '--method', choices=['tsne', 'pca'], default='tsne', help='次元削減手法')
    parser.add_argument('--api-key', help='OpenAI APIキー（環境変数OPENAI_API_KEYでも可）')
    args = parser.parse_args()

    # テキスト読み込み
    print(f"ファイルを読み込み中: {args.input}")
    text = load_text(args.input)

    # 単語抽出と頻度カウント
    words = extract_words(text)
    word_freq = Counter(words)

    print(f"抽出した単語数: {len(words)}")
    print(f"ユニークな単語数: {len(word_freq)}")

    # 上位N個の単語を取得
    top_words = [word for word, _ in word_freq.most_common(args.num_words)]
    top_freqs = {word: word_freq[word] for word in top_words}

    print(f"上位{len(top_words)}単語を使用")
    print(f"上位10単語: {word_freq.most_common(10)}")

    # 埋め込み取得
    embeddings = get_embeddings(top_words, args.api_key)

    # 次元削減
    coords = reduce_dimensions(embeddings, args.method)

    # ワードクラウド生成
    create_semantic_wordcloud(top_words, top_freqs, coords, args.output)


if __name__ == '__main__':
    main()
