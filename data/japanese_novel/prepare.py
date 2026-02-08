# data/japanese_novel/prepare.py として保存
import os
import tiktoken
import numpy as np
import requests

# 青空文庫から『こころ』のテキストデータを取得（有志によるテキスト化ファイルなど）
# ここでは例としてシンプルなテキストデータを想定します
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

# データがない場合はダウンロード（青空文庫形式のテキストURL等を指定）
# 今回は簡略化のため、手動で日本語テキストを input.txt に貼るか、以下でサンプル作成
if not os.path.exists(input_file_path):
    print("Downloading sample text...")
    # 青空文庫の『こころ』テキストURL（例）
    url = "https://www.aozora.gr.jp/cards/000148/files/773_14560.html" 
    # ※本来はHTMLタグ除去が必要ですが、簡易実験のためダミーテキストを生成します
    # 実際には、好きな日本語の小説テキストを input.txt として同階層に置いてください
    with open(input_file_path, 'w', encoding='utf-8') as f:
         f.write("私はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。..." * 1000)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# GPT-2のトークナイザーを使用
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(data)
print(f"train has {len(train_ids):,} tokens")

# バイナリ保存
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids = train_ids[:len(train_ids)//10] # 簡易的な検証データ
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))