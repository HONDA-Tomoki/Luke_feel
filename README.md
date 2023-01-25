# You could do fine-tuning luke-japanese-base-lite be using these code
You could do fine tuning luke, by using "luke_train_squad.py".
However, fine-tuning needs a lot of time >_<.
So, please use "luke_squad.py".
Pre-made model "Mizuiro-sakura/luke-japanese-base-finetuned-QA" is available !

# このリポジトリのコードを利用することで、luke-japanese-base-liteのファインチューニングを行うことができます。
ファインチューニングは"luke_train_squad.py"を実行することで行うことができます。
しかし、ファインチューニングには時間がかかるため、私が事前に作成しておいたモデル”Mizuiro-sakura/luke-japanese-base-finetuned-QA”を利用することをお勧めします。
"luke_squad.py"を実行することで、自動的にモデルがダウンロードされ、QAタスクを解くことができます。

# 環境
torch 1.12.1
transformers 4.24.0
Python 3.9.13
sentencepiece 0.1.97


**transformersのバージョンが古いとLukeForQuestionAnsweringが含まれていないので注意してください。**（上記のバージョンまでアップデートしてください）

# LUKE

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/bd1b4096-0520-7a5b-2377-1fa1295fedb0.png)


2020年4月当時、5つのタスク（Open Entity, TACRED, CoNLL2003, ReCoRD, SQuAD 1.1）で世界最高精度を達成した新しい言語モデル。
日本語バージョンのLUKEは執筆現在（2023年1月）も4つのタスク（MARC-ja, JSTS, JNLI, JCommonsenseQA）で最高スコアを有しています。RoBERTaを元として構成され、entity-aware self-attentionという独自のメカニズムを用いています。LUKEに関して詳しくは下記記事をご覧ください。

https://qiita.com/Mizuiro__sakura/items/9ccbd655501e78df5cc6

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/c2451340-beeb-8664-dff8-e93004350c52.png)

# データセット

今回はファインチューニングのためのデータとして[運転ドメインQAデータセット（DDQA）](https://nlp.ist.i.kyoto-u.ac.jp/?Driving+domain+QA+datasets)を用いました。
このデータセットはウェブ上で公開されている運転ドメインのブログ記事を基に構築されており、述語項構造QAデータセット（PAS-QAデータセット）と文章読解QAデータセット（RC-QAデータセット）から構成されています。その中でも、RC-QAデータセットは文章の中から質問に対する答えを抽出する問題です。今回はSQuAD（質問に対する応答（該当部分の抜き出し））の学習を行いたいので、RC-QAデータセットを用いました。

# 学習結果

```
'em': 0.845933014354067, 'f1': 0.9197176274789681
```

厳密一致（Exact match）は約85％でした。またｆ１値は0.92でした。厳密一致の割合は非常に高く（BERTでは0.15）、正確に応答できることがわかります。

# 参考・謝辞

著者である山田先生およびStudio ousiaさんには感謝いたします

https://arxiv.org/abs/2010.01057
