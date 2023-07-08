# ishimoto-NN-repair-break-JSS

CAREだけのreplicationをgitlab上でやっていた．
他の手法の実装も合わせて一つのレポジトリにしたいということでこちらで管理．

## 対象のリペア手法やツール
| 手法名 | 出典 | package URL |
| -------- | -------- | -------- |
| CARE     | ICSE'22   | https://github.com/longph1989/Socrates |
| Arachne   | TOSEM'23 | https://github.com/coinse/arachne |
| AIREPAIR (手法のツール実装) | ICSE'23 (tool demo) | https://github.com/theyoucheng/AIRepair |
| Apricot     | ASE'19  | Arachne, AIREPAIR (どちらもほぼ同じ実装) の実装を利用 |
| (RNNRepair)  | ICML'21  | PAFLの時やったのが使える |


## ディレクトリ構成
```
.
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt (必要なPythonパッケージのリスト, イメージビルド時に参照)
├── data (対象となるデータセットを格納)
│   ├──...
├── models (修正前の学習済みモデルを格納)
│   ├──...
├── notebooks (一時的な結果確認など用のnotebook．最終的に公開するときはpyファイルのみで完結するようにする)
│   ├──...
├── shell (pyファイルをまとめてor複数回実行するための実験スクリプト)
│   ├──...
├── src (直下にはコアとなるrepair手法等のスクリプトを格納)
│   ├──lib (ログやモデル，データセット一般に関する便利関数群）
│   |   ├──...
│   ├──...
├── {repair手法名}_experiments (直下には実験の各プロセスにおける成果物を格納するディレクトリや，設定用のjsonファイルなどを格納)
│   ├──...
└── .gitignore
```
