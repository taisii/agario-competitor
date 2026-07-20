# 保存戦略の強さと教師選定（2026-07-16〜17）

## 結論

現行 commit `cdeac9f` の保存戦略では、平均最終質量を主指標にすると
`replay_distilled`（教師補正上限 2.5°）を第一候補とする。

- 提出用の既定戦略: `replay_distilled`
- 1 本だけ選ぶ行動教師: `replay_distilled`
- 改善用の補助教師:
  - `replay_dominance`: 資産保全、損失回避、探索でしか見つからない候補
  - `semantic_potential`: 通常移動と下側安定性の比較基準
- 教師候補から外す:
  - `expected_final_mass`
  - `outcome_teacher_hybrid`

`replay_dominance` は一部の異種リーグで大勝ちするが、現行
`replay_distilled` との直接対決、広い 17 戦略リーグ、全 slot 巡回リーグの
いずれでも総合首位ではない。単独教師として全面模倣するより、探索由来の
高価値な局面だけを補助ラベルとして使う。

## バージョンを分けた理由

調査開始時の固定 snapshot は commit `2e597af` 相当で、
`replay_distilled` の補正上限は 5° だった。

```text
/tmp/agario-broad-strategy-study-20260716-230609
replay_distilled.py sha256:
a5a91ebc103a3f9df1b9514fd2a0d69dd5ade955124dfb8879b517a986ed100a
```

調査中に main が `cdeac9f` へ更新され、補正上限が 2.5° へ変わった。
最終判断は次の現行 snapshot で取り直した結果を優先する。

```text
/tmp/agario-current-strategy-study-cdeac9f-2355
replay_distilled.py sha256:
e82b970f49b96241c251dfcdcd954de8f10f9804a9976f106bb6ee42e160db30
```

すべての最終比較で `PYTHONHASHSEED=0` を固定した。広い旧版リーグは
hash seed 未固定であり、実運用に近いプロセス差を含む参考値として扱う。

## 今回何をしたか

保存されていた 17 戦略を確認し、単一の対戦形式だけで順位を決めず、次の順序で
評価範囲を広げた。

1. 全 17 戦略から 7 体を抽選する広い異種リーグ
2. `replay_distilled` と `replay_dominance` の 4 対 4 直接対決
3. replay、semantic、探索系を集めた固定強豪リーグ
4. `expected_final_mass` を加えた教師候補リーグ
5. 上位 8 戦略を全 slot へ巡回させる配置均衡リーグ
6. 調査中に main が更新されたため、現行 2.5° 版で主要 3 実験を再実行

採用結果に使った現行版は合計 80 試合で、すべて `SUCCESS` だった。

| 実験 | 試合数 | 確認したこと | 出力 |
|---|---:|---|---|
| 17 戦略広域リーグ | 48 | 異種相手へ 1 枠だけ参加した強さ | `current2_5-broad-slot*-8x-hash0` |
| dominance 直接対決 | 16 | replay 系同士の相対強度と配置反転 | `direct-current2_5-*-8x-hash0` |
| 上位 8 全 slot 巡回 | 16 | 各戦略を全初期 slot へ置いた順位 | `top8-balanced-current2_5-hash0` |

旧 5° snapshot でも、広域 128、直接対決 32、固定強豪 64、教師候補 64 の
合計 288 試合を実行した。旧版の結果は、補正角を強くした場合の失敗と、
各戦略の弱点を知るための参考値として残した。

順位付けの一次指標はプロジェクト方針どおり平均最終質量とした。ただし大勝ち 1 件で
平均が逆転しやすいため、次も同時に確認した。

- 中央値と q25
- 質量 2 未満で終わる割合
- Top1 / Top4 率と平均順位
- slot 0 / 7、または全 slot 巡回による配置差
- 同じ seed の paired 差と bootstrap 区間
- エンジン計測の累積応答時間、decision p99
- metrics に記録された実際の戦略名と補正角

`--throughput` は統計実験を完走するために timeout を緩和するので、
成功件数だけでは提出安全性を判断していない。`response_timings.json` の実測と、
別途行った strict / official smoke を使って 8 秒制約を確認した。

## 現行 2.5° 版の結果

### 広い 17 戦略リーグ

全 17 戦略から 7 体の相手を選び、候補を slot 0 / slot 7 に置いた。
seed `20268000`、各配置 8 試合、合計 16 試合 / 候補。

| 戦略 | 平均最終質量 | 中央値 | Top4 | 平均順位 | 質量 2 未満 | 累積応答平均 |
|---|---:|---:|---:|---:|---:|---:|
| replay_distilled | 28.636 | 2.381 | 75.0% | 3.438 | 43.8% | 1.357s |
| replay_dominance | 22.287 | 1.380 | 56.3% | 4.125 | 56.3% | 3.023s |
| semantic_lookahead | 13.110 | 1.945 | 68.8% | 3.750 | 50.0% | 1.396s |

`replay_distilled - semantic_lookahead` の paired 平均差は `+15.525`、
10 勝 5 敗 1 分、bootstrap の片側 95% 下限は `+3.011`。
slot 0 / 7 の両方で `replay_distilled` の平均最終質量が首位だった。

出力:

```text
.agario/benchmarks/current2_5-broad-slot0-8x-hash0
.agario/benchmarks/current2_5-broad-slot7-8x-hash0
```

### replay_dominance との直接対決

4 体対 4 体を配置順反転で各 8 試合、合計 16 試合実行した。

| 戦略 | 平均最終質量 | 中央値 | 平均順位 | 質量 2 未満 | 累積応答平均 |
|---|---:|---:|---:|---:|---:|
| replay_distilled | 24.792 | 5.767 | 3.828 | 45.3% | 1.205s |
| replay_dominance | 4.956 | 1.243 | 5.172 | 76.6% | 2.347s |

試合内の 4 体平均では `replay_distilled` が 15 勝 1 敗。
同じ seed の二配置を平均した差は `+19.835`、95% CI は
`[+12.530, +25.383]` だった。

出力:

```text
.agario/benchmarks/direct-current2_5-dominance-first-8x-hash0
.agario/benchmarks/direct-current2_5-distilled-first-8x-hash0
```

### 上位 8 戦略の全 slot 巡回

次の 8 戦略を各 slot へ 1 回ずつ置く 8 rotation を、2 engine seed で実行した。

```text
expected_final_mass
replay_distilled
semantic_lookahead
replay_dominance
outcome_teacher_hybrid
semantic_potential
threat_aware_receding_horizon
event_driven_static_search
```

| 戦略 | 平均最終質量 | 中央値 | Top1 | Top4 | 平均順位 |
|---|---:|---:|---:|---:|---:|
| replay_distilled | 31.201 | 1.859 | 31.25% | 62.50% | 3.813 |
| semantic_potential | 27.514 | 2.163 | 25.00% | 68.75% | 3.250 |
| replay_dominance | 21.804 | 1.441 | 12.50% | 37.50% | 4.438 |
| semantic_lookahead | 17.682 | 1.520 | 18.75% | 50.00% | 4.313 |
| event_driven_static_search | 6.447 | 1.370 | 6.25% | 50.00% | 4.688 |
| expected_final_mass | 5.760 | 1.242 | 6.25% | 43.75% | 5.000 |
| outcome_teacher_hybrid | 5.379 | 1.262 | 0.00% | 37.50% | 5.313 |
| threat_aware_receding_horizon | 5.232 | 1.324 | 0.00% | 50.00% | 5.188 |

平均最終質量では `replay_distilled` が首位。ただし
`semantic_potential` との差 `+3.688` の 95% CI は
`[-32.01, +39.39]` で、この 16 試合だけでは差を確定できない。
seed ごとの首位も `replay_distilled` と `replay_dominance` に分かれた。

累積応答時間は `replay_distilled` 平均 `1.081s`、p95 `1.681s`、
最大 `2.211s`。全 128 player-match の metrics で戦略名を照合し、
教師補正が最大 2.5° であることも確認した。

出力:

```text
.agario/benchmarks/top8-balanced-current2_5-hash0
```

## 2.5° を支持する既存の決定論的比較

公式リプレイ由来の決定論的クローンを相手にした既存 8 試合では次の結果だった。

| 補正角 | 平均最終質量 |
|---|---:|
| 0°（semantic_potential） | 61.387 |
| 2.5° | 62.068 |
| 5° | 48.820 |

2.5° の優位は小さいが、5° の教師補正は強すぎる。今回の広域・直接・全 slot
巡回でも現行 2.5° が首位になったため、既定値 2.5° を維持する。

## 旧 5° 版から得た参考知見

旧 snapshot では次を実行した。

- 全 17 戦略の広いリーグ: 128 試合
- replay_distilled / replay_dominance 4v4: 32 試合
- 固定強豪リーグ: 64 試合
- expected_final_mass を含む教師候補リーグ: 64 試合

主な結果:

- 旧 5° の直接対決でも distilled が dominance に 26 勝 6 敗。
- 固定強豪リーグでは distilled が平均最終質量首位。
- 広い 1 枠リーグでは semantic_lookahead が首位で、旧 5° distilled は 3 位。
- expected_final_mass は独立教師候補リーグで最下位。
- local_tactical_search は相手側集計で平均最終質量 `4.57`、
  累積応答平均 `9.03s`、8 秒超過 40/64 と弱く遅かった。

5° 版は現行戦略ではないため、採用順位には使わない。教師補正を強くしすぎると
広い異種環境で semantic の安全性を壊す、という失敗知見として残す。

## 何がダメだったか

### 評価基盤でダメだったこと

#### Python hash seed を固定していなかった

初期の広域実験ではゲーム seed と相手抽選 seed は固定していたが、
`PYTHONHASHSEED` は固定していなかった。さらに一部戦略は壁時計の累積時間で
探索モードを変えるため、CPU スケジューリングでも状態遷移が変わり得た。

この条件は実運用の揺らぎを見る参考にはなるが、細かな A/B 差や補正角の決定には
使えない。最終比較では `PYTHONHASHSEED=0` を明示し、固定 snapshot から起動した。
`scripts/benchmark_simulations.py` にも hash seed の既定値を追加済みである。

#### throughput の成功を本番 timeout 合格と解釈できない

`--throughput` は全 player の累積 timeout を 600 秒へ緩和する。したがって
`SUCCESS` でも本番の累積 8 秒を超えている場合がある。

実際、旧広域リーグでは `local_tactical_search` の累積応答平均が `9.03s`、
64 player-match 中 40 件が 8 秒超過だった。`replay_dominance` や
`outcome_teacher_hybrid` にも 8 秒超過があった。一方、現行
`replay_distilled` は主要実験で平均約 `1.1〜1.4s`、最大 `2.33s` だった。

#### entry ファイル名を戦略名だと仮定した

最初の上位 8 巡回では `event_driven_static_search.py` が存在せず、bot 起動失敗から
全試合が無効になった。

次に既存 entry への symlink を目的のファイル名で作ったが、Python の `__file__` は
symlink の target 実体名を返した。その結果、`event_driven_static_search` のつもりで
`expected_final_mass` を重複起動していた。

修正後は 3 行の entry wrapper を目的の実ファイル名で作り、smoke の
`bot_metrics.jsonl` で `strategy=event_driven_static_search` を確認してから本番を
開始した。今後もコマンドのファイル名ではなく、metrics の実行戦略名を検証する。

#### 調査中の main 更新を同じ戦略として混ぜかけた

調査開始時の `replay_distilled` は補正上限 5°、途中で更新された現行版は 2.5°。
同じ名前でも方策は異なる。旧 snapshot のまま正しい entry で走らせた v3 も、
現行順位には混ぜなかった。

長時間評価では開始時に commit、戦略ファイル hash、主要設定値を保存し、
main が更新された場合は新 snapshot で主要パネルを取り直す必要がある。

#### 1 つの配置や対戦生態だけでは順位が反転した

固定強豪リーグでは `replay_distilled` が slot 0 で平均 `0.912`、slot 7 で
`54.218` となるなど、全候補で極端な配置差があった。4 対 4 の直接対決も
同じ戦略が 4 体いる特殊な生態であり、公式の 1 枠参加と同一ではない。

そのため、次のどれか一つだけで最強を決めなかった。

- slot 0 だけ
- 直接対決だけ
- Top1 回数だけ
- 同じ seed 集合だけ

最終判断では、異種 1 枠の広域リーグ、配置反転した直接対決、全 slot 巡回の
3 パネルで平均最終質量が首位かを確認した。

#### 相手側の重複観測を独立試合として数えられない

広域リーグでは候補 variant が違っても、同じ trial / slot の random opponent が
同じ戦略になる。相手側戦略の 4 variant 分は独立した抽選ではない。

相手戦略の参考順位では生の player-match 数だけを信頼せず、
`trial / slot` を cluster として平均した。`expected_final_mass` が初期集計で
暫定首位になったのも、独立配置が少なく大勝ち 1 cluster の影響が大きかったためで、
追加 seed では教師候補最下位になった。

### 戦略としてダメだったこと

#### replay_distilled 5°

旧 5° 版は dominance との直接対決や固定強豪では強かったが、広い異種 1 枠では
`semantic_lookahead` に負けた。決定論的クローン 8 試合でも平均最終質量が
`48.820` で、0° の `61.387` と 2.5° の `62.068` を下回った。

教師方向へ近づける量が大きいほど良いわけではない。semantic の安全判断を壊さない
2.5° まで縮めた結果、現行広域リーグでも首位になった。

#### replay_dominance

探索により大勝ちを作る試合はあるが、中央値、低質量率、速度が安定しない。
現行 distilled との 4 対 4 では 1 勝 15 敗、平均最終質量
`4.956` 対 `24.792`。単独の通常行動教師には向かない。

ただし探索でしか出ない候補には価値があるため、distilled と大きく異なる局面だけを
outcome で選別する補助教師として残す。

#### expected_final_mass

旧広域の相手側暫定集計では平均が高く見えたが、独立 cluster が少なく、
大勝ち外れ値の影響だった。hash 固定の教師候補リーグでは平均 `12.721`、
Top1 `6.3%`、低質量率 `68.8%` で最下位。上位 8 全 slot 巡回でも平均 `5.760`。

名前と設計目的だけで教師候補にせず、独立 seed と配置均衡で再検証する必要がある。

#### outcome_teacher_hybrid

semantic が見つけた局所捕食候補を dominance の proxy で採用しても、その後
1,000 ターン以上の分岐を評価できない。保存リプレイ比較では行動変更した試合が
`+79.198` と `-129.781` に分かれ、分散が大きすぎた。

さらに二つの重い方策を実行する初期版は timeout した。prefilter で速度問題を
抑えても、現行全 slot 巡回の平均は `5.379` であり、提出・通常教師の双方で不採用。

#### local_tactical_search

旧広域リーグの相手側 cluster 集計で平均最終質量 `4.57`、
低質量率 `73%`、累積応答平均 `9.03s`、最大 `17.14s`。
局所探索の計算量に対して、長期の最終質量へつながる行動を選べていない。

削除した `local_tactical_search_reference` と同様に、広い探索をそのまま提出方策へ
持ち込む方向は採らない。必要ならオフライン oracle として使い、軽量方策へ
局面限定で蒸留する。

## 教師としての使い分け

### 1 本だけ選ぶ場合

`replay_distilled` を選ぶ。

平均最終質量で、現行広域リーグ、現行直接対決、現行全 slot 巡回のすべてで首位。
速度も semantic 系と同じ桁で、8 秒制限に余裕がある。

### 改善用に複数教師を使える場合

`replay_distilled` を通常ラベルとし、次の比較ラベルを追加する。

- `replay_dominance`:
  distilled と大きく異なる探索候補だけを抽出し、将来の最終質量、死亡、
  lost mass で採否を決める。
- `semantic_potential`:
  distilled の補正で方向が変わった局面の安全側比較教師にする。

教師への方向一致そのものは目的にしない。採用ラベルは replay outcome、
平均最終質量、死亡、lost mass で検証する。

## 除外した実行結果

上位 8 巡回の初回試行には entry 起動上の不備があったため使用しない。

- `top8-balanced-hash0`: `event_driven_static_search` の entry 不在で起動失敗
- `top8-balanced-hash0-v2`: symlink の実体名解決により
  `expected_final_mass` を重複起動
- `top8-balanced-hash0-v3`: entry は正しいが旧 5° snapshot

正式な集計対象は `top8-balanced-current2_5-hash0` だけである。

## 次回の再評価手順

1. 開始 commit と対象戦略ファイルの SHA-256 を記録する。
2. `PYTHONHASHSEED=0` を設定する。
3. 壁時計で方策を変える戦略は、比較目的に応じて時間 fallback を固定する。
4. 1 試合 smoke 後、metrics の `strategy`、主要設定値、補正角を読む。
5. 候補を少なくとも二つの離れた slot に置く。
6. 上位候補は全 slot 巡回を行う。
7. 平均最終質量を一次指標にし、中央値、q25、低質量率、順位も併記する。
8. paired 差は同じ seed / 対戦構成だけで計算する。
9. throughput の結果と strict 8 秒検証を分ける。
10. main が変わった場合は結果を混ぜず、新 snapshot で主要パネルを再実行する。
11. entry や shim を使った場合は、実行 metrics の戦略名を全件検査する。
12. 失敗した出力は消さず、無効理由と正式出力を knowledge に記録する。

## 限界

- 最終質量の分散と slot 依存は非常に大きい。
- 全 slot 巡回でも engine seed は 2 個であり、上位 2 戦略の差は確定していない。
- ローカル相手は公式環境より弱い可能性がある。
- 新しい公式リプレイを取得した場合は、行動一致ではなく outcome 指標も再計測する。
- `replay_distilled` を教師として再蒸留するだけでは新しい行動を増やせない。
  改善時は dominance の探索候補や公式強者リプレイを追加データ源にする。
