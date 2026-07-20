# semantic / replay_dominance の結果ベース教師

## 目的

平均最終質量を目的関数とし、`semantic` と `replay_dominance` の役割を
死亡、中盤成長、捕食、損失、最終状態から決める。

## 同一 seed の保存リプレイ 4 組

ランダム相手との同一 seed 4 組を `scripts/compare_strategy_outcomes.py` で集計した。
質量は `radius²`。捕食質量と損失質量は player fragment のイベント合計であり、
food、virus、decay は差し引かない。

| 指標 | replay_dominance | semantic |
|---|---:|---:|
| 平均最終質量 | 17.936 | 8.156 |
| 生存率 | 100% | 75% |
| 序盤平均質量 | 1.802 | 1.461 |
| 中盤平均質量 | 6.156 | 5.183 |
| round 700 質量 | 6.276 | 4.184 |
| 捕食数 | 14.5 | 30.5 |
| 捕食質量 | 16.262 | 22.932 |
| 失った fragment 数 | 18.5 | 21.5 |
| 失った質量 | 24.112 | 31.857 |
| 捕食質量 - 損失質量 | -7.849 | -8.925 |
| virus 数 | 12.75 | 8.75 |

確認できた事実は、semantic は捕食回数と gross 捕食質量を増やす一方、
損失と死亡も増え、最終質量を残せていないこと。replay_dominance は捕食数が
少なくても、序盤・中盤の質量、virus、生存、最終質量で上回った。

したがって教師の役割は次のように定義する。

- semantic: 捕食候補の発見と、軽量な基準方向
- replay_dominance: 損失回避、資産保全、方向補正の教師
- 最終採用判断: 平均最終質量
- 診断指標: 死亡、中盤質量、捕食数、捕食質量、失った質量

## 実行時ハイブリッド

`outcome_teacher_hybrid` では、semantic が見つけた短距離の孤立 prey 方向を
replay_dominance の候補集合へ追加し、replay の proxy で採否を決めた。

### 不採用理由

決定論的な公式リプレイ由来クローン相手 4 seed:

| 方策 | 平均最終質量 |
|---|---:|
| replay_dominance | 58.804 |
| outcome_teacher_hybrid | 46.158 |

semantic 候補で盤面が変わった 2 試合の差は `+79.198` と `-129.781`。
局所 proxy で良い捕食でも、その後 1,000 ターン以上の状態分岐を安定して
評価できず、分散が大きすぎた。

また、semantic を毎ターン実行する初期版は累積時間制限に達した。
現在の実験実装は、試合 45% 以前、自分質量 8 以下、単一 blob、可視敵 1 体、
捕食可能な場合だけ semantic を起動するため時間問題は解消しているが、
平均最終質量が replay 単体を下回るため提出候補にはしない。

## 採用する構造

実行時に二つの重い方策を切り替えず、semantic の方向へ、公式リプレイ上で
replay_dominance を教師として学習した線形残差を小さく加える。

決定論的 8 seed の補正角比較:

| 補正角 | 平均最終質量 |
|---|---:|
| 0°（semantic） | 61.387 |
| 2.5° | 62.068 |
| 5° | 48.820 |

よって `replay_distilled` の既定補正は 2.5° とする。2.5° の改善幅は小さく、
試合差の分散も大きいため、公式結果で悪化した場合は semantic へ戻す。

## 再現コマンドの要点

強さ比較では壁時計による方策変化を止め、決定論的クローンを相手にする。

```bash
env BOT_REPLAY_PROXY_COARSE_AFTER_SECONDS=0 \
  .venv/bin/python scripts/benchmark_simulations.py \
  --throughput --trials 4 --jobs 4 \
  --variants \
    semantic:BOT_STRATEGY=semantic_potential \
    residual2_5:BOT_STRATEGY=replay_distilled,REPLAY_DISTILLED_MAX_CORRECTION_DEGREES=2.5 \
    residual5:BOT_STRATEGY=replay_distilled,REPLAY_DISTILLED_MAX_CORRECTION_DEGREES=5 \
  --submission \
    1:bots/entries/semantic_potential.py \
    7:bots/entries/replay_opponent.py
```

提出安全性はこの throughput 実験とは分け、既定の 8 秒累積制限で確認する。
