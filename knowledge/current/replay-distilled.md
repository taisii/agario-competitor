# Replay teacher を使った軽量 bot

## 結論

`replay_distilled` は `semantic_potential` の決定を土台にし、保存済み公式
リプレイの観測状態へ `replay_dominance` を教師として適用して学習した線形残差で、
方向を最大 2.5° だけ補正する。

実行時に `replay_dominance`、先読み、物理シミュレーションは追加しない。追加処理は
16 個の意味方向特徴の生成、固定係数との内積、最大 2.5° の回転だけである。

採用根拠は次の 3 点である。

1. 31 公式リプレイ、41,026 観測を使った 20 試合学習 / 11 試合 holdout で、
   教師方向への残差モデルが semantic 単体より汎化した。
2. 決定論的な公式リプレイ由来クローンを相手にした 2 つの独立 seed 集合、
   合計 8 試合で平均最終質量が `61.39 -> 62.07` に増えた。
3. 累積応答時間は平均 `1.43s -> 1.59s`、最大 `3.05s` であり、
   8 秒制限を十分下回り、semantic と同じ計算量の桁に収まった。

決定論的 8 試合の差は統計的な確定勝利を示すほど大きくない。採用理由は
「再現可能なローカル平均が正で、公式リプレイ上の教師方向にも近づき、
計算量制約も守る」
という複数の証拠が同時に成立したためである。

## 2026-07-16 の再評価

従来の 28 試合比較では、対戦相手に壁時計の累積計算時間で動作モードを変える
戦略が含まれていた。ゲーム乱数と相手選択 seed は固定されていたが、
CPU スケジューリングによって同じ基準方策の最終質量が再実行ごとに変わる条件だった。
この結果は実戦的な相手集合での参考値として残すが、補正角の決定には使わない。

再評価では次を固定した。

- `PYTHONHASHSEED=0`
- `BOT_REPLAY_PROXY_COARSE_AFTER_SECONDS=0`
- 対戦相手は決定論的な公式リプレイ由来クローン 7 体
- 同じ replay 方策を A/B 実行し、最終質量・最大質量・順位・行動統計が一致すること
  を事前確認

2 つの独立 seed 集合、合計 8 試合の結果:

| 補正角 | 平均最終質量 | semantic との差 | 中央値 |
|---|---:|---:|---:|
| 0°（semantic） | 61.387 | — | 60.298 |
| 2.5° | 62.068 | +0.681 | 57.755 |
| 5° | 48.820 | -12.567 | 59.357 |

試合単位の差は大きく、2.5° の優位は統計的に確定していない。一方、5° は
2 つの集合を統合すると大きく下回ったため、教師の寄与が強すぎると判断した。
既定値は 2.5° とする。

出力:

```text
/private/tmp/agario-outcome-distilled-deterministic4-233658
/private/tmp/agario-replay-residual-angle-deterministic4b-234145
/private/tmp/agario-replay-residual-2_5-deterministic4a-234500
```

2.5° 版の提出安全性を既定の累積 8 秒制限で確認した。

| 条件 | 正常終了 | 累積応答時間平均 | decision p99 |
|---|---:|---:|---:|
| replay_dominance 7 体、fast strict | 2/2 | 2.167s | 2.677ms |
| semantic_potential 7 体、公式 `simulation --headless` | 1/1 | 計測なし | 1.606ms |

出力:

```text
/private/tmp/agario-replay-distilled-2_5-strict2-234716
/private/tmp/agario-replay-distilled-2_5-official1-234824
```

## 教師データ

- リプレイ集合:
  `.agario/replays/official/current-submission-49/`
- 試合数: 31
- 自チーム: `team_id=73`
- 観測数: 41,026
- 教師: `ReplayDominanceStrategy`
- 学習: match ID 昇順の先頭 20 試合、26,740 観測
- holdout: 残り 11 試合、14,286 観測
- 特徴:
  semantic の選択方向、前回方向、盤面中央・壁、food、prey、predator、
  neutral、取得可能 virus の合計 16 方向特徴

再現コマンド:

```bash
.venv/bin/python scripts/distill_replay_teacher.py \
  .agario/replays/official/current-submission-49 \
  --team-id 73 \
  --train-matches 20 \
  --every-n 1 \
  --profile-out /tmp/replay-dominance-residual-profile.json \
  --report-out /tmp/replay-dominance-residual-report.json
```

## 教師への汎化

20 試合だけで係数を学習し、未学習 11 試合で評価した。

| 方策 | 角度中央値 | 30°以内 | 90°超 |
|---|---:|---:|---:|
| semantic 単体 | 29.52° | 50.55% | 19.13% |
| 制限なし残差モデル | 25.43° | 56.08% | 8.51% |

制限なし残差は教師再現では改善したが、そのまま実戦へ出すと semantic の安全判断を
大きく変える。この時点の実験 bot では補正量を 5° に制限した。その後の決定論的な
再評価で 5° が悪化したため、現行 bot は 2.5° に制限している。

9 リプレイ、2,358 サンプルへ実際の 5° 制限を適用した結果:

| 方策 | 角度中央値 | 30°以内 | 90°超 | 教師 proxy regret |
|---|---:|---:|---:|---:|
| semantic | 28.82° | 51.06% | 18.96% | 11.659 |
| 5° 残差 | 26.95° | 53.05% | 17.68% | 11.669 |

角度一致は改善した一方、教師の cheap proxy は `+0.010` 悪化した。したがって
教師一致だけでは採用せず、最終質量ベンチマークを採用判断の主指標とした。

## 平均最終質量

固定スナップショット:

```text
/tmp/agario-replay-distilled-angle-20260716-222635
semantic_potential.py sha256:
3023416aeba173f9147c52e182f1c282361dd13725d1f608f4fbc74f2764f3b1
```

対戦相手は各 trial / slot で同じになるよう固定し、次の 6 戦略から選択した。

```text
semantic_lookahead
semantic_potential
replay_dominance
threat_aware_receding_horizon
event_driven_static_search
static_retained_growth
```

### 第 1 集合

- seed: `20260900`
- 16 試合 / 方策

| 方策 | 平均最終質量 | 中央値 |
|---|---:|---:|
| semantic | 29.423 | 5.377 |
| 5° | 33.884 | 16.653 |
| 15° | 34.114 | 2.389 |
| 30° | 22.871 | 1.535 |

### 独立検証集合

- seed: `20261000`
- 12 試合 / 方策

| 方策 | 平均最終質量 | semantic との差 |
|---|---:|---:|
| semantic | 26.787 | — |
| 5° | 27.935 | +1.148 |
| 15° | 21.924 | -4.863 |

### 統合

| 方策 | 試合数 | 平均最終質量 | 中央値 | 30質量以上 |
|---|---:|---:|---:|---:|
| semantic | 28 | 28.293 | 3.854 | 9 |
| 5° | 28 | 31.334 | 10.635 | 9 |
| 15° | 28 | 28.890 | 2.294 | 8 |

目的関数の平均最終質量では 5° が semantic より `+3.041`、約 `+10.8%`。
15° は第 1 集合では僅差で首位だったが、独立集合で下振れし、統合結果でも
5° を下回った。そのため既定値は 5° とした。

ベンチマーク出力:

```text
.agario/benchmarks/replay-distilled-residual-16x/
.agario/benchmarks/replay-distilled-residual-validation-12x/
```

## 実行時間

上の 28 試合を統合した tracked bot の実測:

| 指標 | semantic | 5° 残差 | 比率 |
|---|---:|---:|---:|
| 累積応答時間 平均 | 1.434s | 1.588s | 1.11x |
| 累積応答時間 最大 | 2.440s | 3.048s | 1.25x |
| decision 平均 | 0.496ms | 0.622ms | 1.25x |
| decision p95 平均 | 1.201ms | 1.368ms | 1.14x |
| decision p99 平均 | 1.924ms | 2.192ms | 1.14x |

平均と tail の両方が semantic と同じ桁で、全試合が正常終了した。

その後、提出バンドルへ安全に含めるため、同じ 15 基礎特徴を
`replay_distilled.py` 内で直接計算する実装へ整理した。学習時の特徴関数との一致を
単体テストで確認し、公式リプレイ 250 サンプルでも整理前後の方向・split・reason が
250/250 で一致した。

最終コードを seed `20261200` の 4 試合で再計測した結果:

| 指標 | semantic | replay_distilled |
|---|---:|---:|
| 正常終了 | 4/4 | 4/4 |
| 平均最終質量 | 27.145 | 28.504 |
| 累積応答時間 | 0.930s | 0.908s |
| decision total | 50.7ms | 49.9ms |
| decision p99 | 3.484ms | 1.600ms |

4 試合は強さを再評価するには少ないため、平均最終質量の採用判断には使わない。
最終リファクタ後も計算量が semantic 水準であることの smoke evidence とする。

出力:

```text
.agario/benchmarks/replay-distilled-final-smoke-4x/
```

## 提出バンドル確認

`scripts/build_submission.py` で生成した単一ファイルを、
実際に 8 bot の fast simulation へ渡して 1,400 ターン完走した。

```text
result_type: SUCCESS
submission sha256:
1863192cbf04f983b748f6d2b77089e2bd2e04db3a0c641ceaefdd708898046a
```

単なる `py_compile` では依存定数の欠落を検出できなかったため、
`test_replay_distilled_submission_is_self_contained` は生成モジュールを実際に import
する回帰テストへ強化した。

## 不採用にした案

### 広い tactical probe

捕食者が見える局面へ最大 1 個の tangent 候補を追加する案は、教師 proxy を
改善したが、12 seed のローカル平均最終質量が `semantic 35.78` に対して
`21.40` まで低下した。状態遷移が変わった後の連鎖的な差が大きく、採用しなかった。

### 壁際だけの tactical probe

追跡中の捕食者、壁への衝突、継続方向を条件に絞った案は、同一版比較で
変更 7 件中 1 件だけが教師 proxy を改善し、6 件は悪化した。候補追加方式を
やめ、semantic 方向を主特徴にした残差学習へ切り替えた。

### 純粋な 15 特徴模倣

semantic の決定を特徴へ入れない線形モデルは、11 試合 holdout の 30°以内率が
45.43% で、semantic 単体の 50.55% より低かった。教師を丸ごと近似するには
特徴が不足しており、semantic からの残差だけを学ぶ方式へ変更した。

### 15° / 30° 補正

補正角を増やすほど教師との角度一致は上がったが、30° は第 1 集合で平均最終質量が
semantic を下回った。15° も独立集合で下振れした。この時点では 5° を採用したが、
後の決定論的な再評価では 2.5° が 5° を上回ったため、現在は 2.5° を採用している。

## 未検証の点

- ローカル相手より強い公式環境で平均最終質量が同じ比率で改善する保証はない。
- 現行 2.5° 補正は split の有無を学習しておらず、split 判断は semantic のままである。
- 教師係数は 31 リプレイ集合に固定されている。新しい公式リプレイを追加した場合は、
  時系列 holdout を維持して再学習し、既存結果と条件差を記録する。

## Submission #56（2026-07-19）

`main` の `replay_distilled`（2.5° 補正）を再ビルドし、公式ポータルへ提出した。

- 提出番号: `#56`
- 提出ファイル: `dist/my_bot.py`
- source fingerprint:
  `1c679692d0c52a9aa398f57499601539de9fa74beca7872c773cb8fe9a6bb04f`
- bundle sha256:
  `29725c9e5ca5ce4e2c986797a643b233b43b472211fbc170f8afb2037cd45d20`
- `ruff check .`: 成功
- 全テスト: `472 passed`
- 通常の累積8秒制限による提出前1試合: `SUCCESS`、最終順位 `3/8`

提出直前に稼働していた `#55` は source fingerprint
`2e5b0ce34bc6527bc143b7ebaf12aca128769b0eda9f3a133487cf31c2a57237`
で、古い worktree の `ReplayDominanceStrategy` バンドルと一致した。
`#56` の提出後、ポータル上の Current Submission と source fingerprint が更新され、
`Active` / `In rotation` であることを確認した。提出直後の記録試合数は `0`。

`2026-07-19 21:17 JST` の公式 leaderboard API では次の状態だった。

- 順位: `3/33`
- 平均最終質量: `34.43963028123894`
- 集計試合数: `20`
- `usesRecentMatchesFallback=true`（20 試合の直近試合補正中）
- 1 位との差: `4.33145900543630`（1 位 `38.77108928667524`）

同日 `23:07 JST` には `Ninja` は `3/35`、平均最終質量
`29.347169735209174`、92試合だった。`23:19 JST` 以降は game hub が
`Bot battle is not enabled right now` を表示し、公開 `/teams/leaderboard` も
空配列になった。したがって、この停止期間には提出後10分の順位変化を観測できない。

rotation再開後の `2026-07-20 00:31 JST` は `9/71`、平均最終質量
`30.519298708857864`、9試合、`usesRecentMatchesFallback=true`。これは新候補の
提出結果ではなく、引き続き Submission #56 の新rotation集計である。
