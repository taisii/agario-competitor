# Submission #53: 公式最新30試合とローカル30試合の差

## 結論

現在提出中の Submission #53 は、ローカルより公式戦で平均最終質量が
`53.23 -> 35.21`（`-33.9%`）に下がった。

主因として確認できた差は、成長源の配分ではなく player-to-player の資産保全である。

- gross mass の発生源比率は、公式とローカルでほぼ同じ。
  - 敵: `41.17%` vs `40.82%`
  - virus: `45.59%` vs `46.97%`
  - food: `13.24%` vs `12.21%`
- 公式では捕食 fragment 数が `-24.9%`、捕食質量が `-11.5%`。
- 公式では被捕食 fragment 数が `+54.3%`、被捕食質量が `+178.4%`。
- 被捕食 1 fragment の平均質量は、公式 `1.970`、ローカル `1.092`。
  公式では約 `80%` 大きい塊を失っている。
- 捕食質量に対する被捕食質量の比率は、公式 `55.6%`、ローカル `17.7%`。
- 最終100 round に停止していた試合は両方 `0/30`。ゼロ方向命令も `0%`。

したがって、次の改善対象は food の比率変更ではなく、大きい fragment を敵へ渡す局面、
特に virus 後の分散、split・追跡後、第三者がいる局面の資産保全である。

## 対象データ

### 公式

- ポータル: `https://syncs.org.au/competition2026/game/history`
- Submission: `#53` (`submissionId=1937`)
- 提出日時: `2026-07-17T10:04:35.441Z`
- チーム: `team_id=73` (`Ninja`)
- 提出コード fingerprint:
  `1c679692d0c52a9aa398f57499601539de9fa74beca7872c773cb8fe9a6bb04f`
- 現在の `dist/my_bot.py` と fingerprint が一致。
- 最新の Successful 30 試合: match `40736`〜`40809`
- 試合時刻: `2026-07-17T11:25:08.992Z`〜`11:34:36.254Z`
- 保存先: `.agario/replays/official/current-submission-53-latest-30/`
- 30/30 JSON が `event_game_started` で始まり、`event_player_won` で終わることを確認。
- 全試合 Engine `2026.1.15`、1400 round。

### ローカル

- 同じ `dist/my_bot.py` を player 0 に配置。
- 相手は `bots/entries/random_replay_opponent.py` から毎試合7体。
- Engine `2026.1.15`、通常の累積8秒制限。`--fast` / `--throughput` は不使用。
- seed `20260717`、30 trials、30/30 `SUCCESS`。
- 保存先:
  `.agario/benchmarks/current-submission-53-local-replay-opponents-30/`

実行コマンド:

```bash
.venv/bin/python scripts/benchmark_simulations.py \
  --trials 30 \
  --jobs 4 \
  --variants current \
  --submission 1:dist/my_bot.py 7:bots/entries/random_replay_opponent.py \
  --tracked-slots 0 \
  --metrics-every-n 100 \
  --random-seed 20260717 \
  --workspace-root \
    .agario/benchmarks/current-submission-53-local-replay-opponents-30
```

## 集計定義

- 質量: `radius²`
- 捕食回数: 敵 player fragment を食べた `event_player_eaten` 数
- 被捕食回数: 自分の fragment が食べられた `event_player_eaten` 数
- 全滅回数: 最後の fragment を失い `eaten_player_alive=false` になった回数
- elimination: 敵の最後の fragment を食べた回数
- resource 比率: decay や後の被捕食を引く前の gross acquired mass の比率
- 終盤: 最後の100 round
- 停止: 終盤の80%以上で生存し、mass center の移動距離が1.0未満
- 最終死亡: round 1399 の snapshot で alive=false

このゲームは全滅後30 round で respawn する。命令が30 round 連続でない区間は
`RESPAWN_DELAY_ROUNDS=30` と一致しており、bot の停止とは数えていない。

再現コマンド:

```bash
.venv/bin/python scripts/compare_official_local_replays.py \
  .agario/replays/official/current-submission-53-latest-30 \
  .agario/benchmarks/current-submission-53-local-replay-opponents-30 \
  --official-team-id 73 \
  --local-player-id 0 \
  --output \
    .agario/analysis/submission-53-official-vs-local-latest-30.json
```

## 結果

### 最終結果

| 指標 | 公式30 | ローカル30 | 公式 - ローカル |
|---|---:|---:|---:|
| 勝率 | 53.3% | 66.7% | -13.3pt |
| 生存率 | 93.3% | 96.7% | -3.3pt |
| 平均最終質量 | 35.210 | 53.232 | -18.022 (-33.9%) |
| 最終質量中央値 | 37.373 | 60.518 | -23.145 |
| 最終質量 q25 | 1.793 | 34.091 | -32.298 |
| 最終質量 2 未満 | 36.7% | 13.3% | +23.3pt |
| 平均 peak 質量 | 48.187 | 59.328 | -11.141 |

平均だけでなく q25 が大きく下がっている。公式では「勝てる試合」はある一方、
小さい質量まで失う試合がローカルより多い。

### 捕食・被捕食

回数は fragment event。全滅しても30 round 後に復活するため、全滅回数は
試合終了時の死亡とは異なる。

| 指標（1試合平均） | 公式30 | ローカル30 | 公式差 |
|---|---:|---:|---:|
| 捕食 fragment 数 | 25.433 | 33.867 | -24.9% |
| 捕食質量 | 44.300 | 50.064 | -11.5% |
| elimination 数 | 12.600 | 17.567 | -28.3% |
| 被捕食 fragment 数 | 12.500 | 8.100 | +54.3% |
| 被捕食質量 | 24.623 | 8.845 | +178.4% |
| 全滅回数 | 1.933 | 1.300 | +48.7% |
| 捕食質量 - 被捕食質量 | 19.678 | 41.219 | -52.3% |

| 質量効率 | 公式30 | ローカル30 |
|---|---:|---:|
| 捕食 1 fragment 当たり質量 | 1.742 | 1.478 |
| 被捕食 1 fragment 当たり質量 | 1.970 | 1.092 |
| 被捕食質量 / 捕食質量 | 55.6% | 17.7% |

公式では捕食する fragment はやや大きいが、失う fragment はさらに大きい。
単に接触回数が増えたのではなく、高価値な塊を敵へ渡すことが最終質量を下げている。

### food・virus・敵の割合

| 発生源 | 公式 count/試合 | ローカル count/試合 | 公式 gross mass share | ローカル gross mass share |
|---|---:|---:|---:|---:|
| food | 633.267 | 665.267 | 13.24% | 12.21% |
| virus | 21.800 | 25.600 | 45.59% | 46.97% |
| enemy | 25.433 | 33.867 | 41.17% | 40.82% |

割合は似ているが、公式では取得量が全発生源で下がった。特に enemy fragment は
`-24.9%`、virus は `-14.8%`、food は `-4.8%`。food を取る比率へ大きく
偏ったのではなく、強い相手の中で敵・virus の取得機会を実現できていない。

### 最後100 round

| 指標 | 公式30 | ローカル30 |
|---|---:|---:|
| 停止した試合 | 0/30 (0%) | 0/30 (0%) |
| ゼロ方向命令 | 0% | 0% |
| 終了時死亡 | 2/30 (6.7%) | 1/30 (3.3%) |
| 被捕食があった試合 | 7/30 (23.3%) | 6/30 (20.0%) |
| 終盤被捕食質量 / 全被捕食質量 | 12.56% | 13.42% |
| 終盤被捕食質量 / 試合 | 3.092 | 1.187 |

終盤の停止は確認できない。終盤に被捕食した試合率と、全損失に占める終盤比率も
大差はない。一方、終盤に失った絶対質量は公式が約2.6倍。公式との差は
「最後だけ止まる」問題ではなく、試合全体を通じて大きい fragment を失う問題である。

## 事実と未検証の仮説

確認済み:

- 公式とローカルの engine version と round 数は同じ。
- 現在の提出コードとローカル試験コードの fingerprint は同じ。
- 成長源の mass share はほぼ同じ。
- 公式は捕食量が減り、被捕食数と被捕食質量が増える。
- 公式の被捕食 fragment はローカルより大きい。
- 終盤停止はない。

未検証の仮説:

- 公式の相手は replay clone より、virus 後や split 後の分散した大 fragment を
  回収する能力が高い。
- 追跡で得る質量より、第三者へ見せる大 fragment のリスクを過小評価している。
- `replay_distilled` の2.5度補正そのものより、基底の semantic policy が
  aggregate enemy risk を弱く見積もる局面が損失の中心である。

次に検証する場合は、公式375件・ローカル243件の被捕食 event を、直前40 round の
virus、split、fragment 数、可視 predator 数、壁距離、追跡中かで分類する。
採用判断は、損失イベントの再現だけでなく、同 seed の平均最終質量で行う。

## 2026-07-19: 保全層と優勢時 snowball 層

公式30試合の全イベントを再集計すると、勝利16試合は平均最終質量 `62.390`、
平均 peak `69.736`、捕食質量 `69.508`、被捕食質量 `22.174` だった。敗戦14試合は
平均最終質量 `4.147`、平均 peak `23.559`、捕食質量 `15.492`、被捕食質量
`27.422`。損失抑制と上側成長は別の目的として扱う必要がある。

### AssetPreservationLayer

方向を最大30度書き換える初期案は、独立 cloud-pressure seed で最終質量
`120.197 -> 0.0` と `94.802 -> 61.630` の破壊的な悪化を起こした。前者には
確定捕食 `8.43` を予測損失 `7.83` のために捨てる介入が含まれ、後者は round
1247 の方向変更が終盤成長を壊した。

採用版は方向を変更せず、実際の split 後形状と同方向 non-split を3 turnで比較する
split veto に限定した。保存対象は総質量の5%以上、確定1-step取得質量が保存量を
覆う場合は veto しない。独立16試合と既知の悪化seedを含む比較では現行と全件同値で、
安全性を壊す挙動変更は確認されなかった。

### SnowballCaptureLayer

勝利16試合の観測状態で、基底戦略が選ばなかった即時 split capture 候補を調べた。
重複追跡roundを含め173局面で1-step捕食と自fragment損失0を予測し、146局面は
捕食後の可視敵に対する safety margin が0以上だった。この件数は同じ獲物を複数round
数えるため、そのまま追加捕食数とは解釈しない。

偽陽性を抑えるため、別層 `bots/strategies/snowball_capture.py` は次を全て満たす場合だけ
基底判断を即時 split capture へ変更する。

- 現在1位、総質量20以上、own fragment 4以下
- 1-step遷移で敵質量1以上かつ総質量の5%以上を確定取得
- 同遷移で own mass lost が0
- 捕食後 safety margin が通常 reserve 3以上
- semantic の split candidate score が基底選択から0.5以内
- 最終的に AssetPreservationLayer の3-turn split vetoも通る

公式30試合を観測状態のまま通した counterfactual では、3勝利試合の4局面だけが発火した。
予測取得質量は `4.946`, `2.818`, `2.334`, `2.593`、取得比率は
`13.74%`, `6.02%`, `6.99%`, `6.71%`。全て AssetPreservationLayer を通過した。
ただし replay は行動後の世界を分岐再生できないため、最終質量の増加は未確認の仮説である。

cloud-pressure proxy の同一seed比較は `20261720` の8試合と `20260712` の16試合、
合計24試合で全件 baseline と最終質量が同値だった。局面条件が発生せず、安全側の
非介入は確認できたが、成長効果のローカル実証にはなっていない。提出バンドルの通常
8秒制限 smoke は `SUCCESS`、1位、最終質量 `99.076`。`ruff`、`493 tests`、
`py_compile` に合格した。バンドル sha256 は
`55383e9910944d413fb3a39659f540d7e40fbec7396b79662a63656905e09fc0`。

その後、異種相手を抽選する独立16 seed（`20268000`）で実際に発火させたところ、
平均最終質量は `23.879 -> 22.403`（paired `-1.477`）、平均順位は
`3.688 -> 4.812` へ悪化した。個別差も `+157.538` から `-109.255` と極端で、
1-step確定捕食と通常 safety reserve だけでは分裂後の長期回収を保証できない。
よって `SnowballCaptureLayer` は提出経路から外し、実験実装としてのみ残した。
次に試すなら split 後の再結合までの retained-mass、第三者の将来流入、同じ prey を
non-split でも数turn以内に取れる機会費用まで比較する必要がある。

さらに分裂を一切せず、基底方向から15度以内で1-step捕食を確定できる場合だけ
方向を補正する contact 版を試した。公式勝利replayでは2件（予測取得質量
`1.092`, `4.004`）の取り逃し候補があった。異種16 seedでは平均最終質量が
`16.158 -> 24.479`（paired `+8.322`）へ上がった一方、個別差に
`-104.935`, `-94.954` があり、one-sided 95% lower bound は `-13.863`。
上側改善より破壊的下振れを重く見て、この版も提出経路から外した。局所的な確定取得は
最終質量の確定改善ではない。上積み層は単発postprocessorでなく、その後の行動系列を
含む outcome label から蒸留する必要がある。

GPT Proが提案した recovery residual guard（merge cooldown 9以上の分裂回復中かつ
safety margin 3未満では2.5度のteacher residualを抑止）も実装してfresh異種16 seedで
評価した。平均最終質量は `35.610 -> 36.948`（paired `+1.338`）だったが、個別差に
`-116.696` があり、one-sided 95% lower bound は `-14.518`。Top1率も
`31.2% -> 25.0%`。Pro自身が示したfresh 64 seed・95%下限>0という採用条件を満たさず、
最終提出から外した。

また `scripts/analyze_loss_contexts.py` の物理定数が旧値だったため、engine
2026.1.15 の `EAT_SIZE_RATIO=1.2`, `SPLIT_MIN_MASS=2.0`,
`SPLIT_EJECT_SPEED=1.6` に修正して再集計した。event由来の主要割合は変わらない。
修正後の `multiple_predator_players` は86/375 events（22.93%）、lost mass
59.559（8.06%）。近傍predator分類は修正版JSONを参照する。

## 2026-07-19: cloud-pressure 再現環境

最新30公式戦で team 73 に与えた被捕食質量を相手team別に集計した。上位は
team 24 `120.184`、85 `119.467`、1 `114.862`、35 `107.479` で、この4チーム
だけで全被捕食質量の `62.54%` を占める。4チームの replay imitation profile は
全て held-out validation 不合格であり、公式ラインナップをそのまま clone にしても
圧力を再現できないことを確認した。

評価用に `bots/entries/cloud_pressure_opponent.py` を追加した。公式30試合の7-team
cohortは維持しつつ、team 73へ20以上の被捕食質量を与えた12チームだけを
`ReplayDominanceStrategy` へ置換する。1 lineup 当たりの強圧力数は平均 `2.67`。
これは提出戦略ではなく、模倣失敗を補う評価fixtureである。

同一 seed `20260719` の通常ランナー12試合:

| 指標 | 公式30 | calibrated cohort 12 | all-dominance 12 |
|---|---:|---:|---:|
| 勝率 | 53.3% | 50.0% | 41.7% |
| 平均最終質量 | 35.210 | 52.225 | 35.244 |
| q25 | 1.793 | 1.014 | 1.569 |
| 質量2未満 | 36.7% | 41.7% | 33.3% |
| 捕食fragment/試合 | 25.433 | 36.667 | 25.333 |
| 被捕食fragment/試合 | 12.500 | 13.417 | 11.667 |
| 被捕食質量/試合 | 24.623 | 20.799 | 18.030 |
| 全滅/試合 | 1.933 | 2.833 | 2.500 |

all-dominance の別 seed `20260730` 30試合では、平均最終質量 `31.109`、捕食
fragment `24.433`、捕食質量 `43.397`、被捕食fragment `13.967`、被捕食質量
`24.251` となり、公式の主要な流量へ近づいた。一方、勝率 `23.3%`、全滅
`3.467/試合` で圧力過多だった。

このため単一fixtureをクラウドの完全再現とは扱わない。今後の採用ゲートは、
calibrated cohortで公式に近い勝率・低質量率・構成を確認し、all-dominanceで
捕食/被捕食流量と強圧力耐性を確認する二面評価とする。

出力:

- `.agario/analysis/submission-53-vs-cloud-pressure-calibrated-v3-12.json`
- `.agario/analysis/submission-53-vs-cloud-pressure-dominance-12.json`
- `.agario/analysis/submission-53-vs-cloud-pressure-dominance-30.json`

## 2026-07-20: exact non-split prey scoring（不採用）

`semantic_potential` は split/virus candidateだけ exact one-step outcomeを採点し、
通常移動のprey接触は概算のみだった。この欠落を補うfeature flag
`SEMANTIC_EXACT_PREY_OUTCOME` を追加して評価した。

全prey candidateへ適用した最初の24 paired seedでは平均 `+31.242`、one-sided
95% lower `+11.605` だったが、fresh 32では平均 `+0.453`、lower `-16.269`。
長い追跡まで状態を変え、局所取得を二重加点していた。

次に1-turn以内の接触だけへ限定し、概算と確定質量を加算せず最大値へ置換した。
24 paired seedで平均 `-6.601`、lower `-16.110`。さらにrank 1かつ総質量20以上へ
限定すると最初の24 seedは平均 `+3.217`、lower `+0.230` だったが、fresh 10 pair
時点で `-119.846` の破壊的下振れが出たため中止した。

結論として、即時確定捕食を既存candidate scoreへ入れるだけでも長期優位は保証されない。
flagは既定OFFとし、Submission候補へは含めない。次の上側改善は、捕食後から再結合までの
retained massまたはmatch-level outcome labelを必要とする。

### safety reserve gate の追加検証

fresh集合で `-119.846` になった trial 6 を全round telemetryで再現した。最初の分岐は
round 939で、総質量 `112.154`、1位、16 fragmentの状態だった。現在の safety margin は
`-2.470` と既に危険域だったが、exact prey版は敵質量 `1.160` の即時取得を加点して、
基準の選択より selected margin が悪い方向へ進み、最終質量は `120.718 -> 0.872` へ
崩壊した。

そこで `SEMANTIC_EXACT_PREY_REQUIRES_SAFETY_RESERVE` を追加し、現在の safety margin が
通常 reserve `3.0` 以上の時だけ exact prey outcome を有効化した。同じ trial 6では
round 939の介入が消え、最初の実介入は脅威のいないround 1116（非分裂で敵質量
`1.347` を確定取得、方向差 `0.644` 度）へ移った。最終質量は `184.097` となった。

しかし独立seed全体では安全性を証明できなかった。

| paired集合 | 試合数 | baseline平均 | candidate平均 | 平均差 | 片側95%下限 | 最悪差 |
|---|---:|---:|---:|---:|---:|---:|
| screen `20261800` | 24 | 41.544 | 44.761 | +3.217 | +0.209 | 0.000 |
| fresh `20261900` | 32 | 44.338 | 46.654 | +2.315 | -2.545 | -36.105 |

fresh集合ではほかに `-30.180`, `-18.507` の下振れもあった。現在時点の安全だけでは、
取得後に生じる将来の軌道差を拘束できない。よって safety gate版も既定OFF・提出不採用と
する。局所接触の採否には、数round後までの retained mass と将来の threat exposure を
直接評価する必要がある。

出力:

- `.agario/benchmarks/exact-prey-safe-screen-24/mass_comparisons.json`
- `.agario/benchmarks/exact-prey-safe-fresh-32/mass_comparisons.json`
- `.agario/diagnostics/ahead-contact-fast-seed6-baseline/submission0/bot_metrics.jsonl`
- `.agario/diagnostics/ahead-contact-fast-seed6-safe/submission0/bot_metrics.jsonl`

## 2026-07-20 00:49 JST: leaderboard再確認

公開 `/teams/leaderboard` で `Ninja` は `7/71`、平均最終質量
`32.63358996684134`、20試合、`usesRecentMatchesFallback=false`。00:31 JSTの
`9/71`、`30.519298708857864`、9試合、fallback trueから20試合集計へ進んだ。
新candidateの提出成功は確認されておらず、引き続きSubmission #56の結果である。

## 2026-07-20: scoped forward exact-prey layer（次提出候補）

### 失敗原因の訂正

先の exact-prey 実験が不安定だった主因は、将来予測以前に採点契約の実装が崩れていた
ことだった。非分裂候補を exact one-step world で採点した際、捕食済みの prey を
directional potential から除去していたため、exact 評価が非分裂候補を基準より減点し、
相対的に split candidate を勝たせていた。fresh `20261900` の旧下振れ3件
`-36.105`, `-30.180`, `-18.507` は、全て最初の分岐でこの層が `split=True` を
新規選択していた。

修正後は通常の非分裂scoreを保持し、exact transitionは確定敵質量が既存intentを
上回る場合の非負 uplift としてだけ使う。したがって exact 層は候補を減点しない。
さらに次の責務境界を追加した。

- 現在1位、総質量20以上
- current safety marginが通常reserve `3.0` 以上
- 接触まで1 turn以内
- 現在の進行方向から90度以内
- exact層が基準のsplit/non-split選択を変更する場合は基準選択を維持

最後の条件により、split判断は既存semantic + AssetPreservationLayerへ任せ、exact層は
非分裂内の方向選択だけを改善する。feature設定は以下で、次候補では全て既定値である。

```text
SEMANTIC_EXACT_PREY_OUTCOME=1
SEMANTIC_EXACT_PREY_AHEAD_ONLY=1
SEMANTIC_EXACT_PREY_MIN_MASS=20
SEMANTIC_EXACT_PREY_REQUIRES_SAFETY_RESERVE=1
SEMANTIC_EXACT_PREY_MAX_TURN_DEGREES=90
SEMANTIC_EXACT_PREY_PRESERVE_SPLIT_CHOICE=1
```

### 独立64 paired seed

all-ReplayDominance、fast strict、base seed `20262000`、64 paired。旧候補は同じbundleで
`SEMANTIC_EXACT_PREY_OUTCOME=0`、候補は上記設定。

| 指標 | baseline | scoped candidate | 差 |
|---|---:|---:|---:|
| 平均最終質量 | 40.196 | 42.566 | +2.370 |
| 1位率 | 42.2% | 45.3% | +3.1pt |
| 平均順位 | 2.625 | 2.594 | -0.031 |
| 片側95%下限 | - | - | +0.408 |
| 正常完走 | 64/64 | 64/64 | 同等 |

責務分離前は trial 55 が `-55.400` だった。最初の分岐でexact層が基準splitを
non-splitへ変更し、その後約600 roundは候補が優位だったが終盤に全損した。責務分離後は
基準splitを維持し、後続の安全な非分裂捕食だけが働いて同trialは `+66.830` になった。

出力:

- `.agario/benchmarks/exact-prey-scoped-fresh-64/mass_comparisons.json`
- `.agario/diagnostics/exact-prey-scoped-fresh64-trial-55/`

### 公式ラインナップ寄せ paired 12戦

normal runner、calibrated official cohort、base seed `20262120`。paired差は対戦全体の
早期分岐により `-110.577` から `+134.286` と高分散で、12戦の片側下限は負だった。
ただし目的指標と損失構造の集合値は候補が改善した。

| 指標 | baseline | scoped candidate |
|---|---:|---:|
| 平均最終質量 | 65.876 | 72.155 |
| q25最終質量 | 10.578 | 48.125 |
| 勝率 | 66.7% | 66.7% |
| 平均順位 | 2.333 | 1.917 |
| 捕食fragment/試合 | 50.500 | 52.583 |
| 捕食質量/試合 | 79.366 | 96.500 |
| 被捕食fragment/試合 | 6.750 | 7.000 |
| 被捕食質量/試合 | 17.553 | 17.753 |
| 全滅/試合 | 1.417 | 1.083 |
| 終盤100round損失総量 | 70.992 | 36.475 |
| 終了時死亡率 | 8.3% | 0% |

これは、損失総量をほぼ増やさず捕食質量を増やし、特に下位tailと終盤資産保持を改善した
ことを示す。一方、12 pairedだけの信頼区間は高カオス性のため採用根拠にせず、独立64
pairedの正の下限を主要根拠とする。

### all-dominance normal 12戦

base seed `20262130`。12/12正常完走、勝率50.0%、平均最終質量46.901、q25
`1.043`、低質量率33.3%。捕食fragment `33.500`、捕食質量 `57.958`、
被捕食fragment `8.583`、被捕食質量 `22.379`、全滅 `1.333`（全て1試合当たり）。
終盤stalled/deadは0、終盤損失は全enemy lossの1.56%だった。

出力:

- `.agario/analysis/scoped-contact-calibrated-paired-baseline-12.json`
- `.agario/analysis/scoped-contact-calibrated-paired-candidate-12.json`
- `.agario/analysis/scoped-contact-vs-official-dominance-12.json`

### 提出成果物

- `dist/my_bot.py`
- size `121862` bytes
- sha256 `decbe572af00a5acaa3f37176a450e8fbdc18b2c54563dfa890aa28a25844aba`
- `ruff check .`: 成功
- 全テスト: `502 passed`
- `py_compile`: 成功
- normal runner / all-ReplayDominance smoke: `SUCCESS`、1位、最終質量 `170.457`

以上から、Submission #56より次に提出すべき候補はこの scoped forward exact-prey版とする。

`2026-07-20 02:05 JST` の公開leaderboardでは、未更新のSubmission #56で
`Ninja` は `5/71`、平均最終質量 `32.92676162271182`、50試合、
`usesRecentMatchesFallback=false`。新候補はMacロックのため未提出。

`2026-07-20 02:10 JST` に公開 `GET https://api.syncs.org.au/teams/leaderboard`
を再確認した。`Ninja` は引き続き `5/71` だが、51試合、平均最終質量
`32.39729236015377` へ低下し、`usesRecentMatchesFallback=false` だった。02:05の
50試合集計との差から、追加された1試合の最終質量は約 `5.923829` と逆算できる。
4位 `Bots for Life` は `33.29993596303603`、6位 `Washed CS Students` は
`32.242994820820726` で、6位との差は `0.154298` と小さい。新候補へ切り替わった
形跡はなく、Submission #56のローテーションが継続している。

公開Game HubはLeaderboardを表示し、順位行も目視確認できた。一方、認証済み
`/submissions/self/history` の再確認はmacOS Keychain待ちで完了せず、提出APIが
終了しているかはこの時点ではAPI応答で断定していない。過去に認証済みポータルで
表示された `Bot battle is not enabled right now` と、新Submissionが増えていないことは
提出停止を強く示すが、これは公開leaderboardだけからの確定事実とは区別する。

## 2026-07-20: 完了監査での提出入口訂正

先の「提出成果物」`decbe572af00a5acaa3f37176a450e8fbdc18b2c54563dfa890aa28a25844aba`
を再監査したところ、`scripts/build_submission.py` の旧既定値で生成した
`SemanticLookaheadStrategy` 入口だった。一方、scoped exact-prey の独立64 pairedは
`bots/my_bot.py`、すなわち `ReplayDistilledStrategy` 入口で実行していた。同じ基底
`SemanticPotentialStrategy` を共有していても、teacher residualと保全split-vetoの有無が
異なるため、前者の64戦をsemantic bundleの採用根拠にすることはできない。

実際にsemantic bundleそのものをall-ReplayDominance、base seed `20262200`、64 pairedで
再検証すると、exact-prey OFF/ONは次の通りだった。

| 指標 | OFF | ON | 差 |
|---|---:|---:|---:|
| 平均最終質量 | 41.829 | 39.380 | -2.450 |
| 1位率 | 45.3% | 43.8% | -1.5pt |
| 平均順位 | 2.781 | 2.812 | +0.031 |
| 片側95%下限 | - | - | -6.254 |
| 正常完走 | 64/64 | 64/64 | 同等 |

したがってhash `decbe...` は提出候補から撤回した。出力は
`.agario/benchmarks/exact-prey-scoped-dist-fresh-64/`。

### 正しい ReplayDistilled 単一ファイル候補

保存済みの実提出 #56 bundle
`.agario/exports/agario-gpt-pro-review-20260719/dist/my_bot.py`（source fingerprint
`1c679692d0c52a9aa398f57499601539de9fa74beca7872c773cb8fe9a6bb04f`）と、現在の
`ReplayDistilledStrategy` bundleを、all-ReplayDominance、base seed `20262300`、
64 pairedで直接比較した。

| 指標 | #56 | 新候補 | 差 |
|---|---:|---:|---:|
| 平均最終質量 | 40.031 | 44.897 | +4.866 |
| 1位率 | 43.8% | 46.9% | +3.1pt |
| 平均順位 | 2.609 | 2.531 | -0.078 |
| 片側95%下限 | - | - | +1.214 |
| 正常完走 | 64/64 | 64/64 | 同等 |

64差のうち負は1件だけ（`-21.359`）。正の非ゼロ差は `+3.493` から
`+116.065` まで8件あり、残りは同値だった。出力は
`.agario/benchmarks/sub56-vs-replay-distilled-candidate-64/`。

さらに公式30 cohortを使うcalibrated fixture、base seed `20262400`、通常ランナー
（公式同等の累積8秒制限）12 pairedでは両方12/12正常完走し、次の集合値だった。

| 指標 | #56 | 新候補 |
|---|---:|---:|
| 平均最終質量 | 62.130 | 65.739 |
| q25最終質量 | 1.315 | 10.090 |
| 勝率 | 50.0% | 66.7% |
| 平均順位 | 2.667 | 1.917 |
| 捕食fragment/試合 | 42.167 | 64.000 |
| 捕食質量/試合 | 75.021 | 93.152 |
| 被捕食fragment/試合 | 11.500 | 10.250 |
| 被捕食質量/試合 | 22.599 | 26.976 |
| 全滅/試合 | 1.917 | 1.250 |
| 最終100round停止 | 0/12 | 0/12 |

paired平均差は `+3.609` だが、12戦は高分散で片側95%下限 `-34.592`。この集合は
通常制限下の実行可能性とクラウド寄せ挙動の確認に使い、採用の統計根拠は独立64戦の
正の下限とする。新候補は損失質量だけなら増えたが、捕食質量の増分がそれを上回り、
ユーザーが求めた「勝っている時の上側成長」とq25・勝率・平均順位が改善した。

最終候補:

- entry: `ReplayDistilledStrategy`
- `dist/my_bot.py`
- size `152508` bytes
- sha256 `eb9a9b17e861e17b26d535ed467f70b71c88f09b1e6602145c692736f2666c2c`

再発防止として `scripts/build_submission.py` の既定戦略を `replay_distilled` に変更し、
既定bundleテストとREADMEも同じ入口へ揃えた。

最終検証:

- `uv run ruff check .`: 成功
- `uv run pytest -q`: `502 passed`
- candidate関連再試験: `77 passed`
- `uv run python scripts/build_submission.py`: 上記hashを再現
- `uv run python -m py_compile dist/my_bot.py`: 成功
- `git diff --check`: 成功

## 2026-07-20 14:12 JST: 認証済みGame Hub再確認

Chromeの認証済みセッションで公開Game Hubを再確認した。アカウントメニューには
`Profile`、`Member Card`、`Bot Battle`、`Log Out` が表示され、ログイン済みであることを
確認した。Leaderboardのグラフを実際に描画させて上位行を目視すると、順位は次の通り。

| 順位 | チーム | 表示試合数 |
|---:|---|---:|
| 1 | PorkyPig.py | 788 |
| 2 | team | 742 |
| 3 | Ninja | 787 |
| 4 | Bots for Life | 786 |

したがって、この時点の `Ninja` は **3位**。確認用スクリーンショットは
`/Users/macbookair/.codex/visualizations/2026/07/20/019f7de9-cb51-70d2-977f-c23c79c09b1c/ninja-leaderboard-top.png`。

注意点として、最初のDOM snapshotはグラフ内のcanvas文字列を返さず、順位表が空に
見えた。また初回スクリーンショットでは `Ninja (213)` を含む古い集合が描画されたが、
認証済みBot Battleへ遷移してGame Hubへ戻ると787試合集合へ更新された。今後は
見出し・説明文だけのDOMを「順位なし」と解釈せず、canvasをスクリーンショットで読み、
再取得後の試合数も確認する。
