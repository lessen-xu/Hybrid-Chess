# Hybrid Chess：实验结果报告

> 最后更新：2026-05-28
---

## 目录

1. [项目结构](#项目结构)
2. [实验概览](#实验概览)
3. [RQ4：早期探索](#rq4早期探索)
4. [AB D2 规则改革扫描](#ab-d2-规则改革扫描)
5. [规则改革工程实现](#规则改革工程实现)
6. [AlphaZero 九变体训练](#alphazero-九变体训练)
7. [因子分析](#因子分析)
8. [跨变体锦标赛（RQ3）](#跨变体锦标赛rq3)
9. [推荐方案](#推荐方案)
10. [训练标准命令](#训练标准命令)
11. [待办事项](#待办事项)

---

## 项目结构

```
hybrid chess/
├── cpp/                   # C++ 引擎 (move gen, AB search, pybind11)
│   └── src/
├── hybrid/
│   ├── core/              # 游戏引擎 (types, board, rules, config, env, fen)
│   ├── agents/            # AI agent (Random, Greedy, AlphaBeta, AlphaZero)
│   └── rl/                # AlphaZero pipeline (network, encoding, selfplay, train, eval, runner)
├── scripts/
│   ├── train_az_iter.py                       # AZ 训练 CLI 入口
│   ├── run_all.py                    # 编排器：顺序训练 9 个变体
│   ├── dashboard.py                  # 实时 HTML 进度面板
│   ├── cross_variant_tournament.py   # 带温度采样的跨变体锦标赛
│   ├── rq4_rule_reform_ab.py         # AB D2 规则改革扫描
│   └── eval_arena.py                          # 换边评估
├── tests/                 # 测试套件（331 测试，含 conftest.py 全局态重置）
├── ui/                    # 浏览器对局 UI
├── runs/         # 实验输出（gitignored）
│   ├── rq4_rule_reform_ab/         # AB 扫描结果
│   ├── rq4_az_default/             # Default 50 轮
│   ├── rq4_az_noq_only/            # noQ 50 轮
│   ├── rq4_az_xqqueen_only/        # xqQueen 50 轮
│   ├── rq4_az_palace_knight/       # PK 50 轮
│   ├── rq4_az_pk_nopromo/          # PK+noPromo 50 轮
│   ├── rq4_az_pk_xqqueen/          # PK+xqQueen 50 轮 ⭐
│   ├── rq4_az_nq_nopromo/          # noQ+noPromo 50 轮
│   ├── rq4_az_nq_pk/               # noQ+PK 50 轮
│   ├── rq4_az_nq_allrules/         # noQ+ALL 50 轮
│   ├── cross_variant_tournament/   # 初版 n=100 锦标赛（3,600 局）
│   └── cross_variant_tournament_ext/ # n=500 扩展（合计 17,969 局）
└── docs/
    ├── ARCHITECTURE.md
    ├── EXPERIMENTS_ZH.md  # 本文件（中文）
    └── EXPERIMENTS_EN.md  # 英文版
```

---

## 实验概览

| 阶段 | 目标 | 状态 | 主要产物 |
|------|------|------|----------|
| AB D2 规则改革扫描 | 23 变体快速筛选 | ✅ 完成 | `runs/rq4_rule_reform_ab/` |
| AZ 九变体对比（各 50 轮） | 寻找最优平衡 | ✅ 完成 | `runs/rq4_az_*` |
| 跨变体锦标赛 | 元策略分析 | ✅ 完成 | `runs/cross_variant_tournament/` |

- **AZ 训练**：9 个变体 × 50 轮 = 450 轮，共 45,000 局自对弈
- **AB 扫描**：23 个变体 × 40 局 = 920 局
- **锦标赛**：36 个 unordered 对 × 每对 500 局 side-swapped = 17,969 局（初版 n=100 共 3,600 局曾出现表观 3-cycle，n=500 复测后被推翻）

---

## RQ4：早期探索

用 AB D2 试验了棋子削弱（no_queen, no_bishop, extra_soldier 等），发现：
- 默认规则 mat_diff ≈ +19（Chess 碾压）
- 棋子削弱可接近 0 但和棋率过高（AB D2 太浅，"平衡"实为无效对弈）
- 引入 `mat_diff` 作为物质差指标区分"真平衡"和"无效局"

**结论**：单纯削减棋子无法消除 Chess 的结构性优势，需要从规则层面改革。

---

## AB D2 规则改革扫描

- **脚本**: `scripts/rq4_rule_reform_ab.py`
- **输出**: `runs/rq4_rule_reform_ab/results.json` + `progress.log`
- **规模**: 23 个变体 × 40 局，Alpha-Beta 深度=2，C++ 加速，8 worker
- **三项改革规则**:
  - `no_promotion`: 兵到底线不升变，保持兵身份
  - `chess_palace`: Chess King 限制在 3×3 宫内 (x=3–5, y=0–2)
  - `knight_block`: Chess Knight 遵循象棋马的蹩脚规则

按 `|avg_mat_diff|` 排名（越接近 0 越平衡）。`mtb*` 是和棋下的物质判决。

| 排名 | 变体 | matdiff | C | X | 和 | mtbC | mtbX | mtbE | 均ply |
|------|------|---------|---|---|----|------|------|------|-------|
| 1 | palace+knight_blk | +0.0 | 0 | 0 | 40 | 0 | 0 | 40 | 85 |
| 2 | ALL_RULES | +0.0 | 0 | 0 | 40 | 0 | 0 | 40 | 85 |
| 3 | nq+ec | +1.0 | 0 | 0 | 40 | 40 | 0 | 0 | 64 |
| 4 | nq+ec+no_promo | +1.0 | 0 | 0 | 40 | 40 | 0 | 0 | 64 |
| 5 | nq+ec+palace | +1.0 | 0 | 0 | 40 | 40 | 0 | 0 | 64 |
| 6 | nq+nb | −2.0 | 0 | 0 | 40 | 0 | 40 | 0 | 45 |
| 7 | nq+nb+no_promo | −2.0 | 0 | 0 | 40 | 0 | 40 | 0 | 45 |
| 8 | nq+nb+palace | −2.0 | 0 | 0 | 40 | 0 | 40 | 0 | 45 |
| 9 | no_queen+ALL_RULES | +3.0 | 0 | 0 | 40 | 40 | 0 | 0 | 101 |
| 10 | nq+nb+knight_blk | −5.0 | 0 | 0 | 40 | 0 | 40 | 0 | 27 |
| 11 | nq+nb+es+ALL_RULES | +7.0 | 0 | 0 | 40 | 40 | 0 | 0 | 108 |
| 12 | no_queen | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 13 | no_queen+no_promo | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 14 | no_queen+palace | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 15 | nq+nb+ALL_RULES | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 88 |
| 16 | default | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 17 | no_promo | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 18 | palace | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 19 | no_promo+palace | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 20 | no_queen+knight_blk | +16.0 | 0 | 0 | 40 | 40 | 0 | 0 | 146 |
| 21 | knight_blk | +17.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 22 | no_promo+knight_blk | +17.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 23 | nq+ec+ALL_RULES | +23.0 | 0 | 0 | 40 | 40 | 0 | 0 | 149 |

**结论**：`palace + knight_block`（以及包含它的 ALL_RULES 组合）在浅层 AB 下达成完美物质平衡（matdiff = 0.0），是这个筛选阶段的最优结构性干预。默认规则下 Chess 物质优势明显（matdiff ≈ +11）。只有 knight_block 单独使用时严格弱于 knight_block + palace 组合，因为 palace 在深度 2 单独起不到决定性作用。

---

## 规则改革工程实现

**C++ 端**（`cpp/src/`）：
- `types.h`：`RuleFlags` 结构 + `thread_local g_rule_flags`；新增 `PieceKind::XQ_QUEEN` 枚举用于 xiangqi 侧 queen-like 棋子。
- `rules.cpp`：三项规则在着法生成、攻击检测、快速 `is_square_attacked_fast` 路径全部接入；XQ_QUEEN 在 xiangqi 侧的正交+对角线攻击都识别。
- `bindings.cpp`：把 `RuleFlags`、`set_rule_flags` 和 `XQ_QUEEN` 暴露给 Python。
- `zobrist.h`：Zobrist 表扩到 14 个棋种；`board.cpp` 重复局面 hash 用完整枚举名作为 token（不是首字母），KING/KNIGHT 和 CHARIOT/CANNON 不会再发生碰撞。

**Python 端**（`hybrid/core/`）：
- `types.py`：`PieceKind` 加 `XQ_QUEEN`。
- `board.py` / `rules.py`：`xq_queen=True` 时左侧 Advisor 位置放 `PieceKind.XQ_QUEEN`；着法生成里 `QUEEN` 和 `XQ_QUEEN` 都按 queen-like slider 处理。
- `config.py`：`VariantConfig` 加 `no_promotion`, `chess_palace`, `knight_block`, `xq_queen` 字段。
- `env.py` `_set_active_variant()`：环境 reset 时自动同步 C++ 规则 flag。

**Ablation 映射**（`hybrid/rl/az_runner.py`）：
```python
'no_promotion':  {'no_promotion': True},
'chess_palace':  {'chess_palace': True},
'knight_block':  {'knight_block': True},
'xq_queen':      {'xq_queen': True},
```

**状态编码**：15 通道二值平面（每个棋种一个通道，`XQ_QUEEN` 独占一个通道，所以 xiangqi 侧 queen-like 棋子和 Chess Queen 在同一格出现时归属不会含糊）+ 1 个 side-to-move 平面。

---

## AlphaZero 九变体训练

### 配置

所有 AZ 运行使用统一配置（50 轮 × 100 局/轮 = 每变体 5000 局自对弈）：
- 自对弈：100 局/轮、50 sims、max_ply=150、4 worker
- 训练：2 epoch、batch=256、replay buffer=50000
- 评测：每 2 轮一次，20 局对 Random + 20 局对 AB(d1)
- 总计：**9 变体 × 50 轮 = 45,000 局自对弈**

> **PK** = chess_palace + knight_block，**noQ** = no_queen，**xqQueen** = xq_queen，**ALL** = PK + no_promotion

### 九变体对比（最后 10 轮平均）

| 变体 | 轮数 | Chess% | XQ% | 和% | C:X | MatDiff |
|------|------|--------|-----|-----|-----|---------|
| Default | 50 | 35.6 | 4.0 | 60.4 | 8.9× | −6.40 |
| noQ | 50 | 0.9 | 1.6 | 97.5 | 0.6× | −11.72 |
| xqQueen | 50 | 22.8 | 7.8 | 69.4 | 2.9× | −11.27 |
| PK | 50 | 30.9 | 9.3 | 59.8 | 3.3× | −6.77 |
| PK+noPromo | 50 | 31.1 | 9.1 | 59.8 | 3.4× | −6.25 |
| **PK+xqQueen** ⭐ | 50 | **21.2** | **18.0** | **60.8** | **1.2×** | **−10.68** |
| noQ+noPromo | 50 | 2.2 | 1.4 | 96.4 | 1.6× | −11.32 |
| noQ+PK | 50 | 1.2 | 3.6 | 95.2 | 0.3× | −11.57 |
| noQ+ALL | 50 | 1.5 | 4.6 | 93.9 | 0.3× | −11.58 |

只看决定性比赛率合理（和棋率不超过 ~70%）的方案，**PK+xqQueen 是最接近 1:1 平衡的，C:X = 1.2×**。
去掉 Chess Queen 的变体（noQ、noQ+*）虽然 C:X 比也接近 1，但靠的是把和棋率推到 95% 以上，这是和棋退化而不是策略平衡。

---

## 因子分析

### Queen 配置 × 结构改革（最后 10 轮平均）

| | 无 PK | 加 PK |
|--|------|------|
| **Chess 有 Q / XQ 无 Q** | Default 8.9× (60% 和) | PK 3.3× (60% 和) |
| **Chess 有 Q / XQ 有 Q** | xqQueen 2.9× (69% 和) | **PK+xqQueen 1.2× (61% 和)** ⭐ |
| **Chess 无 Q / XQ 无 Q** | noQ 0.6× (98% 和) | noQ+PK 0.3× (95% 和) |

> 单一维度的干预不够。只加 `xq_queen`（xqQueen）后 Chess 仍有 ~3× 的优势；只加 `PK` 后仍有 ~3.3×。
> **必须把 `PK` 和 `xq_queen` 组合起来才能把 ratio 拉到 1.x 区间，且和棋率与 Default 同档。**
> 去掉 Chess Queen 能把 ratio 压到 1 以下，但代价是 >95% 的和棋率。这是一个决断匮乏的退化博弈，不是策略平衡。

### xq_queen 稳定性（PK+xqQueen 每 10 轮趋势）

PK+xqQueen 在 ~20 轮时达到 1.2× 稳态，之后稳定在 1.0–1.5× 区间（见 `runs/rq4_az_pk_xqqueen/metrics.csv`）。

### 棋子存活率（PK+xqQueen 变体，最后 10 轮平均）

存活率分母按变体感知（xq_queen 变体起始有 1 个 XQ_QUEEN + 1 个右侧 Advisor）。详见 `metrics.csv` 中 `surv_*` 列。

---

## 跨变体锦标赛（RQ3）

### 目的

让用不同规则变体训练出的 AZ agent，在**默认规则**下相互对战，看训练条件如何塑造策略。

### 配置

- **Agent 池**：9 个变体的 `best_model.pt`（都训练满 50 轮）
- **对局规则**：Default（标准 Hybrid Chess，无改革）
- **局数**：36 个 unordered 对 × 每对 500 局 side-swapped = **17,969 局**（扣除 31 局 seed-collision 重复）
- **搜索**：50 sims MCTS，C++ 引擎，6 并行 worker
- **动作选择**：访问数温度采样（`temperature=0.5`），保证同一对 agent + 同一颜色 + 不同种子的对局真正分散。
- **种子**：用 `hashlib.sha256((name_a, name_b, half, gi))` 生成（跨进程/跨 session 完全可复现）。
- **输出**：`runs/cross_variant_tournament/analysis_500/` 包含 `payoff_matrix.csv`、`elo.csv`、`per_side.csv`、`pairwise_significance.csv`、`decisive_rate.csv`、`game_length.csv`。

### Payoff 矩阵（n=500，行对列）

| | Default | noQ | xqQueen | PK | PK_noPromo | PK_xqQueen | noQ_noPromo | noQ_PK | noQ_ALL |
|--|------|------|------|------|------|------|------|------|------|
| **Default** | 0.500 | 0.484 | 0.509 | 0.492 | 0.522 | 0.522 | 0.487 | 0.537 | 0.499 |
| **noQ** | 0.516 | 0.500 | 0.516 | 0.503 | 0.521 | 0.501 | 0.502 | 0.586 | 0.550 |
| **xqQueen** | 0.491 | 0.484 | 0.500 | 0.460 | 0.476 | 0.497 | 0.466 | 0.538 | 0.516 |
| **PK** | 0.508 | 0.497 | 0.540 | 0.500 | 0.530 | 0.505 | 0.491 | 0.528 | 0.509 |
| **PK_noPromo** | 0.478 | 0.479 | 0.524 | 0.470 | 0.500 | 0.525 | 0.482 | 0.585 | 0.514 |
| **PK_xqQueen** | 0.478 | 0.499 | 0.503 | 0.495 | 0.475 | 0.500 | 0.438 | 0.509 | 0.495 |
| **noQ_noPromo** | 0.513 | 0.498 | 0.534 | 0.509 | 0.518 | 0.562 | 0.500 | 0.524 | 0.507 |
| **noQ_PK** | 0.463 | 0.414 | 0.462 | 0.472 | 0.415 | 0.491 | 0.476 | 0.500 | 0.498 |
| **noQ_ALL** | 0.501 | 0.450 | 0.484 | 0.491 | 0.486 | 0.505 | 0.493 | 0.502 | 0.500 |

### Agent 排名（Bradley–Terry Elo，均值锚定 1500，500 次 bootstrap 95% CI）

| 排名 | Agent | Elo | 95% CI | 平均分 | 训练规则 |
|------|-------|-----|--------|--------|---------|
| 1 | **noQ_noPromo** | 1520.8 | [1502.5, 1527.3] | 0.521 | noQ + 禁升变 |
| 2 | **noQ** | 1520.5 | [1503.7, 1529.6] | 0.524 | 去 Chess Queen |
| 3 | **PK** | 1514.5 | [1494.4, 1521.4] | 0.514 | 宫 + 蹩脚 |
| 4 | Default | 1503.4 | [1490.0, 1517.9] | 0.506 | 标准规则 |
| 5 | PK_noPromo | 1502.5 | [1490.3, 1517.7] | 0.507 | PK + 禁升变 |
| 6 | noQ_ALL | 1488.4 | [1478.4, 1505.5] | 0.489 | noQ + ALL |
| 7 | xqQueen | 1488.2 | [1480.6, 1505.5] | 0.491 | 给 XQ Queen |
| 8 | PK_xqQueen | 1486.2 | [1479.5, 1502.2] | 0.487 | PK + XQ Queen |
| 9 | noQ_PK | 1475.6 | [1467.8, 1487.3] | 0.461 | noQ + PK |

> 平均分区间从 n=100 的 0.449–0.531 收窄到 n=500 的 **0.461–0.524**。Bootstrap CI 跨度约 15–30 Elo，36 对里仍有 32 对 Wilson CI 包含 0.50，说明多数 agent 间差距仍在采样噪声之内。

### 关键发现

#### 1. In-variant 平衡不能预测 default 下的迁移强度

训练时最平衡的变体（PK+xqQueen，in-variant C:X = 1.2×）在 default 规则下落到**第 8 名**（avg 0.487）。排名靠前的 noQ、noQ_noPromo 反而是自对弈和棋率 >96% 的退化变体。在我们测试的方案集合里，in-variant 平衡和 default 强度衡量的是 agent 的不同属性。

#### 2. n=100 时的表观 3-cycle，n=500 复测后被推翻

初版 100 局锦标赛在 `PK`、`xqQueen`、`PK_xqQueen` 之间表面构成一个"石头剪刀布"循环：

| Edge | 分数 (n=100) | 95% CI (n=100) |
|------|-------------|----------------|
| PK vs xqQueen | 0.575 | [0.477, 0.667] |
| xqQueen vs PK_xqQueen | 0.520 | [0.423, 0.615] |
| PK_xqQueen vs PK | 0.515 | [0.418, 0.611] |

对这三对各跑 500 局复测（相同 seed 约定，side-swapped，T=0.5）：

| Edge | 分数 (n=500) | 95% CI (n=500) | 方向 |
|------|-------------|----------------|------|
| PK vs xqQueen | 0.540 | [0.508, 0.572] | 方向保持，CI 排除 0.5 |
| xqQueen vs PK_xqQueen | 0.497 | [0.453, 0.541] | 方向翻转，CI 横跨 0.5 |
| PK_xqQueen vs PK | 0.495 | [0.453, 0.541] | 方向翻转，CI 横跨 0.5 |

n=500 下三条边里有两条翻向，cycle 结构消失，只剩 PK vs xqQueen 显著偏离 0.5。**原本的 3-cycle 是小样本伪结构**。复测输出在 `runs/cycle_3pair_ci/`。

复现命令：`python -m scripts.cycle_3pair_ci --games 250 --workers 12`。

#### 3. n=500 下的成对显著性

按 Wilson 95% CI，36 对里有 4 对 CI 排除 0.50：noQ vs noQ_PK (0.586)、PK_noPromo vs noQ_PK (0.585)、noQ_noPromo vs PK_xqQueen (0.562)、noQ vs noQ_ALL (0.550)。这 4 对在 n=100 时 CI 都包含 0.50，多出来的分辨率完全来自更大的 N。详见 `runs/cross_variant_tournament/analysis_500/pairwise_significance.csv`。

---

## 推荐方案

**`chess_palace + knight_block + xq_queen` (PK+xqQueen)** 是最干净的 in-variant 平衡：
- In-variant C:X ≈ **1.2×**（在非退化的方案里最接近 1:1）
- 和棋率 ~61%（与 Default 相当，远低于去 Queen 类的 95%+）
- 把结构性约束（Chess 的宫 + 蹩脚）和战术资源（xiangqi 侧的 queen-like 棋子）组合起来，不依赖单一维度干预。

任何单一改动（只 `xq_queen`、只 `PK`、或只 `no_queen`）都会留下明显的 Chess 残余优势，或者把和棋率推到 ~100%。

---

## 训练标准命令

```bash
# 单个变体
python scripts/train_az_iter.py \
  --iterations 50 --selfplay-games-per-iter 100 --simulations 50 \
  --selfplay-max-ply 150 --batch-size 256 --train-epochs 2 \
  --eval-games 20 --eval-interval 2 --eval-simulations 100 \
  --disable-gating 1 --resign-enabled 1 --device auto --seed 42 \
  --ablation "chess_palace,knight_block,xq_queen" --use-cpp --num-workers 4 \
  --outdir runs/rq4_az_pk_xqqueen

# 9 个变体顺序训（自动 resume + retry）
python -m scripts.run_all

# 实时 HTML 进度面板（另一个终端）
python -m scripts.dashboard
# 然后浏览器打开 runs/progress.html（每 30 秒自动刷新）

# 9 个 best_model.pt 的跨变体锦标赛（n=500 设置）
python -m scripts.cross_variant_tournament \
  --games 250 --sims 50 --workers 6 --temperature 0.5 --seed 42
```

---

## 待办事项

- [x] AB D2 规则改革扫描（23 变体）
- [x] AZ 9 变体训练（每个 50 轮 × 100 局 × 50 sims × 150 ply）
- [x] 跨变体锦标赛（n=500，共 17,969 局，温度采样，确定性种子）
- [x] Bradley–Terry Elo + bootstrap CI、per-side breakdown、pairwise significance、decisive rate、game-length 分析
- [x] 因子分析（Queen × PK）
- [x] 非传递循环检测（n=100 时出现，n=500 复测后被推翻）
- [x] 500 局 cycle 复测 + Wilson 95% CI（`scripts/cycle_3pair_ci.py`）
- [x] 全部图表从数据重新生成（`course_project/plot_figures.R`、`course_project/plot_cycle_replay.py`）
- [x] 课程最终报告改写（n=500，通过 side-of-play 和 in-variant-vs-transfer 分析闭环 RQ3）
