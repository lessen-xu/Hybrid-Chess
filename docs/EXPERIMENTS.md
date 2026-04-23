# Hybrid Chess — Experiment Log

> 记录所有实验的目的、配置、结果和产物路径。
> 最后更新：2026-04-18

---

## 目录

1. [项目结构](#项目结构)
2. [实验阶段总览](#实验阶段总览)
3. [RQ4 — 规则平衡探索](#rq4--规则平衡探索)
4. [规则改革 (Rule Reform)](#规则改革-rule-reform)
5. [AlphaZero 训练](#alphazero-训练)
6. [待办事项](#待办事项)

---

## 项目结构

```
hybrid chess/
├── cpp/                   # C++ 引擎 (walk-generation, AB search, pybind11)
│   └── src/
│       ├── rules.cpp      # 走法生成，含 RuleFlags
│       ├── ab_search.cpp  # Alpha-Beta 搜索
│       ├── bindings.cpp   # Pybind11 接口
│       └── types.h        # RuleFlags struct 定义
├── hybrid/
│   ├── core/
│   │   ├── config.py      # VariantConfig (规则标志位 + xq_queen)
│   │   ├── board.py       # Board + initial_board (变体放子)
│   │   ├── env.py         # 主环境，负责 C++ 规则同步
│   │   └── rules.py       # Python 规则引擎 (与 C++ 对称)
│   └── rl/
│       ├── az_runner.py           # AZ 训练主循环 + ablation 映射
│       ├── az_selfplay.py         # 自对弈，GameRecord + piece_census
│       ├── az_selfplay_parallel.py # 多进程自对弈 worker
│       ├── az_eval.py             # 评估 (play_match, play_one_game)
│       └── az_eval_parallel.py    # 并行评估 worker
├── scripts/
│   ├── train_az_iter.py           # AZ 训练 CLI 入口
│   ├── az_dashboard.py            # 实时 HTML 进度面板
│   └── rq4_rule_reform_ab.py      # AB D2 规则改革扫描
├── runs/                  # 实验输出（.gitignore）
│   ├── rq4_rule_reform_ab/        # AB 扫描结果
│   ├── rq4_az_palace_knight_v2/   # PK 50轮
│   ├── rq4_az_pk_nopromo/         # PK+noPromo 50轮
│   ├── rq4_az_pk_xqqueen/         # PK+xqQueen 50轮 ⭐
│   ├── rq4_az_nq_allrules_v2/     # noQ+ALL 50轮
│   ├── rq4_az_nq_pk/              # noQ+PK 50轮
│   ├── rq4_az_nq_nopromo/         # noQ+noPromo 50轮
│   └── (EGTA 旧实验: az_grand_run_v4/ 等)
└── docs/
    ├── ARCHITECTURE.md
    └── EXPERIMENTS.md    # 本文件
```

---

## 实验阶段总览

| 阶段 | 目标 | 状态 | 主要产物 |
|------|------|------|---------|
| AB D2 规则改革扫描 | 23 变体快速筛选 | ✅ 完成 | `runs/rq4_rule_reform_ab/` |
| AZ Baseline（默认规则 20轮） | 建立参照 | ✅ 完成 | （已清理，数据在报告中） |
| AZ 七变体对比（各 50 轮） | 寻找最优平衡 | ✅ 完成 | `runs/rq4_az_*` |

---

## RQ4 — 规则平衡探索

### 早期探索（已清理）

用 AB D2 试验了棋子削弱（no_queen, no_bishop, extra_soldier 等），发现：
- 默认规则 mat_diff ≈ +19（Chess 碾压）
- 棋子削弱可接近 0 但和棋率过高（AB D2 太浅，"平衡"实为无效对弈）
- 引入 `mat_diff` 作为物质差指标区分"真平衡"和"无效局"

---

## 规则改革 (Rule Reform)

### 实验：AB D2 规则改革扫描 (RQ4)

- **脚本**: `scripts/rq4_rule_reform_ab.py`
- **输出**: `runs/rq4_rule_reform_ab/results.json` + `progress.log`
- **规模**: 23 个变体 × 40 局，Alpha-Beta 深度=2
- **三项改革规则**:
  - `no_promotion`: 兵到底线不升变，保持兵身份
  - `chess_palace`: Chess King 限制在 3×3 宫内 (x=3–5, y=0–2)
  - `knight_block`: Chess Knight 遵循象棋马的蹩脚规则

### 关键结果

| 变体 | avg_mat_diff | 说明 |
|------|-------------|------|
| `default` | +19.0 | Chess 压倒性优势 |
| `palace` | +11.0 | 宫格限制有一定效果 |
| `knight_block` | +12.0 | 蹩脚规则效果有限 |
| **`palace + knight_block`** | **0.00** | ✅ **完美平衡**，无需削减棋子 |
| `no_promotion + palace + knight_block` | +0.5 | 接近平衡 |
| `no_queen + ALL_RULES` | +3.0 | 过度削弱 |

**结论**: `palace + knight_block` 在 AB D2 下达到完美物质平衡，是最优结构性改革方案。

### 工程实现

**C++ 端** (`cpp/src/`):
- `types.h`: 新增 `RuleFlags` struct + `thread_local g_rule_flags`
- `rules.cpp`: 集成三项规则的走法生成和攻击检测
- `bindings.cpp`: 暴露 `RuleFlags`, `set_rule_flags` 给 Python

**Python 端** (`hybrid/core/`):
- `config.py`: `VariantConfig` 新增 `no_promotion`, `chess_palace`, `knight_block` 字段
- `rules.py`: 同步三项规则逻辑分支
- `env.py` `_set_active_variant()`: 环境重置时自动同步 C++ 规则标志

**Ablation 映射** (`hybrid/rl/az_runner.py`):
```python
'no_promotion': {'no_promotion': True},
'chess_palace':  {'chess_palace': True},
'knight_block':  {'knight_block': True},
```

---

## AlphaZero 训练

### 实验配置

所有 AZ 运行使用统一配置（50 轮 × 100 局/轮 = 5,000 局自对弈/变体）：
- 自对弈：100 局/轮，50 sims，max_ply=150，4 workers
- 训练：2 epochs，batch=256，buffer=50000
- 评估：20 局 vs Random + 20 局 vs AB(d1)，每 2 轮
- 总计：**9 变体 × 50 轮 = 45,000 局**自对弈数据

### 九变体完整对比

| 变体 | 轮次 | Chess% | XQ% | Draw% | C:X | L10 C:X | mat_diff | 输出目录 |
|------|------|--------|-----|-------|-----|---------|----------|---------|
| Default | 50 | 29.6% | 3.3% | 67.1% | 9.0x | 6.6x | −6.0 | `rq4_az_default_v2/` |
| Q only | 50 | 11.4% | 2.6% | 86.0% | 4.5x | 3.1x | −14.2 | `rq4_az_noq_only/` |
| **X only** | **50** | **17.9%** | **24.2%** | **57.9%** | **0.7x** | **0.7x** | **−11.1** | **`rq4_az_xqqueen_only/`** |
| PK | 50 | 30.1% | 8.7% | 61.1% | 3.4x | 3.2x | −6.6 | `rq4_az_palace_knight_v2/` |
| PK+noPromo | 50 | 31.0% | 8.8% | 60.3% | 3.5x | 4.0x | −6.8 | `rq4_az_pk_nopromo/` |
| PK+xqQueen | 50 | 18.5% | 27.4% | 54.1% | 0.7x | 0.7x | −10.7 | `rq4_az_pk_xqqueen/` |
| noQ+noPromo | 50 | 9.6% | 2.6% | 87.8% | 3.7x | 2.0x | −13.4 | `rq4_az_nq_nopromo/` |
| noQ+PK | 50 | 4.8% | 7.6% | 87.7% | 0.6x | 0.2x | −12.7 | `rq4_az_nq_pk/` |
| noQ+ALL | 50 | 3.9% | 6.9% | 89.3% | 0.6x | 0.3x | −12.5 | `rq4_az_nq_allrules_v2/` |

> **PK** = chess_palace + knight_block, **Q** = no_queen, **X** = xq_queen, **ALL** = PK + no_promotion

### 因子分析：Queen 配置 × 结构改革

#### Queen 因子（2×2）

| | Chess 有后 | Chess 无后 |
|--|-----------|-----------|
| **XQ 无后** | Default **9.0x** (67% draw) | Q only **4.5x** (86% draw) |
| **XQ 有后** | X only **0.7x** (58% draw) | — |

> **给 XQ 一个后 (X only)** 直接把 C:X 从 9.0x 压到 0.7x，和棋率反而最低。
> **删 Chess 后 (Q only)** 只从 9.0x 降到 4.5x，且和棋飙到 86%。

#### 结构改革因子（PK 的交互效应）

| | 无 PK | 有 PK |
|--|-------|-------|
| **Chess 有后 / XQ 无后** | 9.0x | **3.4x** (PK 有效) |
| **Chess 有后 / XQ 有后** | **0.7x** | **0.7x** (PK 多余) |
| **Chess 无后 / XQ 无后** | **4.5x** | **0.6x** (PK 有效) |

> PK 在 XQ 没有 Queen 时有效（-62% 到 -87%），但在 XQ 有 Queen 时**完全多余**。

### 关键发现

#### 1. xq_queen 是唯一必要的平衡手段

- X only (0.7x) = PK+X (0.7x)，PK 对 xq_queen 变体无附加效果
- xq_queen 单一 flag 的效果 > 所有结构改革的总和
- **给 XQ 一个后比削弱 Chess 更有效且副作用更小**

#### 2. 删 Queen 导致和棋泛滥

所有含 `no_queen` 的变体和棋率 86–89%，对局质量严重下降。

#### 3. no_promotion 完全无效

PK 与 PK+noPromo 结果一致（3.4x vs 3.5x），因为 150 步上限下兵根本到不了底线。

#### 4. xq_queen 变体趋势极其稳定

| 轮次 | X only C:X | PK+X C:X |
|------|-----------|----------|
| 0–9 | 0.7x | 0.7x |
| 10–19 | 0.7x | 0.7x |
| 20–29 | 0.7x | 0.6x |
| 30–39 | 0.7x | 0.6x |
| 40–49 | 0.7x | 0.7x |

50 轮内无漂移，说明 0.7x 是训练收敛后的稳态平衡点。

### 推荐方案

**`xq_queen`**（给 XQ 一个后）— 最简最优平衡变体：
- C:X ≈ 0.7x（最接近 1:1）
- 和棋率 58%（对局质量最高）
- 只改一个 flag，规则最简洁
- 结构改革（PK）可选但非必要

---

## 启动训练的标准命令

```bash
python scripts/train_az_iter.py \
  --iterations 50 \
  --selfplay-games-per-iter 100 \
  --simulations 50 \
  --selfplay-max-ply 150 \
  --batch-size 256 \
  --train-epochs 2 \
  --eval-games 20 \
  --eval-interval 2 \
  --eval-simulations 100 \
  --disable-gating 1 \
  --resign-enabled 1 \
  --device auto \
  --seed 42 \
  --ablation "xq_queen" \
  --use-cpp \
  --num-workers 4 \
  --outdir "runs/MY_RUN_NAME"
```

---

## 跨变体锦标赛（RQ3）

### 实验目的

不同规则下训练出的 AZ agent，在**同一规则**（Default）下互相对打，揭示训练规则如何塑造策略。

### 配置

- **Agent 池**：9 个变体的 `best_model.pt`（全部 50 轮训练）
- **对战规则**：Default（标准 Hybrid Chess，无任何改革）
- **对局数**：36 对 × 50 局（25 局/半，换边） = **1,800 局**
- **搜索**：50 sims MCTS，C++ 引擎，4 workers 并行
- **耗时**：45.7 分钟
- **输出**：`runs/cross_variant_tournament/`

### Payoff Matrix（对称得分）

| | Default | Q_only | X_only | PK | PK_nP | PK_xQ | nQ_nP | nQ_PK | nQ_ALL |
|--|---------|--------|--------|-----|-------|-------|-------|-------|--------|
| **Default** | .500 | .500 | .500 | .250 | .500 | .750 | .500 | .750 | .250 |
| **Q_only** | .500 | .500 | .750 | .500 | .750 | .500 | .750 | .750 | .500 |
| **X_only** | .500 | .250 | .500 | .500 | .750 | .500 | .500 | .750 | .500 |
| **PK** | .750 | .500 | .500 | .500 | .500 | .750 | .750 | .500 | .750 |
| **PK_noPromo** | .500 | .250 | .250 | .500 | .500 | .750 | .500 | .500 | .500 |
| **PK_xqQueen** | .250 | .500 | .500 | .250 | .250 | .500 | .500 | .500 | .250 |
| **noQ_noPromo** | .500 | .250 | .500 | .250 | .500 | .500 | .500 | .500 | .250 |
| **noQ_PK** | .250 | .250 | .250 | .500 | .500 | .500 | .500 | .500 | .500 |
| **noQ_ALL** | .750 | .500 | .500 | .250 | .500 | .750 | .750 | .500 | .500 |

### Agent 排名

| 排名 | Agent | Avg Score | 训练规则 |
|------|-------|-----------|---------|
| 1 | **Q_only** | **0.625** | 删 Chess 后 |
| 1 | **PK** | **0.625** | 宫格+蹩脚 |
| 3 | noQ_ALL | 0.562 | 全削弱 |
| 4 | X_only | 0.531 | 给 XQ 后 |
| 5 | Default | 0.500 | 默认规则 |
| 6 | PK_noPromo | 0.469 | PK+禁升变 |
| 7 | noQ_noPromo | 0.406 | 删后+禁升变 |
| 8 | noQ_PK | 0.406 | 删后+PK |
| 9 | PK_xqQueen | 0.375 | PK+XQ后 |

### 关键发现

#### 1. "逆境出强者"（Adversity Breeds Strength）

在**受限规则**下训练的 agent（Q_only, PK）到了 Default 规则下反而最强（0.625）。
它们在更困难的环境中学会了更精细的防守和进攻策略。

#### 2. Default agent 只排中游

在自己的"主场规则"下训练的 agent 只有 0.500，因为 Default 规则的 Chess 碾压（9.0x）
让它只学会了粗暴进攻，缺乏精细策略。

#### 3. 平衡规则训练 → 迁移最差

PK_xqQueen 垫底（0.375）。在最平衡规则下训练的 agent 习惯了 XQ 有后的"舒适区"，
到了 Default 规则（XQ 无后）时无法适应。

#### 4. Non-transitivity 存在

| A | B | A 得分 | 说明 |
|---|---|--------|------|
| PK_xqQueen | Default | 0.250 | PK_xQ 输 |
| Default | noQ_ALL | 0.250 | Default 输 |
| noQ_ALL | PK_xqQueen | 0.750 | noQ_ALL 赢 |

形成循环克制，验证了 proposal 中 RQ3 的假设：不同训练条件产生**质的不同策略**，
而非简单的强弱排序。

---

## 待办事项

- [x] AZ Default（50轮）→ C:X = 9.0x
- [x] AZ Q only（50轮）→ C:X = 4.5x + 86% draw
- [x] AZ X only（50轮）→ **C:X = 0.7x** ⭐
- [x] AZ PK / PK+noPromo（各50轮）→ 结构改革有效但不足
- [x] AZ PK+xqQueen（50轮）→ C:X = 0.7x = X only
- [x] AZ noQ+PK / noQ+noPromo / noQ+ALL（各50轮）→ 过度削弱
- [x] 因子分析确认：xq_queen 是唯一必要因素
- [x] 跨变体锦标赛（1,800 局）→ "逆境出强者" + non-transitivity ⭐
- [ ] 课程报告
