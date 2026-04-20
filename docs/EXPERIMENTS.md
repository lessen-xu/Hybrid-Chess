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
│   │   ├── config.py      # VariantConfig (规则标志位)
│   │   ├── env.py         # 主环境，负责 C++ 规则同步
│   │   └── rules.py       # Python 规则引擎 (与 C++ 对称)
│   └── rl/
│       ├── az_runner.py           # AZ 训练主循环 (run_iterations)
│       ├── az_selfplay.py         # 自对弈，GameRecord + piece_census
│       ├── az_selfplay_parallel.py # 多进程自对弈 worker
│       ├── az_eval.py             # 评估 (play_match, play_one_game)
│       └── az_eval_parallel.py    # 并行评估 worker
├── scripts/
│   ├── train_az_iter.py           # AZ 训练 CLI 入口
│   ├── az_dashboard.py            # 实时 HTML 进度面板（每 15s 刷新）
│   ├── rq4_rule_reform_ab.py      # AB D2 规则改革扫描实验
│   └── launch_az_palace_knight.py # 已废弃，用 train_az_iter.py
├── runs/                  # 所有实验输出（已 .gitignore）
└── docs/
    ├── ARCHITECTURE.md   # 系统架构说明
    └── EXPERIMENTS.md    # 本文件
```

---

## 实验阶段总览

| 阶段 | 目标 | 状态 | 主要产物 |
|------|------|------|---------|
| RQ4 初步平衡 | 探索 no_queen/no_bishop 等棋子削弱 | ✅ 完成 | `runs/rq4_*` |
| AB D2 规则改革扫描 | 测试 palace/knight_block/no_promotion 23个变体 | ✅ 完成 | `runs/rq4_rule_reform_ab/` |
| AZ Baseline（默认规则） | 20轮自对弈建立参照 | ✅ 完成 | `runs/rq4_az_nq_allrules/` |
| AZ no_queen+ALL_RULES | 50轮全规则改革+删后 | ✅ 完成 | `runs/rq4_az_nq_allrules_v2/` |
| AZ palace+knight_block | 50轮纯结构改革（保留Queen） | ✅ 完成 | `runs/rq4_az_palace_knight_v2/` |

---

## RQ4 — 规则平衡探索

### 默认规则基准
- **路径**: `runs/rq4_default/`
- **结果**: Chess 大幅优势（AB D2 下 mat_diff ≈ +19，即 Chess 多 19 点子力）
- **结论**: 原始规则严重失衡，Chess 结构性优势明显

### 棋子削弱系列
- **路径**: `runs/rq4_balance_search/`, `runs/rq4_balance_refine/`, `runs/rq4_no_queen/`
- **方法**: 逐步删除 Chess 棋子（no_queen, no_bishop, extra_soldier 等）
- **最佳发现**: `no_queen + no_bishop + extra_soldier` → mat_diff 接近 0，但和棋率极高（抽屉问题）
- **问题**: AB D2 太浅，"平衡"实为无效对弈（双方无法突破）

### AB D2 诊断
- **路径**: `runs/rq4_draw_diagnosis/`, `runs/rq4_side_balance/`
- **发现**: 引入 `mat_diff`（物质差）作为平局决胜指标，区分"真平衡"和"无效局"

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

所有 AZ 运行使用统一配置：
- 自对弈：100 局/轮，50 sims，max_ply=150，4 workers
- 训练：2 epochs，batch=256，buffer=50000
- 评估：20 局 vs Random + 20 局 vs AB(d1)，每 2 轮

### 七变体完整对比

| 变体 | 轮次 | Chess% | XQ% | Draw% | C:X | L10 C:X | mat_diff | 输出目录 |
|------|------|--------|-----|-------|-----|---------|----------|---------|
| Default | 20 | 32.4% | 2.6% | 65.0% | 12.4x | 13.5x | −6.8 | `rq4_az_nq_allrules/` |
| PK | 50 | 30.1% | 8.7% | 61.1% | 3.4x | 3.2x | −6.6 | `rq4_az_palace_knight_v2/` |
| PK+noPromo | 50 | 31.0% | 8.8% | 60.3% | 3.5x | 4.0x | −6.8 | `rq4_az_pk_nopromo/` |
| **PK+xqQueen** | **50** | **18.5%** | **27.4%** | **54.1%** | **0.7x** | **0.7x** | **−10.7** | **`rq4_az_pk_xqqueen/`** |
| noQ+noPromo | 50 | 9.6% | 2.6% | 87.8% | 3.7x | 2.0x | −13.4 | `rq4_az_nq_nopromo/` |
| noQ+PK | 50 | 4.8% | 7.6% | 87.7% | 0.6x | 0.2x | −12.7 | `rq4_az_nq_pk/` |
| noQ+ALL | 50 | 3.9% | 6.9% | 89.3% | 0.6x | 0.3x | −12.5 | `rq4_az_nq_allrules_v2/` |

> **PK** = palace + knight_block, **noQ** = no_queen, **ALL** = palace + knight_block + no_promotion

### 最佳变体：PK + xqQueen

`palace + knight_block + xq_queen`（宫格 + 蹩脚 + 给 XQ 一个后）

| 轮次 | Chess | XQ | Draw | C:X |
|------|-------|----|------|-----|
| 0–9 | 188 | 257 | 555 | 0.7x |
| 10–19 | 196 | 264 | 540 | 0.7x |
| 20–29 | 170 | 287 | 543 | 0.6x |
| 30–39 | 179 | 287 | 534 | 0.6x |
| 40–49 | 193 | 274 | 533 | 0.7x |

- C:X = 0.7x **极其稳定**（50 轮内无漂移）
- 和棋率 54%（所有变体中最低 → 最多对决局）
- XQ 略占优但幅度可控

### 关键发现

#### 1. Queen 是平衡性的核心变量

| Chess 有后 / XQ 无后 | C:X 3.4–12.4x（Chess 碾压） |
|-----|-----|
| **双方都有后** | **C:X = 0.7x（接近平衡）** |
| Chess 无后 / XQ 无后 | C:X = 0.3–0.6x（XQ 反超 + 88% 和棋） |

Queen 的存在与否是决定胜负比的**最强因素**。给 XQ 配后比削弱 Chess 更有效。

#### 2. no_promotion 完全无效

PK 与 PK+noPromo 结果一致（C:X = 3.4x vs 3.5x），因为：
- `surv_chess_QUEEN` 两者均 ≈0.49，**零次升变**发生
- 棋盘 9×10，兵需 8 步到底线，150 步上限下根本到不了
- **升变在当前对局条件下是不可能事件**

#### 3. 结构改革（宫格+蹩脚）效果稳定

把 C:X 从 12.4x 压到 3.4x，效果约 **−70%**。但单靠结构改革无法达到 1:1。

#### 4. 删 Queen 导致和棋泛滥

所有 noQ 变体和棋率 87–89%，对局质量下降。C:X 看似接近但大多数局无意义。

### 推荐方案

**`palace + knight_block + xq_queen`** — 最优平衡变体：
- C:X ≈ 0.7x（最接近 1:1）
- 和棋率 54%（最多对决局）
- 趋势稳定（无学习漂移）
- 不削弱任何一方，而是通过增强 XQ 来实现平衡

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
  --ablation "chess_palace,knight_block,xq_queen" \
  --use-cpp \
  --num-workers 4 \
  --outdir "runs/MY_RUN_NAME"
```

---

## 待办事项

- [x] AZ Baseline（20轮默认规则）→ C:X = 12.4x
- [x] AZ palace+knight_block（50轮）→ C:X = 3.4x
- [x] AZ PK+noPromo（50轮）→ C:X = 3.5x（确认 no_promo 无效）
- [x] AZ noQ+PK / noQ+noPromo / noQ+ALL（各50轮）→ 过度削弱
- [x] AZ PK+xqQueen（50轮）→ **C:X = 0.7x（最优方案）** ⭐
- [ ] 将实验结果写入课程报告 (`course_project/`)
