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
6. [已知 Bug 记录](#已知-bug-记录)
7. [待办事项](#待办事项)

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
| AZ 训练 (默认规则) | 确立 baseline，发现 bug | ✅ 完成 | `runs/rq4_az_palace_knight/` `runs/rq4_az_nq_allrules/` |
| AZ 训练 (修复 bug 后真正规则) | no_queen+ALL_RULES 真实训练 | 🔄 进行中 | `runs/rq4_az_nq_allrules_v2/` |

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

> ⚠️ **重要 Bug（已修复）**: `az_selfplay_parallel.py` 的 worker 之前调用 `_apply_ablation(ablation)` 但丢弃了返回值，导致所有 worker 始终使用**默认规则**而非指定变体。`az_eval_parallel.py` 和 `az_eval.py` 也存在同样问题。
> **修复时间**: 2026-04-18，提交时已修正如下三个文件：
> - `hybrid/rl/az_selfplay_parallel.py` line 44–63
> - `hybrid/rl/az_eval.py` `play_one_game` / `play_match`
> - `hybrid/rl/az_eval_parallel.py` `_eval_worker` / `_gating_worker`

---

### Run 1: palace + knight_block (10 轮) — ⚠️ 实为默认规则

- **输出**: `runs/rq4_az_palace_knight/`
- **配置**: `--ablation chess_palace,knight_block --iterations 10 --selfplay-games-per-iter 100 --simulations 50 --use-cpp --num-workers 4`
- **实际规则**: 默认规则（bug 导致 worker 未应用 variant）
- **结果**: Chess 32.3% / XQ 3.2% / Draw 64.5%, C:X=10.1x

### Run 2: no_queen + ALL_RULES (20 轮) — ⚠️ 实为默认规则

- **输出**: `runs/rq4_az_nq_allrules/`
- **配置**: `--ablation no_queen,chess_palace,knight_block,no_promotion --iterations 20 --use-cpp --num-workers 4`
- **实际规则**: 默认规则（同样的 bug）
- **结果**: Chess 32.4% / XQ 2.6% / Draw 65.0%, C:X=12.4x
- **诊断**: `surv_chess_QUEEN ≈ 0.48` 揭示了 Queen 仍然存在 → 确认 bug

**两次运行实际上都是同一个默认规则下的 AZ baseline，可合并理解。**

新增诊断指标（在 bug 修复后的运行中正式生效）：
- `sp_chess_end_mat` / `sp_xq_end_mat`: 终局时双方总子力值
- `surv_chess_QUEEN` 等: 各棋子存活率
- `lost_chess_PAWN` 等: 各棋子平均损失数量

### Run 3: no_queen + ALL_RULES v2 (20 轮) — ✅ 规则正确

- **输出**: `runs/rq4_az_nq_allrules_v2/` ← **当前正在进行**
- **配置**: 同 Run 2，但已修复 variant bug
- **规则**: `no_queen + no_promotion + chess_palace + knight_block`（实际生效）
- **状态**: 🔄 训练中（启动于 2026-04-18 06:41）
- **Dashboard**: `runs/rq4_az_nq_allrules_v2/dashboard.html`（每 15s 刷新）

---

## 启动训练的标准命令

```bash
# 标准 AZ 训练（20轮，加速配置）
python scripts/train_az_iter.py \
  --iterations 20 \
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
  --ablation "no_queen,chess_palace,knight_block,no_promotion" \
  --use-cpp \
  --num-workers 4 \
  --outdir "runs/MY_RUN_NAME"

# 启动进度 Dashboard（另一个终端）
python scripts/az_dashboard.py runs/MY_RUN_NAME "variant_name"
```

---

## 已知 Bug 记录

### Bug #1: Variant 未传入 self-play worker (已修复)

- **影响范围**: 所有使用 `--num-workers > 1` 的并行训练（即所有生产训练）
- **现象**: `_apply_ablation(ablation)` 返回 `VariantConfig` 但被丢弃，worker 用默认规则
- **受影响文件**: `az_selfplay_parallel.py`, `az_eval_parallel.py`, `az_eval.py`, `az_runner.py`
- **修复**: 使用返回值 `variant_cfg = _apply_ablation(...)` 并传入 `HybridChessEnv(variant=variant_cfg)`
- **修复日期**: 2026-04-18

### Bug #2: INITIAL_PIECES 含 KING/GENERAL (已修复)

- **现象**: piece_census 计算时包含 chess_KING / xiangqi_GENERAL，导致 CSV 多出不在 CSV_COLUMNS 的列，训练崩溃
- **修复**: 从 `INITIAL_PIECES` 中删除 `chess_KING` 和 `xiangqi_GENERAL`（王/将永不被吃）
- **修复日期**: 2026-04-17

---

## 待办事项

- [ ] 分析 Run 3 (`rq4_az_nq_allrules_v2`) 结果，与 Run 1/2（默认规则 baseline）对比
- [ ] 若 C:X 比仍 > 5x，考虑进一步测试 `palace+knight_block`（不删后）的正确训练版本
- [ ] 更多轮训练（≥50 轮）验证 AZ 收敛后的平衡性
- [ ] 将实验结果写入课程报告 (`course_project/`)
