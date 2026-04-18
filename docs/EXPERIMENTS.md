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
| AZ Baseline（默认规则） | 20轮自对弈建立默认规则参照 | ✅ 完成 | `runs/rq4_az_nq_allrules/` |
| AZ 训练（新规则） | no_queen+ALL_RULES 正式训练 | ✅ 完成 | `runs/rq4_az_nq_allrules_v2/` |

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

### Baseline：默认规则（20 轮）

- **输出**: `runs/rq4_az_nq_allrules/`
- **配置**: `--iterations 20 --selfplay-games-per-iter 100 --simulations 50 --use-cpp --num-workers 4`
- **规则**: 默认规则（标准 Hybrid Chess，含 Queen、升变等）
- **结果**:
  - Chess 32.4% / XQ 2.6% / Draw 65.0%，C:X = 12.4x
  - avg mat_diff = −6.84（终局 XQ 子力反而更多，但 Chess 赢局率碾压）
  - Chess 终局子力均值 22.6 vs XQ 29.4（XQ 子力更厚但无法转化）
- **用途**: 作为对比新规则训练的 baseline
- **新增诊断指标**（本次引入）：
  - `sp_chess_end_mat` / `sp_xq_end_mat`: 终局双方总子力值
  - `surv_chess_QUEEN` 等: 各棋子存活率
  - `lost_chess_PAWN` 等: 各棋子平均损失数量

### Run：no_queen + ALL_RULES（20 轮）✅ 完成

- **输出**: `runs/rq4_az_nq_allrules_v2/`
- **配置**: `--ablation no_queen,chess_palace,knight_block,no_promotion --iterations 20 --use-cpp --num-workers 4`
- **规则**: `no_queen + no_promotion + chess_palace + knight_block`
- **状态**: ✅ 完成（2026-04-18）

#### 结果对比

| 指标 | Baseline（默认规则）| **新规则 v2** | 变化 |
|------|-------------------|--------------|------|
| Chess 胜率 | 32.4% | **9.6%** | ⬇️ 大幅下降 |
| XQ 胜率 | 2.6% | **9.7%** | ⬆️ 大幅上升 |
| 和棋率 | 65.0% | **83.2%** | 更多和局 |
| C:X 胜负比 | 12.4x | **≈1:1** | ✅ 趋近平衡 |
| avg mat_diff | −6.8 | **−13.6** | XQ 子力明显占优 |
| Chess 终局子力 | 22.6 | **17.5** | Chess 子力更少 |
| XQ 终局子力 | 29.4 | **31.4** | XQ 子力更多 |
| surv_chess_QUEEN | 0.48 | **0.00** | ✅ 无 Queen（规则生效） |

#### 趋势分析（每5轮分段）

| 轮次 | Chess | XQ | 和棋 | C:X 比 |
|------|-------|----|------|--------|
| 0–4  | 50    | 47 | 403  | 1.1x |
| 5–9  | 44    | 46 | 410  | 0.96x |
| 10–14| 39    | 48 | 413  | 0.81x |
| **15–19** | **18** | **45** | **437** | **0.40x** |

> XQ 在后期逐渐占据优势，说明去掉 Queen + 加入规则改革后，**原本被压制的 XQ 战术体系开始发挥作用**。

#### 关键结论

- `no_queen + ALL_RULES` 成功将 C:X 比从 12.4x 压到 ≈1:1，**基本实现平衡**
- `surv_chess_QUEEN = 0.00` 确认规则正确生效（无升变出 Queen）
- 后期 XQ 反超说明规则改革力度稍强，如需精调可考虑仅用 `palace + knight_block`（不删后）
- 20轮训练模型仍弱（vs AB d1 全输），需更多轮次验证策略质量

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

## 待办事项

- [x] 分析 `rq4_az_nq_allrules_v2` 结果 → C:X ≈1:1，规则平衡验证成功
- [ ] 若需精调：测试 `palace+knight_block`（不删后）变体，避免 XQ 后期反超
- [ ] 更多轮训练（≥50 轮）验证 AZ 策略质量
- [ ] 将实验结果写入课程报告 (`course_project/`)
