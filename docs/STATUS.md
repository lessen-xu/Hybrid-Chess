# Round 02 Execution Status

**Goal**: Option A — Balanced Sampling & Clean Zero-Sum AlphaZero Training  
**Ledger**: `/storage/homefs/lx24y045/hybrid-chess/outputs/round02/budget.json`  
**Release**: `/storage/homefs/lx24y045/hybrid-chess/releases/round02-r01`  
**Git Base**: `main` (commit `46578fc`)

---

## 1. Pipeline Stages

| Stage | Job Name | Target | Specs | Status | Notes |
|---|---|---|---|---|---|
| **Stage 1: Teacher Dataset** | `hc2-teacher01` | 50,000 samples | 14 CPUs, 32GB RAM, partition `epyc2` | **QUEUED** (JobID: `14958307`) | Zero-sum evaluated, unbiased self-play games |
| **Stage 2: GPU Training** | `hc2-train01` | 10 supervised epochs + iterative MCTS | 1x H100, 16 CPUs, 64GB RAM, partition `gpu` | **PENDING** | Uses `BalancedBuffer` (uniform over 6 strata) |
| **Stage 3: Evaluation** | `hc2-eval01` | 20 games vs Minimax Teacher | 15 CPUs, 32GB RAM, partition `epyc2` | **PENDING** | Measure White/Chess win-rate balance (target: 45%-55%) |
| **Stage 4: Archival & Audit** | `hc2-archive01` | Archive checkpoint & reconcile ledger | Submit node / login | **PENDING** | Finalize ledger, freeze candidate weights |

---

## 2. Resource Budget Ledger

| Resource | Round 02 Limit | Allocated / Reserved | Remaining Headroom | Account |
|---|---|---|---|---|
| **GPU Card-Seconds** | 28,800 s (8.0 h) | 0 s (0.0 h) | 28,800 s (8.0 h) | `gratis` |
| **CPU Core-Seconds** | 230,400 core-s (64.0 h) | 100,800 core-s (28.0 h reserved) | 129,600 core-s (36.0 h) | `gratis` |

---

## 3. Patrol & Inspection Log

| Timestamp (CEST) | Job ID | Job Name | Slurm State | Priority | Sched / Node | Details |
|---|---|---|---|---|---|---|
| 2026-09-12 10:51 | 14958307 | `hc2-teacher01` | PENDING | 1105 | `bnode032` (15:35) | Initial submission via `cluster_submit.py` |
| 2026-09-12 11:00 | 14958307 | `hc2-teacher01` | PENDING | 1108 | `bnode032` (15:35) | Rank 1 non-blocked pending job on `epyc2` |
| 2026-09-12 11:05 | 14958307 | `hc2-teacher01` | PENDING | 1111 | `bnode055` (14:30) | Backfill window advanced by 65 min |
| 2026-09-12 11:10 | 14958307 | `hc2-teacher01` | PENDING | 1115 | `bnode055` (14:30) | Steady queue progression |
| 2026-09-12 11:15 | 14958307 | `hc2-teacher01` | PENDING | 1118 | `bnode056` (14:30) | Dynamic reassignment to `bnode056` |
| 2026-09-12 11:20 | 14958307 | `hc2-teacher01` | PENDING | 1122 | `bnode032` (15:35) | Backfill evaluation update |

---

## 4. Key Architectural Upgrades in Round 02

1. **Exact Anti-Symmetry**: Eliminated legacy `-8.0` anti-stalemate distortion; enforced strict antisymmetry $v_{\text{Chess}} = -v_{\text{Xiangqi}}$ in both Python (`hybrid/agents/eval.py`) and C++ engine (`cpp/src/ab_search.cpp`).
2. **Balanced Stratified Buffer**: `BalancedBuffer` partitions samples across the 6 strata: $(Side \times Outcome)$ where $Side \in \{\text{Chess}, \text{Xiangqi}\}$ and $Outcome \in \{-1, 0, +1\}$, drawing uniformly across available strata during batch training.
3. **Repository Pruning**: Removed obsolete course-project files (RQ1–RQ4 scripts, legacy ablation tests, and old static diagnostic dumps).
