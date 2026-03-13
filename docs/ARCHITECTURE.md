# Architecture Overview

This document describes the high-level architecture of Hybrid Chess, a production-grade AlphaZero implementation for asymmetric board games.

## System Architecture

```mermaid
graph TB
    subgraph UI["🎮 User Interface"]
        WebUI["Browser UI<br/>(HTML/JS/CSS)"]
        CLI["CLI Entry Point<br/>(__main__.py)"]
    end

    subgraph Core["🧩 Core Engine (hybrid/core/)"]
        Types["types.py<br/>Side, PieceKind, Piece, Move"]
        Board["board.py<br/>9×10 Grid, initial_board()"]
        Rules["rules.py<br/>Legal Moves, Terminal Detection"]
        Config["config.py<br/>VariantConfig, Constants"]
        Env["env.py<br/>HybridChessEnv"]
        FEN["fen.py<br/>FEN Parser/Serializer"]
        Render["render.py<br/>ASCII Renderer"]
    end

    subgraph Agents["🤖 Agents (hybrid/agents/)"]
        Base["base.py<br/>Agent ABC"]
        Random["RandomAgent"]
        Greedy["GreedyAgent"]
        AB["AlphaBetaAgent<br/>Negamax α-β + Eval"]
        AZStub["AlphaZeroMiniAgent<br/>MCTS + Neural Net"]
    end

    subgraph RL["🧠 RL Pipeline (hybrid/rl/)"]
        Network["az_network.py<br/>BaseModel ABC + PolicyValueNet"]
        Encoding["az_encoding.py<br/>State (14ch) + Action (92 planes)"]
        SelfPlay["az_selfplay.py<br/>Self-Play + Reward Shaping"]
        Train["az_train.py<br/>Policy CE + Value MSE"]
        Eval["az_eval.py<br/>Match Evaluation"]
        Runner["az_runner.py<br/>Iterative Runner + Logging"]
        Replay["az_replay.py<br/>Replay Buffer"]
        InfServer["az_inference_server.py<br/>GPU Batch Inference"]
        SHM["az_shm_pool.py<br/>Shared Memory IPC"]
        ParallelSP["az_selfplay_parallel.py<br/>Multi-Process Self-Play"]
        ParallelEval["az_eval_parallel.py<br/>Multi-Process Evaluation"]
        Endgame["endgame_spawner.py<br/>Curriculum Positions"]
    end

    subgraph CppEngine["⚡ C++ Engine (cpp/)"]
        CppBoard["Board + Rules"]
        CppAB["Alpha-Beta Search"]
        Bind["pybind11 Bindings"]
    end

    subgraph Gym["🏋️ Gymnasium"]
        GymEnv["gym_env.py<br/>HybridChess-v0"]
    end

    CLI --> Env
    CLI --> Runner
    WebUI --> |HTTP| Server["server.py"]
    Server --> Env
    Server --> Agents

    Env --> Board
    Env --> Rules
    Env --> Config
    Env --> FEN
    Board --> Types
    Rules --> Types

    Base --> Random & Greedy & AB & AZStub
    AZStub --> Encoding
    AZStub --> Network

    Runner --> SelfPlay
    Runner --> Train
    Runner --> Eval
    Runner --> ParallelSP
    Runner --> ParallelEval
    SelfPlay --> Env
    SelfPlay --> AZStub
    Train --> Network
    Train --> Replay
    ParallelSP --> InfServer
    InfServer --> SHM
    InfServer --> Network

    GymEnv --> Env

    Env -.-> |use_cpp=True| CppEngine
    AB -.-> |C++ search| CppEngine

    style Core fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style RL fill:#0f3460,stroke:#16213e,color:#e0e0e0
    style Agents fill:#533483,stroke:#16213e,color:#e0e0e0
    style CppEngine fill:#e94560,stroke:#16213e,color:#e0e0e0
    style UI fill:#16213e,stroke:#1a1a2e,color:#e0e0e0
    style Gym fill:#1a535c,stroke:#16213e,color:#e0e0e0
```

---

## Module Details

### Core Engine (`hybrid/core/`)

The game engine is the foundation. It implements the rules and state management for **asymmetric chess** — International Chess (bottom) vs Chinese Chess / Xiangqi (top) on a shared 9×10 board.

| Module | Responsibility |
|--------|---------------|
| **types.py** | Frozen dataclasses: `Side` (Chess/Xiangqi), `PieceKind` (13 kinds), `Piece`, `Move` |
| **board.py** | `Board` class (9×10 grid stored as `grid[y][x]`), `initial_board(variant)` setup |
| **rules.py** | `generate_legal_moves()`, `apply_move()`, `terminal_info()`, check detection |
| **config.py** | `VariantConfig` (frozen dataclass), board constants, legacy global flags |
| **env.py** | `HybridChessEnv` — gym-like API: `reset()`, `step()`, `legal_moves()` |
| **fen.py** | `parse_fen()` / `board_to_fen()` for arbitrary position setup |
| **render.py** | ASCII board renderer (uppercase = Chess, lowercase = Xiangqi) |

#### Key Design Decisions

- **Asymmetric piece types**: Chess and Xiangqi pieces have distinct `PieceKind` enums (e.g., `KNIGHT` vs `HORSE`), allowing each side to retain its native piece movement rules.
- **VariantConfig**: Frozen dataclass passed to `HybridChessEnv`, replacing mutable global flags. Thread-safe and serializable.
- **Dual engine**: Pure Python path for simplicity; optional C++ path (pybind11) for ~50× speedup.

### AlphaZero Pipeline (`hybrid/rl/`)

The RL pipeline implements a complete AlphaZero training system:

```mermaid
flowchart LR
    SP["🎮 Self-Play<br/>(MCTS + NN)"] --> |"(state, π, z)"| RB["📦 Replay<br/>Buffer"]
    RB --> |"mini-batches"| TR["📈 Train<br/>(CE + MSE)"]
    TR --> |"candidate model"| GA["🔒 Gating<br/>(Wilson CI)"]
    GA --> |"accept / reject"| BM["🏆 Best<br/>Model"]
    BM --> |"load weights"| SP
    BM --> |"evaluate"| EV["📊 Eval<br/>(vs Random, AB)"]

    style SP fill:#e94560,stroke:#16213e,color:#fff
    style RB fill:#533483,stroke:#16213e,color:#fff
    style TR fill:#0f3460,stroke:#16213e,color:#fff
    style GA fill:#1a535c,stroke:#16213e,color:#fff
    style BM fill:#f5a623,stroke:#16213e,color:#000
    style EV fill:#16213e,stroke:#533483,color:#fff
```

| Module | Responsibility |
|--------|---------------|
| **az_network.py** | `BaseModel` ABC + `PolicyValueNet` (dual-head ResNet: 14→64ch, 3 res blocks) |
| **az_encoding.py** | State → (14, 10, 9) tensor; Move → 92-plane action index |
| **az_selfplay.py** | Full self-play game loop with resign, draw adjudication, reward shaping hook |
| **az_train.py** | Training loop: policy cross-entropy + value MSE loss |
| **az_eval.py** | Match evaluation with win/draw/loss statistics, Wilson & score CI |
| **az_runner.py** | Iterative orchestrator: self-play → train → gate → eval, CSV + WandB logging |
| **az_replay.py** | Ring buffer with `.npz` serialization |
| **az_inference_server.py** | Centralized GPU inference: batches N workers' requests into one forward pass |
| **az_shm_pool.py** | Zero-copy shared memory pool for worker ↔ server communication |
| **az_selfplay_parallel.py** | Multi-process self-play via `torch.multiprocessing.spawn` |
| **az_eval_parallel.py** | Multi-process gating and evaluation matches |
| **endgame_spawner.py** | Generate random endgame positions for curriculum learning |

#### State & Action Encoding

```
State Encoding — 14 binary planes (10 × 9 each):
┌───────────────────────────────────────┐
│ Ch 0:  King positions                 │
│ Ch 1:  Queen positions                │
│ Ch 2:  Rook positions                 │
│ Ch 3:  Bishop positions               │
│ Ch 4:  Knight positions               │
│ Ch 5:  Pawn positions                 │
│ Ch 6:  General positions              │
│ Ch 7:  Advisor positions              │
│ Ch 8:  Elephant positions             │
│ Ch 9:  Horse positions                │
│ Ch 10: Chariot positions              │
│ Ch 11: Cannon positions               │
│ Ch 12: Soldier positions              │
│ Ch 13: Side-to-move (1 = Chess)       │
└───────────────────────────────────────┘

Action Space — 92 planes × 10 × 9 = 8,280 actions:
  Planes  0–71: Sliding (8 directions × 9 distances)
  Planes 72–79: Knight/Horse (8 L-shaped deltas)
  Planes 80–91: Promotions (3 dx × 4 piece types)
```

#### Neural Network Architecture

```
PolicyValueNet (default):
  Input: (B, 14, 10, 9)
    ↓
  Conv2d 3×3 (14 → 64) + BN + ReLU
    ↓
  3 × ResidualBlock (64ch: Conv → BN → ReLU → Conv → BN → Skip → ReLU)
    ↓
  ┌─── Policy Head ──────────────────┐  ┌─── Value Head ─────────────────┐
  │ Conv2d 1×1 (64 → 92)            │  │ Conv2d 1×1 (64 → 1) + BN      │
  │ Output: (B, 92, 10, 9) logits   │  │ → flatten → FC(90→64) → ReLU  │
  └──────────────────────────────────┘  │ → FC(64→1) → tanh             │
                                        │ Output: (B, 1) ∈ [-1, 1]      │
                                        └─────────────────────────────────┘
```

Users can subclass `BaseModel` to define custom architectures (Transformer, MobileNet, etc.).

### Performance Optimizations

The system uses multiple layers of acceleration:

```mermaid
graph LR
    subgraph Workers["CPU Workers (N processes)"]
        W1["Worker 1<br/>MCTS + Env"]
        W2["Worker 2<br/>MCTS + Env"]
        W3["Worker ..."]
    end

    subgraph SHM["Shared Memory Pool"]
        Buf["Pre-allocated<br/>Pinned Buffers"]
        Sig["8-byte Queue<br/>Signals"]
    end

    subgraph Server["GPU Inference Server"]
        Batch["Batch Collector<br/>(up to K leaves)"]
        GPU["Forward Pass<br/>(FP16 AMP)"]
    end

    W1 & W2 & W3 --> |"zero-copy write"| Buf
    Buf --> |"batch read"| Batch
    Batch --> GPU
    GPU --> |"scatter results"| Buf
    Buf --> |"signal done"| W1 & W2 & W3

    style Workers fill:#533483,stroke:#16213e,color:#e0e0e0
    style SHM fill:#e94560,stroke:#16213e,color:#e0e0e0
    style Server fill:#0f3460,stroke:#16213e,color:#e0e0e0
```

| Layer | Technique | Impact |
|-------|-----------|--------|
| **Engine** | C++ via pybind11 | ~50× faster move gen + terminal detection |
| **Inference** | Centralized GPU server, batched forward passes | Amortize GPU kernel launch across N workers |
| **Memory** | `SharedMemoryPool`, zero-copy, pinned DMA buffers | Eliminate pickling overhead |
| **MCTS** | Leaf batching with virtual loss | K leaves evaluated in one GPU call |
| **Precision** | FP16 autocast + TF32 matmul (Ampere+) | ~2× inference throughput |
| **Encoding** | `encode_batch_gpu()` — scatter-based one-hot on GPU | Zero-allocation hot path |
| **Parallelism** | `torch.multiprocessing.spawn` | Scales linearly with CPU cores |

### Agents (`hybrid/agents/`)

All agents implement the `Agent` ABC from `base.py`:

```python
class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        ...
```

| Agent | Strategy | Typical Use |
|-------|----------|-------------|
| `RandomAgent` | Uniform random over legal moves | Baseline / smoke testing |
| `GreedyAgent` | 1-ply capture maximizer | Fast baseline |
| `AlphaBetaAgent` | Negamax α-β + hand-crafted eval | Non-learned baseline (depth 1–4) |
| `AlphaZeroMiniAgent` | MCTS + PolicyValueNet | RL-trained agent |

### Web Interface (`ui/`)

The browser-based UI is served by a zero-dependency HTTP server (`server.py`):

- **Landing page** (`ui/index.html`): Choose game mode
- **Play** (`ui/play/`): Interactive human-vs-AI with move highlighting
- **Replay** (`ui/replay/`): Review recorded games move-by-move
- **Shared** (`ui/shared/`): Common board renderer and CSS

### Gymnasium Integration (`gym_env.py`)

Standard Gymnasium wrapper registered as `HybridChess-v0`:

```
Observation: Box(14, 10, 9)  — binary piece planes + side-to-move
Action:      Discrete(8280)  — 92 × 10 × 9 flat action space
Reward:      +1 / 0 / -1    — win / draw / loss from mover's perspective
Info:        {"legal_actions": [...], "side_to_move": "chess", "ply": 0}
```

Compatible with any RL framework that supports Gymnasium (Stable-Baselines3, CleanRL, RLlib, etc.).

---

## Data Flow: One Training Iteration

```mermaid
sequenceDiagram
    participant R as Runner
    participant SP as Self-Play
    participant E as Env
    participant M as MCTS
    participant N as Neural Net
    participant B as Replay Buffer
    participant T as Trainer

    R->>SP: Start N games
    loop Each game
        SP->>E: reset()
        loop Until terminal
            SP->>E: legal_moves()
            SP->>M: search(state, legal_moves)
            M->>N: evaluate(states)
            N-->>M: (policy, value)
            M-->>SP: (move, π, root_value)
            SP->>E: step(move)
        end
        SP-->>B: append(examples)
    end
    R->>T: train(buffer, net)
    T->>B: sample mini-batches
    T->>N: loss = CE(π) + MSE(z)
    T->>N: backprop + update
    R->>R: Gating (candidate vs best)
    R->>R: Eval (vs Random, vs AB)
```
