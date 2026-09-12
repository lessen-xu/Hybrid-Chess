# Game rules

Hybrid Chess puts International Chess and Xiangqi armies on the same board.
These are the rules implemented by this repository. Custom settings below can
change the starting pieces and selected movement rules.

## Board and opening

The board has nine files, **a–i**, and ten ranks, **1–10**. Coordinates stay the
same when the board is rotated. Chess starts on ranks 1–2 and moves toward rank
10; Xiangqi starts on ranks 7–10 and moves toward rank 1. **Chess moves first.**

The river separates ranks 5 and 6. Xiangqi's palace covers d8–f10. A Chess palace
covering d1–f3 exists only when its optional rule is enabled.

| Army | Starting pieces |
| --- | --- |
| Chess | Rook, knight, bishop, queen, king, bishop, knight, rook on a1–h1; i1 is empty. Nine pawns on a2–i2. |
| Xiangqi | Chariot, horse, elephant, advisor, general, advisor, elephant, horse, chariot on a10–i10. Cannons on b8 and h8; soldiers on a7, c7, e7, g7 and i7. |

## Moving and capturing

Players alternate one move at a time. A piece cannot land on a friendly piece.
Sliding pieces cannot pass through occupied squares, except for the cannon's
capture described below. A legal move must leave your own king or general safe.

### Chess pieces

| Piece | Movement |
| --- | --- |
| King | One square in any direction. There is no castling. |
| Queen | Any distance along a rank, file or diagonal. |
| Rook | Any distance along a rank or file. |
| Bishop | Any distance diagonally. |
| Knight | Two squares in one direction and one perpendicular to it. It jumps over other pieces unless leg blocking is enabled. |
| Pawn | One square forward into an empty square. From rank 2 it may advance two squares if both are empty. It captures one square diagonally forward. There is no en passant. |

A pawn reaching rank 10 must promote to a queen, rook, bishop or knight unless a
promotion restriction is enabled. The player chooses the piece. Promotion is
available regardless of which starting pieces have been captured or removed.

### Xiangqi pieces

| Piece | Movement |
| --- | --- |
| General | One square along a rank or file, staying in d8–f10. It can also use the flying-general capture. |
| Advisor | One square diagonally, staying in d8–f10. |
| Elephant | Two squares diagonally. The intervening diagonal square must be empty. It stays on ranks 6–10. |
| Horse | An L-shaped move like the Chess knight, blocked if the adjacent square in the two-square direction is occupied. |
| Chariot | Any distance along a rank or file, like a Chess rook. |
| Cannon | Moves along an unobstructed rank or file without capturing. To capture, it must jump over exactly one intervening piece, of either army, and land on an enemy piece. |
| Soldier | One square forward. After reaching rank 5 or lower, it may also move one square sideways. It never moves backward and does not promote. |
| Xiangqi queen, optional | Moves like a Chess queen and belongs to the Xiangqi army. It replaces the advisor on d10 and is not confined to the palace. |

With **flying general** enabled, the Xiangqi general attacks and can capture the
Chess king anywhere along an unobstructed shared file. This attack counts when
checking whether the Chess king is safe. It does not give the Chess king the same
long-range capture.

## Ending a game (Canonical Baseline)

| Condition | Result |
| --- | --- |
| The opponent's king or general is captured | The capturing army wins. |
| The player to move has no legal move and is in check | Checkmate: that player loses. |
| The player to move has no legal move and is not in check | Stalemate: that player **loses** under default baseline rules (Xiangqi convention), or **draws** under FIDE stalemate rules (`stalemate_rule="draw"`). |
| The same board and side to move occur three times | Draw. |
| 400 half-moves have been played without another ending | Draw. |
| A player resigns | The opponent wins. |

One half-move is one move by either player. Repetition uses the board and side to
move; there are no additional perpetual-check or chase adjudications. The game
does not apply the Chess fifty-move or insufficient-material draw rules.

## Rule Architecture & Three-Tier Ontology

To support both casual play and rigorous balance research, rules are organized into three tiers:

### 1. Canonical Baseline Rules
The standard initial game setup: original full armies, Chess moves first (`first_side="chess"`), stalemate is loss for the player with no legal moves (`stalemate_rule="loss"`), and flying general is enabled.

### 2. Experimental Rule Dimensions
Orthogonal parameter switches exposed via `VariantConfig`:

| Dimension / Switch | Options / Type | Baseline Default | Research Effect |
| --- | --- | --- | --- |
| Stalemate rule / `stalemate_rule` | `"loss"` \| `"draw"` | `"loss"` | `"draw"` adopts FIDE rules, providing Xiangqi defensive resilience (-7.3% Chess edge). |
| First side / `first_side` | `"chess"` \| `"xiangqi"` | `"chess"` | Alternates tempo advantage between armies. |
| Chess king palace / `chess_palace` | `bool` | `False` | Confines Chess king to d1–f3, preventing board-wide king flight (-8.9% Chess edge). |
| Block knight legs / `knight_block` | `bool` | `False` | Applies Xiangqi horse leg-blocking to Chess knights. |
| Remove queen / `no_queen` | `bool` | `False` | Removes the starting Chess queen on d1 (-14.1% Chess edge). |
| Remove left bishop / `no_bishop` | `bool` | `False` | Removes bishop on c1; retains bishop on f1. |
| One rook / `one_rook` | `bool` | `False` | Removes rook on h1; retains rook on a1. |
| Remove ninth pawn / `remove_extra_pawn` | `bool` | `False` | Removes pawn on i2, leaving eight pawns. |
| Xiangqi queen / `xq_queen` | `bool` | `False` | Replaces the advisor on d10 with a Xiangqi queen. |
| Extra cannon / `extra_cannon` | `bool` | `False` | Adds a third cannon on e8. |
| Extra soldier / `extra_soldier` | `bool` | `False` | Adds a soldier on e6. |
| Disable promotion / `no_promotion` | `bool` | `False` | Pawns remain pawns on rank 10 and cannot move farther. |
| No queen promotion / `no_queen_promotion` | `bool` | `False` | Allows promotion only to rook, bishop or knight. |
| Flying general / `flying_general` | `bool` | `True` | General attacks king across unobstructed shared file. |
| Repetition rule / `repetition_rule` | `"draw"` \| `"loss"` | `"draw"` | Canonical threefold repetition adjudication. |

### 3. Presets & Research Benchmarks

| Preset ID | Name | Category | Key Configuration Changes |
| --- | --- | --- | --- |
| `none` | Original rules | Baseline | Baseline standard rules (no modifications). |
| `golden_palace_draw` | Golden Balanced Variant | Research Candidate | `chess_palace=True`, `stalemate_rule="draw"` (Minimal intervention, ~50/50 balance). |
| `pk_xq_queen` | Palace, knights & queen | Heuristic Composite | `chess_palace=True`, `knight_block=True`, `xq_queen=True`. |
| `pk` | Palace & blocked knights | Movement Restriction | `chess_palace=True`, `knight_block=True`. |
| `xq_queen` | A queen for Xiangqi | Asymmetric Piece | `xq_queen=True`. |
| `no_queen` | Chess without a queen | Asymmetric Piece | `no_queen=True`. |
| `extra_cannon` | An extra cannon | Asymmetric Piece | `extra_cannon=True`. |

Disabling promotion makes the separate queen-promotion restriction irrelevant;
the UI disables it and configuration parsing clears it. Removing the starting
queen alone does **not** prohibit promoting a pawn to a queen.

Pass a configuration to `HybridChessEnv(variant=...)` as shown in the
[README](README.md#develop). A game's configuration stays fixed until a new game.

