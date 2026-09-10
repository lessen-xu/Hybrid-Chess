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

## Ending a game

| Condition | Result |
| --- | --- |
| The opponent's king or general is captured | The capturing army wins. |
| The player to move has no legal move and is in check | Checkmate: that player loses. |
| The player to move has no legal move and is not in check | Stalemate: that player **loses**, following the Xiangqi convention. |
| The same board and side to move occur three times | Draw. |
| 400 half-moves have been played without another ending | Draw. |
| A player resigns | The opponent wins. |

One half-move is one move by either player. Repetition uses the board and side to
move; there are no additional perpetual-check or chase adjudications. The game
does not apply the Chess fifty-move or insufficient-material draw rules.

## Presets and custom rules

The six presets are starting points for exploration, with no guaranteed balance
between armies. Every preset begins with the original rules and applies the
changes listed here:

| Preset | Enabled changes |
| --- | --- |
| Original rules | None. |
| Palace & blocked knights | `chess_palace`, `knight_block` |
| A queen for Xiangqi | `xq_queen` |
| Palace, knights & queen | `chess_palace`, `knight_block`, `xq_queen` |
| Chess without a queen | `no_queen` |
| An extra cannon | `extra_cannon` |

Custom settings expose twelve effective switches:

| Setting / configuration field | Effect when enabled |
| --- | --- |
| Remove queen / `no_queen` | Remove the Chess queen on d1. |
| Remove left bishop / `no_bishop` | Remove only the bishop on c1; the bishop on f1 remains. |
| One rook / `one_rook` | Remove the rook on h1; keep the rook on a1. |
| Remove ninth pawn / `remove_extra_pawn` | Remove the pawn on i2, leaving eight pawns. |
| Xiangqi queen / `xq_queen` | Replace the advisor on d10 with a Xiangqi queen. |
| Extra cannon / `extra_cannon` | Add a third cannon on e8. |
| Extra soldier / `extra_soldier` | Add a soldier on e6. |
| Chess king palace / `chess_palace` | Confine the Chess king to d1–f3. |
| Block Chess knight legs / `knight_block` | Apply the Xiangqi horse's leg-blocking rule to Chess knights. |
| Disable promotion / `no_promotion` | A pawn remains a pawn on rank 10 and cannot move farther. |
| No queen promotion / `no_queen_promotion` | Allow promotion only to rook, bishop or knight. |
| Flying general / `flying_general` | Enable the general's unobstructed-file attack on the Chess king. This is on by default. |

Disabling promotion makes the separate queen-promotion restriction irrelevant;
the UI disables it and configuration parsing clears it. Removing the starting
queen alone does **not** prohibit promoting a pawn to a queen.

For Python users, `VariantConfig` also retains `extra_pawn_i_file` for compatibility.
Setting it to `False` has the same effect as `remove_extra_pawn=True`; the web and
general-AI interfaces normalize both into the single remove-ninth-pawn switch.
Pass a configuration to `HybridChessEnv(variant=...)` as shown in the
[README](README.md#develop). A game's configuration stays fixed until a new game.
