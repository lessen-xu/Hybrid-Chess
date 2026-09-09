"""Rule catalog for local play; movement semantics stay in core.config."""
from dataclasses import fields
from hybrid.core.board import initial_board
from hybrid.core.config import VariantConfig
from hybrid.core.render import render_board


def bilingual(zh, en):
    return {"zh": zh, "en": en}


def rule(key, group, zh, en, zh_help, en_help):
    return {"key": key, "group": group, "name": bilingual(zh, en),
            "description": bilingual(zh_help, en_help)}


RULES = [
    rule("no_queen", "chess", "移除皇后", "Remove queen", "移除国际象棋的初始皇后。", "Remove the starting Chess queen."),
    rule("no_bishop", "chess", "移除左侧象", "Remove left bishop", "仅移除 c1 的象，保留另一枚。", "Remove only the bishop on c1."),
    rule("one_rook", "chess", "只保留一辆车", "One rook", "移除 h1 的车，保留 a1 的车。", "Remove the rook on h1; keep the rook on a1."),
    rule("remove_extra_pawn", "chess", "移除第九兵", "Remove ninth pawn", "移除 i2 的兵，保留八个兵。", "Remove the pawn on i2, leaving eight pawns."),
    rule("xq_queen", "xiangqi", "增加象棋皇后", "Xiangqi queen", "用象棋方的皇后替换 d10 的士；沿直线和斜线移动。", "Replace the advisor on d10 with a Xiangqi queen, moving along ranks, files and diagonals."),
    rule("extra_cannon", "xiangqi", "增加一门炮", "Extra cannon", "在 e8 增加第三门炮。", "Add a third cannon on e8."),
    rule("extra_soldier", "xiangqi", "增加一个卒", "Extra soldier", "在 e6 增加一个卒。", "Add a soldier on e6."),
    rule("chess_palace", "movement", "国际象棋王限于九宫", "Chess king palace", "王只能在 d1–f3 的九宫内移动。", "Confine the Chess king to the d1–f3 palace."),
    rule("knight_block", "movement", "国际象棋马也蹩腿", "Block Chess knight legs", "国际象棋马使用中国象棋的蹩马腿规则。", "Chess knights follow the Xiangqi horse's leg-blocking rule."),
    rule("no_promotion", "movement", "禁止兵升变", "Disable promotion", "兵到达底线后仍然是兵。", "Pawns remain pawns on the last rank."),
    rule("no_queen_promotion", "movement", "禁止升为皇后", "No queen promotion", "兵仅可升为车、象或马。", "Pawns can promote only to rook, bishop or knight."),
    rule("flying_general", "movement", "启用飞将", "Flying general", "将与王同列且中间无子时，将可以直接吃王。", "The general can capture the king across an unobstructed file."),
]


def preset(key, zh, en, zh_help, en_help, **flags):
    return {"id": key, "name": bilingual(zh, en),
            "description": bilingual(zh_help, en_help),
            "variant": VariantConfig(**flags).to_dict()}


PRESETS = [
    preset("none", "标准规则", "Original rules", "两套棋子，保留各自走法。", "Two armies, each with its own movement rules."),
    preset("pk", "宫与蹩马腿", "Palace & blocked knights", "限制王的活动范围，马也会被蹩腿。", "A palace for the king, blocked legs for knights.", chess_palace=True, knight_block=True),
    preset("xq_queen", "象棋加后", "A queen for Xiangqi", "用一枚皇后替换象棋的左士。", "Replace Xiangqi's left advisor with a queen.", xq_queen=True),
    preset("pk_xq_queen", "宫、蹩马腿与象棋后", "Palace, knights & queen", "组合行动限制与象棋皇后，探索新的平衡。", "Combine movement restrictions and a Xiangqi queen.", chess_palace=True, knight_block=True, xq_queen=True),
    preset("no_queen", "国际象棋去后", "Chess without a queen", "从没有皇后的国际象棋阵容开始。", "Start with the Chess queen removed.", no_queen=True),
    preset("extra_cannon", "象棋加炮", "An extra cannon", "为象棋增加一门中炮。", "Give Xiangqi a third, central cannon.", extra_cannon=True),
]


def parse_variant(value="none"):
    if isinstance(value, str):
        match = next((p for p in PRESETS if p["id"] == value), None)
        if match is None:
            raise ValueError("invalid_variant")
        return VariantConfig(**match["variant"])
    valid = {f.name for f in fields(VariantConfig)}
    if not isinstance(value, dict) or set(value) - valid:
        raise ValueError("invalid_variant")
    if any(type(v) is not bool for v in value.values()):
        raise ValueError("invalid_variant")
    config = VariantConfig(**value).to_dict()
    config["remove_extra_pawn"] |= not config["extra_pawn_i_file"]
    config["extra_pawn_i_file"] = True
    if config["no_promotion"]:
        config["no_queen_promotion"] = False
    return VariantConfig(**config)


def preview(value="none"):
    variant = parse_variant(value)
    # Resetting an environment would mutate the active rule globals.
    return {"variant": variant.to_dict(),
            "board_ascii": render_board(initial_board(variant=variant))}


def catalog():
    return {"presets": PRESETS, "rules": RULES,
            "defaults": VariantConfig().to_dict()}
