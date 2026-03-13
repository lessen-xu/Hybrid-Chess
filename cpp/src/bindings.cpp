// bindings.cpp — pybind11 bindings for the C++ hybrid chess engine.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "types.h"
#include "board.h"
#include "rules.h"
#include "ab_search.h"

namespace py = pybind11;

PYBIND11_MODULE(hybrid_cpp_engine, m) {
    m.doc() = "C++ hybrid chess engine (pybind11)";
    py::enum_<Side>(m, "Side")
        .value("CHESS", Side::CHESS)
        .value("XIANGQI", Side::XIANGQI)
        .export_values();

    m.def("opponent", &opponent, "Return the opposing side");
    py::enum_<PieceKind>(m, "PieceKind")
        .value("KING", PieceKind::KING)
        .value("QUEEN", PieceKind::QUEEN)
        .value("ROOK", PieceKind::ROOK)
        .value("BISHOP", PieceKind::BISHOP)
        .value("KNIGHT", PieceKind::KNIGHT)
        .value("PAWN", PieceKind::PAWN)
        .value("GENERAL", PieceKind::GENERAL)
        .value("ADVISOR", PieceKind::ADVISOR)
        .value("ELEPHANT", PieceKind::ELEPHANT)
        .value("HORSE", PieceKind::HORSE)
        .value("CHARIOT", PieceKind::CHARIOT)
        .value("CANNON", PieceKind::CANNON)
        .value("SOLDIER", PieceKind::SOLDIER)
        .value("NONE", PieceKind::NONE)
        .export_values();
    py::class_<Piece>(m, "Piece")
        .def(py::init<PieceKind, Side>(), py::arg("kind"), py::arg("side"))
        .def_readwrite("kind", &Piece::kind)
        .def_readwrite("side", &Piece::side)
        .def("__eq__", &Piece::operator==)
        .def("__repr__", [](const Piece& p) {
            auto sk = [](PieceKind k) -> const char* {
                switch(k) {
                    case PieceKind::KING: return "KING"; case PieceKind::QUEEN: return "QUEEN";
                    case PieceKind::ROOK: return "ROOK"; case PieceKind::BISHOP: return "BISHOP";
                    case PieceKind::KNIGHT: return "KNIGHT"; case PieceKind::PAWN: return "PAWN";
                    case PieceKind::GENERAL: return "GENERAL"; case PieceKind::ADVISOR: return "ADVISOR";
                    case PieceKind::ELEPHANT: return "ELEPHANT"; case PieceKind::HORSE: return "HORSE";
                    case PieceKind::CHARIOT: return "CHARIOT"; case PieceKind::CANNON: return "CANNON";
                    case PieceKind::SOLDIER: return "SOLDIER"; default: return "?";
                }
            };
            auto ss = p.side == Side::CHESS ? "CHESS" : "XIANGQI";
            return std::string("Piece(") + sk(p.kind) + ", " + ss + ")";
        });
    py::class_<Move>(m, "Move")
        .def(py::init<int,int,int,int,PieceKind>(),
             py::arg("fx"), py::arg("fy"), py::arg("tx"), py::arg("ty"),
             py::arg("promotion") = PieceKind::NONE)
        .def_readwrite("fx", &Move::fx)
        .def_readwrite("fy", &Move::fy)
        .def_readwrite("tx", &Move::tx)
        .def_readwrite("ty", &Move::ty)
        .def_readwrite("promotion", &Move::promotion)
        .def("from_sq", [](const Move& m) { return py::make_tuple(m.fx, m.fy); })
        .def("to_sq",   [](const Move& m) { return py::make_tuple(m.tx, m.ty); })
        .def("__eq__", &Move::operator==)
        .def("__repr__", [](const Move& mv) {
            std::string s = "Move(" + std::to_string(mv.fx) + "," +
                std::to_string(mv.fy) + "→" + std::to_string(mv.tx) + "," +
                std::to_string(mv.ty);
            if (mv.promotion != PieceKind::NONE)
                s += ",promo";
            s += ")";
            return s;
        });
    py::class_<Board>(m, "Board")
        .def_static("empty", &Board::empty)
        .def("clone", &Board::clone)
        .def("in_bounds", &Board::in_bounds)
        .def("get", &Board::get, py::arg("x"), py::arg("y"))
        .def("set", [](Board& b, int x, int y, std::optional<Piece> p) {
            b.set(x, y, p);
        }, py::arg("x"), py::arg("y"), py::arg("piece"))
        .def("move_piece", &Board::move_piece)
        .def("iter_pieces", &Board::iter_pieces)
        .def("board_hash", &Board::board_hash, py::arg("side_to_move"))
        .def("zobrist_key_hex", &Board::zobrist_key_hex, py::arg("side_to_move"))
        .def("zobrist_key_hex_recompute", &Board::zobrist_key_hex_recompute, py::arg("side_to_move"))
        .def("royal_square", &Board::royal_square, py::arg("side"))
        .def("has_royal", &Board::has_royal, py::arg("side"))
        .def("royal_square_recompute", &Board::royal_square_recompute, py::arg("side"));
    m.def("generate_pseudo_legal_moves", &generate_pseudo_legal_moves);
    m.def("generate_legal_moves", &generate_legal_moves);
    m.def("apply_move", &apply_move);
    m.def("is_square_attacked", &is_square_attacked);
    m.def("is_square_attacked_slow", &is_square_attacked_slow);
    m.def("is_square_attacked_fast", &is_square_attacked_fast);
    m.def("is_in_check", &is_in_check);
    m.def("perft_nodes", &perft_nodes);

    // terminal_info: returns a GameInfo struct
    py::class_<GameInfo>(m, "GameInfo")
        .def_readonly("status", &GameInfo::status)
        .def_readonly("winner", &GameInfo::winner)
        .def_readonly("reason", &GameInfo::reason);

    m.def("terminal_info", &terminal_info,
          py::arg("board"), py::arg("side_to_move"),
          py::arg("repetition_table"), py::arg("ply"), py::arg("max_plies"));
    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("best_move", &SearchResult::best_move)
        .def_readonly("score",     &SearchResult::score)
        .def_readonly("nodes",     &SearchResult::nodes);

    m.def("best_move", &best_move,
          py::arg("board"), py::arg("side_to_move"), py::arg("depth"),
          py::arg("repetition_table"), py::arg("ply"), py::arg("max_plies"));
    m.attr("BOARD_W") = BOARD_W;
    m.attr("BOARD_H") = BOARD_H;
    m.attr("MAX_PLIES") = MAX_PLIES;
}
