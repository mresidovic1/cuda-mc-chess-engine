#pragma once

#include "include/chess.hpp"
#include <array>

namespace chess_engine {

constexpr int INFINITY_SCORE = 30000;
constexpr int MATE_SCORE = 10000;
constexpr int LMR_FULL_DEPTH_MOVES = 4;
constexpr int LMR_MIN_DEPTH = 3;
constexpr int FUTILITY_MARGIN = 100;
constexpr int DELTA_MARGIN = 100;
constexpr int MAX_QUIESCENCE_DEPTH = 20;
constexpr int MAX_SEE_DEPTH = 32;
constexpr int MAX_SEARCH_DEPTH = 100;
constexpr int CHECK_EXTENSION_DEPTH = 2;
constexpr int MAX_CHECK_EXTENSIONS = 2;
constexpr int RAZOR_MARGIN_BASE = 514;
constexpr int RAZOR_MARGIN_DEPTH = 294;
constexpr int FUTILITY_MARGIN_BASE = 91;
constexpr int FUTILITY_MARGIN_DEPTH_MULT = 21;

constexpr std::array<int, 6> piece_values = {100, 300, 320, 500, 900, 0};

constexpr std::array<int, 64> pawn_table = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

constexpr std::array<int, 64> knight_table = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

constexpr std::array<int, 64> bishop_table = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

constexpr std::array<int, 64> rook_table = {
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

constexpr std::array<int, 64> queen_table = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};

constexpr std::array<int, 64> king_table = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
};

// Helper functions for piece-square tables
inline int mirror_index_for_color(int index, chess::Color color) {
  return (color == chess::Color::WHITE) ? index : (index ^ 56);
}

inline int pstScore(chess::Bitboard bb, chess::Color color, const std::array<int, 64>& table) {
  int score = 0;
  chess::Bitboard temp = bb;
  while (temp) {
    chess::Square sq = temp.pop();
    int idx = mirror_index_for_color(static_cast<int>(sq.index()), color);
    score += table[idx];
  }
  return score;
}

constexpr std::array<const std::array<int, 64>*, 6> piece_square_tables = {
    &pawn_table, &knight_table, &bishop_table, &rook_table, &queen_table, &king_table};

inline const std::array<int, 64>& tableForPiece(chess::PieceType pt) {
  return *piece_square_tables[static_cast<int>(pt)];
}

}