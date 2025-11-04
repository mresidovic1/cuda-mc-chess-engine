#include "../include/chess.hpp"
#include <iostream>

using namespace chess;

const int INFINITY_SCORE = 30000;
const int MATE_SCORE = 10000;

constexpr std::array<int, 6> piece_values = {10, 30, 32, 50, 90, 0};

int evaluate(const chess::Board &board) {
  int white_material = 0;
  int black_material = 0;

  const std::array<chess::PieceType, 6> piece_types = {
      chess::PieceType::PAWN,   chess::PieceType::KNIGHT,
      chess::PieceType::BISHOP, chess::PieceType::ROOK,
      chess::PieceType::QUEEN,  chess::PieceType::KING};

  for (const auto &pt : piece_types) {
    int piece_value = piece_values[static_cast<int>(pt)];

    white_material +=
        board.pieces(pt, chess::Color::WHITE).count() * piece_value;
    black_material +=
        board.pieces(pt, chess::Color::BLACK).count() * piece_value;
  }

  int evaluation = white_material - black_material;
  return evaluation;
}
int minimax(Board &board, int depth, bool is_maximizing, int alpha, int beta) {
  Movelist movelist;
  movegen::legalmoves(movelist, board);
  if (movelist.empty()) {
    if (board.inCheck()) {
      return -MATE_SCORE;
    }

    return 0;
  }

  if (depth == 0) {
    return evaluate(board);
  }
  if (is_maximizing) {
    int bestValue = -INFINITY_SCORE;
    for (auto &move : movelist) {
      board.makeMove(move);
      int returned_score = minimax(board, depth - 1, false, alpha, beta);
      board.unmakeMove(move);
      bestValue = std::max(bestValue, returned_score);
      alpha = std::max(alpha, bestValue);
      if (beta <= alpha)
        break;
    }
    return bestValue;
  } else {
    int bestValue = INFINITY_SCORE;
    for (auto &move : movelist) {
      board.makeMove(move);
      int returned_score = minimax(board, depth - 1, true, alpha, beta);
      board.unmakeMove(move);
      bestValue = std::min(bestValue, returned_score);
      beta = std::min(beta, bestValue);
      if (beta <= alpha)
        break;
    }
    return bestValue;
  }
}

int main() {
  attacks::initAttacks();
  Board board =
      Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  int alpha = -INFINITY_SCORE;
  int beta = INFINITY_SCORE;
  int eval = minimax(board, 9, true, alpha, beta);
  std::cout << eval;

  return 0;
}
