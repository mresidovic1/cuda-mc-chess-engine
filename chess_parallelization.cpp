#include "../include/chess.hpp"
#include <array>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>

#include <iostream>

using namespace chess;

const int INFINITY_SCORE = 30000;
const int MATE_SCORE = 10000;

constexpr std::array<int, 6> piece_values = {100, 300, 320, 500, 900, 0}; 

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
  
  if (board.sideToMove() == chess::Color::WHITE) {
      evaluation += 10;
  } else {
      evaluation -= 10;
  }

  return evaluation;
}

void order_moves(std::vector<Move> &moves, const Board &board) {
    std::sort(moves.begin(), moves.end(), [&board](const Move& a, const Move& b) {
        bool a_is_capture = board.isCapture(a);
        bool b_is_capture = board.isCapture(b);

        if (a_is_capture && !b_is_capture) return true;
        if (!a_is_capture && b_is_capture) return false;

        return false; 
    });
}


int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root) {
  Movelist movelist;
  movegen::legalmoves(movelist, board);
  
  std::vector<Move> moves_to_search;
  for (const auto& m : movelist) {
      moves_to_search.push_back(m);
  }

  if (moves_to_search.empty()) {
    if (board.inCheck()) {
      return -MATE_SCORE + current_depth_from_root; 
    }
    return 0;
  }

  if (depth == 0) {
    return evaluate(board);
  }

  order_moves(moves_to_search, board);

  int bestValue = -INFINITY_SCORE;
  
  for (auto &move : moves_to_search) {
    board.makeMove(move);

    int score = -negamax(board, depth - 1, -beta, -alpha, current_depth_from_root + 1);
    
    board.unmakeMove(move);

    bestValue = std::max(bestValue, score);
    alpha = std::max(alpha, bestValue);

    if (alpha >= beta) {
      break;
    }
  }
  return bestValue;
}

Move find_best_move(Board& board, int max_depth) {
    Movelist movelist;
    movegen::legalmoves(movelist, board);
    
    std::vector<Move> initial_moves;
    for (const auto& m : movelist) {
        initial_moves.push_back(m);
    }
    order_moves(initial_moves, board);

    int best_score = -INFINITY_SCORE;
    Move best_move = Move::NO_MOVE;

    int alpha = -INFINITY_SCORE;
    int beta = INFINITY_SCORE;

    for (auto &move : initial_moves) {
        board.makeMove(move);
        int score = -negamax(board, max_depth - 1, -beta, -alpha, 1);
        board.unmakeMove(move);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        alpha = std::max(alpha, best_score);
    }
    
    std::cout << "Final Score: " << best_score << std::endl;

    return best_move;
}

int main() {
  attacks::initAttacks();
  
  Board board =
      Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Board board2 = 
        Board("4r1k1/r4p1p/p3bRpQ/q2pP3/2pP4/Bpn1R3/6PP/1B4K1 w - - 0 1"); // Best move is b1g6
  
  std::cout << "Initial Board:\n";
  std::cout << board2 << std::endl;
  
  auto start_time = std::chrono::high_resolution_clock::now();

  Move best_move = find_best_move(board, 9);

  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "\nBest Move found: " << chess::uci::moveToUci(best_move) << std::endl;
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;


  return 0;
}