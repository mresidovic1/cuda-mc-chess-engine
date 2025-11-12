#include "include/chess.hpp"
#include "killer_move.h"
#include "transposition_table.h"
#include <array>
#include <algorithm>
#include <iostream>
#include <vector>    
#include <chrono>    

using namespace chess;

const int INFINITY_SCORE = 30000;
const int MATE_SCORE = 10000;

KillerMoves killer_moves;

TranspositionTable tt;

constexpr std::array<int, 6> piece_values = {100, 300, 320, 500, 900, 0}; 


int evaluate(const chess::Board &board) {
  int white_material = 0;
  int black_material = 0;

  const std::array<chess::PieceType::underlying, 6> piece_types_to_count_underlying = {
      chess::PieceType::underlying::PAWN,   chess::PieceType::underlying::KNIGHT,
      chess::PieceType::underlying::BISHOP, chess::PieceType::underlying::ROOK,
      chess::PieceType::underlying::QUEEN,  chess::PieceType::underlying::KING};

  for (const auto &pt_underlying : piece_types_to_count_underlying) {
    int piece_value = piece_values[static_cast<int>(pt_underlying)]; 

    white_material +=
        board.pieces(chess::PieceType(pt_underlying), chess::Color::WHITE).count() * piece_value;
    black_material +=
        board.pieces(chess::PieceType(pt_underlying), chess::Color::BLACK).count() * piece_value;
  }

  int evaluation = white_material - black_material;
  
  if (board.sideToMove() == chess::Color::WHITE) {
      evaluation += 10;
  } else {
      evaluation -= 10;
  }

  return evaluation;
}

void order_moves(std::vector<Move> &moves, Board &board, int depth) {
    auto score_of = [&](const Move &m) -> int {
        int s = 0;
        if (board.isCapture(m)) s += 2000;
        if (m.typeOf() == Move::PROMOTION) s += 1500;
        if (killer_moves.isKiller(depth, m)) s += 1000;
        return s;
    };

    std::stable_sort(
        moves.begin(),
        moves.end(),
        [&](const Move &a, const Move &b) {
            return score_of(a) > score_of(b);
        }
    );
}

int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root) {
  uint64_t zobrist_key = 0;
  if (depth >= 2) {
    zobrist_key = board.hash();
    TTEntry* tt_entry = tt.probe(zobrist_key);
    
    if (tt_entry->key == zobrist_key && tt_entry->depth >= depth) {
      int tt_score = tt_entry->score;
      
      if (tt_score > MATE_SCORE - 1000) tt_score -= current_depth_from_root;
      if (tt_score < -MATE_SCORE + 1000) tt_score += current_depth_from_root;
      
      if (tt_entry->flag == 0) return tt_score;
      if (tt_entry->flag == 1 && tt_score >= beta) return tt_score;
      if (tt_entry->flag == 2 && tt_score <= alpha) return tt_score;
    }
  }
  
  Movelist movelist;
  movegen::legalmoves(movelist, board);
  
  std::vector<Move> moves_to_search;
  moves_to_search.reserve(movelist.size());
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

  int material_count = board.occ().count();
  bool in_endgame = (material_count <= 6);

  if (depth >= 3 && !board.inCheck() && !in_endgame && beta < MATE_SCORE - 1000) {
    const int R = (depth >= 6) ? 3 : 2;
    const int null_depth = depth - 1 - R;
    
    if (null_depth >= 0) {
      board.makeNullMove();
      int null_score = -negamax(board, null_depth, -beta, -beta + 1, current_depth_from_root + 1);
      board.unmakeNullMove();
      
      if (null_score >= beta) {
        return null_score;
      }
    }
  }
  
  order_moves(moves_to_search, board, depth);

  int bestValue = -INFINITY_SCORE;
  int original_alpha = alpha;
  
  bool first_move = true;

  for (auto &move : moves_to_search) {
    bool is_capture = board.isCapture(move);
    board.makeMove(move);

    int score;

    if(first_move) {
      score = -negamax(board, depth - 1, -beta, -alpha, current_depth_from_root + 1);
      first_move = false;
    } else {
      score = -negamax(board, depth - 1, -alpha - 1, - alpha, current_depth_from_root + 1);
      if(score > alpha && score < beta) {
        score = -negamax(board, depth - 1, -beta, -score, current_depth_from_root + 1);
      }
    }

    board.unmakeMove(move);

    bestValue = std::max(bestValue, score);

    alpha = std::max(alpha, bestValue);

    if(alpha >= beta) {
      if (!is_capture) {
        killer_moves.addKiller(depth, move);
      }
      break;
    }

  }
 
  if (depth >= 2) {
    if (zobrist_key == 0) zobrist_key = board.hash();
    uint8_t flag = (bestValue <= original_alpha) ? 2 : (bestValue >= beta) ? 1 : 0;
    tt.store(zobrist_key, depth, bestValue, flag);
  }
  
  return bestValue;
}

Move find_best_move(Board& board, int max_depth) {
    Movelist movelist;
    movegen::legalmoves(movelist, board);
    
    std::vector<Move> initial_moves;
    initial_moves.reserve(movelist.size());
    for (const auto& m : movelist) {
        initial_moves.push_back(m);
    }
    order_moves(initial_moves, board, max_depth);

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
        Board("4r1k1/r4p1p/p3bRpQ/q2pP3/2pP4/Bpn1R3/6PP/1B4K1 w - - 0 1");
  
  std::cout << "Initial Board:\n";
  std::cout << board2 << std::endl;
  
  auto start_time = std::chrono::high_resolution_clock::now();

  Move best_move = find_best_move(board2, 8); 

  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "\nBest Move found: " << chess::uci::moveToUci(best_move) << std::endl;
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;


  return 0;
}