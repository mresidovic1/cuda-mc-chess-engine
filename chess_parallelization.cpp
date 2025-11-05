#include "include/chess.hpp"
#include <array>
#include <algorithm>
#include <iostream>
#include <vector>    
#include <chrono>    
#include <cstring>  

using namespace chess;

// --- Constants ---
const int INFINITY_SCORE = 30000;
const int MATE_SCORE = 10000;
const int TT_SIZE = 8388608; // 8M entries (~128MB)

// --- Transposition Table Entry ---
struct TTEntry {
    uint64_t key;
    int depth;
    int score;
    uint8_t flag; // 0 = exact, 1 = lower bound, 2 = upper bound
};

// --- Global Transposition Table ---
struct TranspositionTable {
    std::vector<TTEntry> table;
    
    TranspositionTable() : table(TT_SIZE) {
        std::memset(table.data(), 0, TT_SIZE * sizeof(TTEntry));
    }
    
    void clear() {
        std::memset(table.data(), 0, TT_SIZE * sizeof(TTEntry));
    }
    
    TTEntry* probe(uint64_t key) {
        return &table[key % TT_SIZE];
    }
    
    void store(uint64_t key, int depth, int score, uint8_t flag) {
        TTEntry* entry = &table[key % TT_SIZE];
        // Always replace (easier and faster)
        entry->key = key;
        entry->depth = depth;
        entry->score = score;
        entry->flag = flag;
    }
};

TranspositionTable tt;

// --- Piece values (remains as it was before the attempted fix) ---
// Because it is used only in the evaluate function, where the context is different.
// Assumes PAWN=0, KNIGHT=1, ..., KING=5 in the evaluate loop.
constexpr std::array<int, 6> piece_values = {100, 300, 320, 500, 900, 0}; 


// --- Evaluation function (reverted to previous version) ---
// This version worked before the issues with order_moves.
int evaluate(const chess::Board &board) {
  int white_material = 0;
  int black_material = 0;

  // The board.pieces() method accepts a chess::PieceType object.
  // Since PieceType::underlying is an enum class underlying type, you need to
  // create a PieceType object from it, as we did before.
  const std::array<chess::PieceType::underlying, 6> piece_types_to_count_underlying = {
      chess::PieceType::underlying::PAWN,   chess::PieceType::underlying::KNIGHT,
      chess::PieceType::underlying::BISHOP, chess::PieceType::underlying::ROOK,
      chess::PieceType::underlying::QUEEN,  chess::PieceType::underlying::KING};

  for (const auto &pt_underlying : piece_types_to_count_underlying) {
    // Assumes mapping to the piece_values array for these enums: PAWN=0 to KING=5.
    // If not, this will still be an issue, but outside of order_moves.
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

// --- OPTIMIZED Helper function for move sorting (simple but robust) ---
// Relies only on isCapture(), typeOf(), and inCheck(), not on PieceType values for indexing.
void order_moves(std::vector<Move> &moves, Board &board) {
    // Static scoring only: no make/unmake here.
    auto score_of = [&](const Move &m) -> int {
        int s = 0;
        if (board.isCapture(m)) s += 2000;        // Captures first
        if (m.typeOf() == Move::PROMOTION) s += 1500; // Promotions next
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

// --- Main Negamax function with Transposition Table ---
int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root) {
  // Transposition table lookup (only for depth >= 2)
  uint64_t zobrist_key = 0;
  if (depth >= 2) {
    zobrist_key = board.hash();
    TTEntry* tt_entry = tt.probe(zobrist_key);
    
    if (tt_entry->key == zobrist_key && tt_entry->depth >= depth) {
      int tt_score = tt_entry->score;
      
      // Adjust mate scores
      if (tt_score > MATE_SCORE - 1000) tt_score -= current_depth_from_root;
      if (tt_score < -MATE_SCORE + 1000) tt_score += current_depth_from_root;
      
      if (tt_entry->flag == 0) return tt_score; // Exact
      if (tt_entry->flag == 1 && tt_score >= beta) return tt_score; // Lower bound
      if (tt_entry->flag == 2 && tt_score <= alpha) return tt_score; // Upper bound
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

  order_moves(moves_to_search, board);

  int bestValue = -INFINITY_SCORE;
  int original_alpha = alpha;
  
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
  
  // Store in transposition table (only for depth >= 2)
  if (depth >= 2) {
    if (zobrist_key == 0) zobrist_key = board.hash();
    uint8_t flag = (bestValue <= original_alpha) ? 2 : (bestValue >= beta) ? 1 : 0;
    tt.store(zobrist_key, depth, bestValue, flag);
  }
  
  return bestValue;
}

// --- Function for finding the best move ---
Move find_best_move(Board& board, int max_depth) {
    Movelist movelist;
    movegen::legalmoves(movelist, board);
    
    std::vector<Move> initial_moves;
    initial_moves.reserve(movelist.size());
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

  Move best_move = find_best_move(board2, 8); 

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "\nBest Move found: " << chess::uci::moveToUci(best_move) << std::endl;
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;


  return 0;
}