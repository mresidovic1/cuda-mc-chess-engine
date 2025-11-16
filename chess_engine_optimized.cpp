#include "include/chess.hpp"
#include "killer_move.h"
#include "tt_parallel.h"
#include "thread_local_data.h"
#include <array>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <atomic>
#include <omp.h>

using namespace chess;

const int INFINITY_SCORE = 30000;
const int MATE_SCORE = 10000;
const int MAX_THREADS = 256;

// Transposition table - HUGE for better hit rate
TTParallel tt(512); // 512 MB

// Thread pool - persistent threads
std::vector<ThreadLocalData> thread_data;

// Evaluation - ultra fast
constexpr std::array<int, 6> piece_values = {100, 320, 330, 500, 900, 0};

inline int evaluate(const Board &board) {
  int score = 0;
  
  // Material counting - branchless
  score += (board.pieces(PieceType::PAWN, Color::WHITE).count() - 
            board.pieces(PieceType::PAWN, Color::BLACK).count()) * 100;
  score += (board.pieces(PieceType::KNIGHT, Color::WHITE).count() - 
            board.pieces(PieceType::KNIGHT, Color::BLACK).count()) * 320;
  score += (board.pieces(PieceType::BISHOP, Color::WHITE).count() - 
            board.pieces(PieceType::BISHOP, Color::BLACK).count()) * 330;
  score += (board.pieces(PieceType::ROOK, Color::WHITE).count() - 
            board.pieces(PieceType::ROOK, Color::BLACK).count()) * 500;
  score += (board.pieces(PieceType::QUEEN, Color::WHITE).count() - 
            board.pieces(PieceType::QUEEN, Color::BLACK).count()) * 900;
  
  return (board.sideToMove() == Color::WHITE) ? score : -score;
}

// Move ordering - critical for speed
inline void order_moves(std::vector<Move> &moves, Board &board, int depth, 
                        ThreadLocalData* tld, Move best_prev = Move::NO_MOVE) {
    // Hash move first
    if (best_prev != Move::NO_MOVE) {
        auto it = std::find(moves.begin(), moves.end(), best_prev);
        if (it != moves.end() && it != moves.begin()) {
            std::iter_swap(moves.begin(), it);
        }
    }
    
    // Score remaining moves
    auto score_of = [&](const Move &m) -> int {
        if (m == best_prev) return 100000;
        int s = 0;
        if (board.isCapture(m)) s += 10000;
        if (m.typeOf() == Move::PROMOTION) s += 9000;
        if (tld && tld->killer_moves.isKiller(depth, m)) s += 8000;
        return s;
    };

    std::stable_sort(moves.begin() + (best_prev != Move::NO_MOVE ? 1 : 0), moves.end(),
        [&](const Move &a, const Move &b) { return score_of(a) > score_of(b); });
}

// Negamax with ALL optimizations
int negamax(Board& board, int depth, int alpha, int beta, int ply, 
            ThreadLocalData* tld, bool allow_null = true) {
  
  if (tld) tld->increment_nodes();
  
  // TT probe
  uint64_t hash_key = 0;
  if (depth >= 1) {
    hash_key = board.hash();
    int tt_score, tt_depth;
    uint8_t tt_flag;
    
    if (tt.probe(hash_key, tt_score, tt_depth, tt_flag, ply)) {
      if (tt_depth >= depth) {
        if (tt_flag == 0) return tt_score;
        if (tt_flag == 1 && tt_score >= beta) return tt_score;
        if (tt_flag == 2 && tt_score <= alpha) return tt_score;
      }
    }
  }
  
  // Generate moves
  Movelist movelist;
  movegen::legalmoves(movelist, board);
  
  if (movelist.empty()) {
    return board.inCheck() ? -MATE_SCORE + ply : 0;
  }

  if (depth <= 0) {
    return evaluate(board);
  }

  // Null move pruning - AGGRESSIVE
  if (allow_null && depth >= 2 && !board.inCheck() && ply > 0) {
    int R = (depth >= 6) ? 3 : 2;
    board.makeNullMove();
    int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, tld, false);
    board.unmakeNullMove();
    
    if (null_score >= beta) return beta;
  }
  
  // Convert to vector for ordering
  std::vector<Move> moves;
  moves.reserve(movelist.size());
  for (const auto& m : movelist) moves.push_back(m);
  
  order_moves(moves, board, depth, tld);

  int best_value = -INFINITY_SCORE;
  int orig_alpha = alpha;
  Move best_move = Move::NO_MOVE;
  int move_count = 0;

  for (auto &move : moves) {
    bool is_capture = board.isCapture(move);
    bool is_promo = (move.typeOf() == Move::PROMOTION);
    bool gives_check = false;
    
    board.makeMove(move);
    gives_check = board.inCheck();
    move_count++;

    int score;
    int reduction = 0;

    // PVS + LMR
    if (move_count == 1) {
      // PV node - full window
      score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, tld);
    } else {
      // Late move reductions
      if (move_count >= 4 && depth >= 3 && !is_capture && !is_promo && 
          !gives_check && !board.inCheck()) {
        reduction = 1;
        if (move_count >= 8) reduction = 2;
        if (depth >= 6 && move_count >= 12) reduction = 3;
      }
      
      // Null window search
      int new_depth = std::max(0, depth - 1 - reduction);
      score = -negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, tld);
      
      // Re-search if necessary
      if (score > alpha) {
        if (reduction > 0) {
          score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, tld);
        }
        if (score > alpha && score < beta) {
          score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, tld);
        }
      }
    }

    board.unmakeMove(move);

    if (score > best_value) {
      best_value = score;
      best_move = move;
    }

    alpha = std::max(alpha, best_value);

    if (alpha >= beta) {
      if (!is_capture && tld) {
        tld->killer_moves.addKiller(depth, move);
      }
      break;
    }
  }
 
  // TT store
  if (depth >= 1) {
    if (hash_key == 0) hash_key = board.hash();
    uint8_t flag = (best_value <= orig_alpha) ? 2 : (best_value >= beta) ? 1 : 0;
    tt.store(hash_key, depth, best_value, flag, ply);
  }
  
  return best_value;
}

// Root search - MASSIVELY parallel
Move find_best_move(Board& board, int max_depth) {
    Move best_move = Move::NO_MOVE;
    int best_score = -INFINITY_SCORE;
    
    // Iterative deepening
    for (int depth = 1; depth <= max_depth; depth++) {
        Movelist movelist;
        movegen::legalmoves(movelist, board);
        
        std::vector<Move> moves;
        moves.reserve(movelist.size());
        for (const auto& m : movelist) moves.push_back(m);
        
        if (moves.empty()) return Move::NO_MOVE;
        
        // Order with previous best
        order_moves(moves, board, depth, &thread_data[0], best_move);
        
        // Parallel search with shared alpha
        std::atomic<int> shared_alpha(best_score);
        std::vector<int> scores(moves.size(), -INFINITY_SCORE);
        Move depth_best = best_move;
        std::mutex best_mutex;
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < moves.size(); i++) {
            int tid = omp_get_thread_num();
            if (tid >= (int)thread_data.size()) tid = 0;
            ThreadLocalData* tld = &thread_data[tid];
            
            Board my_board = board;
            my_board.makeMove(moves[i]);
            
            int alpha = shared_alpha.load(std::memory_order_relaxed);
            int score = -negamax(my_board, depth - 1, -INFINITY_SCORE, -alpha, 1, tld);
            
            my_board.unmakeMove(moves[i]);
            scores[i] = score;
            
            // Update shared alpha
            int current_alpha = shared_alpha.load();
            while (score > current_alpha && 
                   !shared_alpha.compare_exchange_weak(current_alpha, score)) {
                current_alpha = shared_alpha.load();
            }
        }
        
        // Find best
        best_score = -INFINITY_SCORE;
        for (size_t i = 0; i < moves.size(); i++) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_move = moves[i];
            }
        }
        
        std::cout << "depth " << depth << " score " << best_score 
                  << " move " << chess::uci::moveToUci(best_move) 
                  << " nodes " << thread_data[0].nodes_searched << std::endl;
        
        if (abs(best_score) > MATE_SCORE - 100) break;
    }
    
    return best_move;
}

std::string run_engine(Board& board, int depth = 20) {
  attacks::initAttacks();
  
  static bool initialized = false;
  if (!initialized) {
      int num_threads = omp_get_max_threads();
      if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
      thread_data.resize(num_threads);
      for (int i = 0; i < num_threads; i++) {
          thread_data[i] = ThreadLocalData(i);
      }
      omp_set_num_threads(num_threads);
      initialized = true;
  }
  
  for (auto& td : thread_data) td.clear();
  tt.new_search();
  
  std::cout << "Initial Board:\n" << board << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  Move best_move = find_best_move(board, depth);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  uint64_t total_nodes = 0;
  for (const auto& td : thread_data) total_nodes += td.nodes_searched;
  
  uint64_t nps = (duration.count() > 0) ? (total_nodes * 1000 / duration.count()) : 0;

  std::cout << "\nBest Move: " << chess::uci::moveToUci(best_move) << std::endl;
  std::cout << "Time: " << duration.count() << " ms" << std::endl;
  std::cout << "Nodes: " << total_nodes << std::endl;
  std::cout << "NPS: " << nps << std::endl;
  std::cout << "Threads: " << thread_data.size() << std::endl;
  
  return chess::uci::moveToUci(best_move);
}
