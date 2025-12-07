#include "include/chess.hpp"
#include "killer_move.h"
#include "transposition_table.h"
#include "constants.h"
#include "history.h"
#include <array>
#include <algorithm>
#include <iostream>
#include <vector>    
#include <chrono>    

using namespace chess;
using namespace chess_engine;

KillerMoves killer_moves;
TranspositionTable tt;
HistoryTable history;

int evaluate(const Board &board) {
  int evaluation = 0;

  const std::array<PieceType, 6> piece_types = {
      PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
      PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
 
  for (const auto &pt : piece_types) {
    int value = piece_values[static_cast<int>(pt)];
    Bitboard white_bb = board.pieces(pt, Color::WHITE);
    Bitboard black_bb = board.pieces(pt, Color::BLACK);

    evaluation += white_bb.count() * value;
    evaluation -= black_bb.count() * value;

    const auto &table = tableForPiece(pt);
    evaluation += pstScore(white_bb, Color::WHITE, table);
    evaluation -= pstScore(black_bb, Color::BLACK, table);
  }

  evaluation += (board.sideToMove() == Color::WHITE) ? 10 : -10;

  return evaluation;
}

Square findLeastValuableAttacker(Bitboard attackers, Color color, Board &board) {
  // Check in order: pawn, knight, bishop, rook, queen, king
  const std::array<PieceType, 6> piece_order = {
    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
    PieceType::ROOK, PieceType::QUEEN, PieceType::KING
  };
  
  for (PieceType pt : piece_order) {
    Bitboard pieces = attackers & board.pieces(pt, color);
    if (pieces != 0) {
      return pieces.lsb(); 
    }
  }
  return Square::NO_SQ;
}

// Static Exchange Evaluation - estimation of the net material outcome of a capture sequence
int SEE(Move move, Board &board) {
  Square from_sq = move.from();
  Square to_sq = move.to();

  PieceType victim_type;
  
  Bitboard occupied = board.occ();
  
  if (move.typeOf() == Move::ENPASSANT) {
    victim_type = PieceType::PAWN;
    Square ep_square = board.enpassantSq();
    // XOR to remove the occupied squares
    occupied = occupied ^ Bitboard::fromSquare(from_sq) ^ Bitboard::fromSquare(ep_square);
  } else {
    victim_type = board.at<PieceType>(to_sq);
    occupied = occupied ^ Bitboard::fromSquare(from_sq) ^ Bitboard::fromSquare(to_sq);
  }

  // Capture sequence array
  int gain[32] = {};
  gain[0] = piece_values[static_cast<int>(victim_type)];

  // Find all pieces that attack the curreny to_sq and remove the attacker 
  Bitboard attackers_white = attacks::attackers(board, Color::WHITE, to_sq);
  Bitboard attackers_black = attacks::attackers(board, Color::BLACK, to_sq);
  
  Color attacker_color = board.at(from_sq).color();
  if (attacker_color == Color::WHITE) {
    attackers_white = attackers_white ^ Bitboard::fromSquare(from_sq);
  } else {
    attackers_black = attackers_black ^ Bitboard::fromSquare(from_sq);
  }

  Color side = ~board.sideToMove();
  
  int depth = 1;  

  while (depth < MAX_SEE_DEPTH) {
    Bitboard attackers = (side == Color::WHITE) ? attackers_white : attackers_black;
    
    if (attackers == 0) {
      break;
    }

    Square attacker_sq = findLeastValuableAttacker(attackers, side, board);
    
    if (attacker_sq == Square::NO_SQ) {
      break;
    }

    PieceType attacker_pt = board.at<PieceType>(attacker_sq);
    int attacker_val = piece_values[static_cast<int>(attacker_pt)];
    
    // Ex. rook takes, pawn takes => = -100 + 500 = 400
    gain[depth] = -attacker_val + gain[depth - 1];
    
    occupied = occupied ^ Bitboard::fromSquare(attacker_sq);
    
    if (side == Color::WHITE) {
      attackers_white = attackers_white ^ Bitboard::fromSquare(attacker_sq);
    } else {
      attackers_black = attackers_black ^ Bitboard::fromSquare(attacker_sq);
    }
    
    // Manually recalculating all attackers (some may be free to attack after the first capture)
    Bitboard white_pawns = board.pieces(PieceType::PAWN, Color::WHITE) & occupied;
    Bitboard black_pawns = board.pieces(PieceType::PAWN, Color::BLACK) & occupied;
    Bitboard white_knights = board.pieces(PieceType::KNIGHT, Color::WHITE) & occupied;
    Bitboard black_knights = board.pieces(PieceType::KNIGHT, Color::BLACK) & occupied;
    Bitboard white_bishops = board.pieces(PieceType::BISHOP, Color::WHITE) & occupied;
    Bitboard black_bishops = board.pieces(PieceType::BISHOP, Color::BLACK) & occupied;
    Bitboard white_rooks = board.pieces(PieceType::ROOK, Color::WHITE) & occupied;
    Bitboard black_rooks = board.pieces(PieceType::ROOK, Color::BLACK) & occupied;
    Bitboard white_queens = board.pieces(PieceType::QUEEN, Color::WHITE) & occupied;
    Bitboard black_queens = board.pieces(PieceType::QUEEN, Color::BLACK) & occupied;
    Bitboard white_kings = board.pieces(PieceType::KING, Color::WHITE) & occupied;
    Bitboard black_kings = board.pieces(PieceType::KING, Color::BLACK) & occupied;
    
    attackers_white = (attacks::pawn(Color::BLACK, to_sq) & white_pawns) |
                      (attacks::knight(to_sq) & white_knights) |
                      (attacks::bishop(to_sq, occupied) & (white_bishops | white_queens)) |
                      (attacks::rook(to_sq, occupied) & (white_rooks | white_queens)) |
                      (attacks::king(to_sq) & white_kings);
    
    attackers_black = (attacks::pawn(Color::WHITE, to_sq) & black_pawns) |
                      (attacks::knight(to_sq) & black_knights) |
                      (attacks::bishop(to_sq, occupied) & (black_bishops | black_queens)) |
                      (attacks::rook(to_sq, occupied) & (black_rooks | black_queens)) |
                      (attacks::king(to_sq) & black_kings);
    
    side = ~side;
    depth++;
  }

  // Calculate the final capture sequence value by backtracking through the capture sequence
  for (int i = depth - 1; i > 0; i--) {
    gain[i - 1] = std::max(-gain[i], gain[i - 1]);
  }

  return gain[0];  
}

int quiescence(Board &board, int alpha, int beta, int current_depth_from_root) {
  // Safety precaution for exploring too deep
  if (current_depth_from_root >= MAX_QUIESCENCE_DEPTH) {
    return evaluate(board);
  }

  Movelist all_moves;
  movegen::legalmoves(all_moves, board);
  
  if (all_moves.empty()) {
    if (board.inCheck()) {
      return -MATE_SCORE + current_depth_from_root;
    }
    return 0;  
  }
  
  // Stand pat - if the current capture path leads to a worse position, bail out
  int stand_pat = evaluate(board);
  if (stand_pat >= beta) return beta;
  if (stand_pat > alpha) alpha = stand_pat;
  
  Movelist captures;
  movegen::legalmoves<movegen::MoveGenType::CAPTURE>(captures, board);
  
  if (captures.empty()) {
    return stand_pat;
  }

  // Sort by Static Exchange Evaluation
  std::stable_sort(captures.begin(), captures.end(), [&board](const Move &move1, const Move &move2) {
    return SEE(move1, board) > SEE(move2, board);
  });

  for (const auto &move : captures) {
    int see_score = SEE(move, board);
    // Delta margin for capturing positional factors not caught by the SEE
    if (stand_pat + see_score + DELTA_MARGIN < alpha) {
      continue;
    }
    
    board.makeMove(move);
    int score = -quiescence(board, -beta, -alpha, current_depth_from_root + 1);
    board.unmakeMove(move);
    
    if (score >= beta) {
      return beta; 
    }
    
    if (score > alpha) {
      alpha = score; 
    }
  }

  return alpha;
}

void order_moves(std::vector<Move> &moves, Board &board, int depth, Move tt_move = Move::NO_MOVE, int static_eval = INFINITY_SCORE) {
    Color side_to_move = board.sideToMove();
    
    auto score_of = [&](const Move &m) -> int {
        int s = 0;
        
        // Order of priority: TranspositionTable (already seen the position, not expensive) move, Captures (ordered by SEE), Promotions, Killer moves, Quiet moves (use the history heuristic)

        if (m == tt_move && tt_move != Move::NO_MOVE) {
            s += 10000;
        }
        else if (board.isCapture(m)) {
          int see_score = SEE(m, board);
          s += 2000 + see_score;
        }
        else if (m.typeOf() == Move::PROMOTION) {
          s += 1500;
        }
        else if (killer_moves.isKiller(depth, m)) {
          s += 1000;
        }
        else {
          int hist_score = history.get(m, side_to_move);
          s += hist_score / 32;  
        }
        
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

int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root, int extension_count = 0, int prev_static_eval = INFINITY_SCORE) {
  // Safety check for infinite recursion
  if (current_depth_from_root > MAX_SEARCH_DEPTH) {
    return evaluate(board);
  }
  
  uint64_t zobrist_key = 0;
  Move tt_move = Move::NO_MOVE;
  
  if (depth >= 1) {
    zobrist_key = board.hash();
    TTEntry* tt_entry = tt.probe(zobrist_key);

    // The goal here is to give the best (previous) move, cached in the transposition table, the highest priority

    if (tt_entry->key == zobrist_key) {
      if (tt_entry->bestMove != Move::NO_MOVE) {
        tt_move = tt_entry->bestMove;
      }

      if (tt_entry->depth >= depth) {
        int tt_score = tt_entry->score;       
        
        if (tt_entry->flag == 0) return tt_score;
        if (tt_entry->flag == 1 && tt_score >= beta) return tt_score;
        if (tt_entry->flag == 2 && tt_score <= alpha) return tt_score;
      }
    }
  }
  
  Movelist movelist;
  movegen::legalmoves(movelist, board);
  
  // As the move stored in the TT is the last position's best move, there must be a check to see if it is legal in this new position
  if (tt_move != Move::NO_MOVE) {
    if (std::find(movelist.begin(), movelist.end(), tt_move) == movelist.end()) {
      tt_move = Move::NO_MOVE;
    }
  }
  
  if (movelist.empty()) {
    if (board.inCheck()) {
      return -MATE_SCORE + current_depth_from_root; 
    }
    return 0;
  }

  if (depth == 0) {
    return quiescence(board, alpha, beta, current_depth_from_root);
  }

  bool in_check = board.inCheck();
  int material_count = board.occ().count();
  bool in_endgame = (material_count <= 6);
  
  // Try to get static eval from TT first, otherwise calculate
  int static_eval = INFINITY_SCORE;
  bool tt_hit_for_eval = false;
  bool improving = false;
  
  if (zobrist_key != 0 && !in_check) {
    TTEntry* tt_entry = tt.probe(zobrist_key);
    if (tt_entry->key == zobrist_key) {
      tt_hit_for_eval = true;

      if (tt_entry->staticEval != 30001) {  
        static_eval = tt_entry->staticEval;
      }
    }
  }
  
  if (!in_check) {
    if (static_eval == INFINITY_SCORE) {
      static_eval = evaluate(board);
    }
    
    // Determine if position is improving, when prev_static_eval is not set, assume not improving
    improving = (prev_static_eval != INFINITY_SCORE) && (static_eval > prev_static_eval);
    
    // Razoring - if static evaluation is really low and a quick tactical sequence won't save it, prune it
    // Formula: eval < alpha - 514 (about 5 pawns) - 294 (tuned value from testing - Stockfish) * depth * depth (quadratic growth, punishes deeper moves heavily)
    if (depth <= 3 && static_eval < alpha - RAZOR_MARGIN_BASE - RAZOR_MARGIN_DEPTH * depth * depth) {
      return quiescence(board, alpha, beta, current_depth_from_root);
    }
  } else {
    static_eval = -INFINITY_SCORE;  
  }

  bool allow_null_move =
      depth >= 3 &&
      !in_check &&
      !in_endgame &&
      beta < MATE_SCORE - 1000 &&
      board.hasNonPawnMaterial(board.sideToMove()) &&
      static_eval != -INFINITY_SCORE &&
      (static_eval - 100) >= beta; // - 100 (centipawns) for passing a move

  if (allow_null_move) {
    const int R = (depth >= 6) ? 2 : 1;
    const int null_depth = depth - 1 - R;
    
    if (null_depth >= 2) {
      board.makeNullMove();
      int null_score = -negamax(board, null_depth, -beta, -beta + 1, current_depth_from_root + 1, extension_count, -static_eval);
      board.unmakeNullMove();
      
      if (null_score >= beta) {
        return null_score;
      }
    }
  }
  
  std::vector<Move> moves_to_search;
  moves_to_search.reserve(movelist.size());
  for (const auto& m : movelist) {
      moves_to_search.push_back(m);
  }
  
  order_moves(moves_to_search, board, depth, tt_move, static_eval);

  int bestValue = -INFINITY_SCORE;
  int original_alpha = alpha;
  Move bestMove = Move::NO_MOVE;
  bool bestMove_is_quiet = false;  
  
  bool first_move = true;
  Color side_to_move = board.sideToMove();
  
  // Track quiet moves searched for history updates
  std::vector<Move> quiets_searched;
  quiets_searched.reserve(moves_to_search.size());
  
  int move_count = 0;
  
  for (auto &move : moves_to_search) {
    bool is_capture = board.isCapture(move);
    bool is_promotion = (move.typeOf() == Move::PROMOTION);
    bool is_killer = killer_moves.isKiller(depth, move);
    
    bool gives_check = false;
    if (depth >= CHECK_EXTENSION_DEPTH && extension_count < MAX_CHECK_EXTENSIONS) {
      gives_check = (board.givesCheck(move) != CheckType::NO_CHECK);
    }

    // Futility pruning - skip quiet moves that are unlikely to raise alpha
    if (!in_check && static_eval != -INFINITY_SCORE && !is_capture && !is_promotion && !gives_check) {
      // Futility margin base * depth => 91 * depth, linear unlike razoring as the position is not so bad, but won't improve alpha
      int futility_margin = FUTILITY_MARGIN_BASE * depth;
      if (tt_hit_for_eval) {
        // Reduce margin if we have TT hit - position known, less risky to prune
        futility_margin -= FUTILITY_MARGIN_DEPTH_MULT;  
      }
      // Increase margin if position is improving
      if (improving) {
        futility_margin = futility_margin * 5 / 4; 
      }
      
      if (depth <= 6 && static_eval + futility_margin <= alpha) {
        continue;
      }
    }

    bool is_quiet = !is_capture && move.typeOf() != Move::PROMOTION && !gives_check;
    if (is_quiet) {
      quiets_searched.push_back(move);
    }
    
    board.makeMove(move);

    // LMR - late move reduction, as the moves are sorted, we reduce the search depth for
    // later moves as they are less likely to be good 
    int search_depth;
    
    // Check extension - extend depth by 1 if move gives check
    bool should_extend = gives_check && 
                         depth >= CHECK_EXTENSION_DEPTH && 
                         extension_count < MAX_CHECK_EXTENSIONS;
    int new_extension_count = should_extend ? extension_count + 1 : extension_count;

    bool full_depth_search = (move_count < LMR_FULL_DEPTH_MOVES) || (depth < LMR_MIN_DEPTH) || is_capture || is_promotion || is_killer || in_check;

    if (full_depth_search) {
      search_depth = should_extend ? depth : depth - 1;
    } else {
      // Used to reduce the search depth by using the no. of moves larger than the const.
     int moves_beyond_full = move_count - LMR_FULL_DEPTH_MOVES;
     int reduction = 0;
     if (depth >= 5) {
      reduction = 1 + (moves_beyond_full / 6);
      reduction = std::min(reduction, 2);
     } else if (depth >= 4) {
      reduction = moves_beyond_full / 8;
      reduction = std::min(reduction, 1);
     }
     reduction = std::min(reduction, depth - 1);
     search_depth = depth - 1 - reduction;

     if (should_extend) {
       search_depth = depth - reduction;
     }
     
     if (search_depth < 0) search_depth = 0;
    }

    int score;

    // Compute static eval only for first move to save time
    // For other moves, pass INFINITY_SCORE (sentinel) to skip improving flag calculation
    int new_static_eval = INFINITY_SCORE;
    if (first_move && !board.inCheck()) {
      new_static_eval = evaluate(board);
    }

    if (first_move) {
      score = -negamax(board, search_depth, -beta, -alpha, current_depth_from_root + 1, new_extension_count, -new_static_eval);
      first_move = false;
    } else {
      score = -negamax(board, search_depth, -alpha - 1, - alpha, current_depth_from_root + 1, new_extension_count, INFINITY_SCORE);
      if(score > alpha && score < beta) {
        // Re-search with full window, using the same extended depth if applicable
        int re_search_depth = should_extend ? depth : depth - 1;
        score = -negamax(board, re_search_depth, -beta, -alpha, current_depth_from_root + 1, new_extension_count, INFINITY_SCORE);
      }
    }

    board.unmakeMove(move);

    if (score > bestValue) {
      bestValue = score;
      bestMove = move;
      bestMove_is_quiet = is_quiet;  // Track if best move is quiet
    }

    alpha = std::max(alpha, bestValue);

    if(alpha >= beta) {
      if (!is_capture && move.typeOf() != Move::PROMOTION) {
        killer_moves.addKiller(depth, move);
      }
      
      // Store the cutoff move for history update at end of search
      bestMove = move;
      bestValue = score;
      bestMove_is_quiet = is_quiet;
      
      break;
    }

    move_count++;
  }
  
  // Update history at the end of search - taken from Stockfish
  // Only update if we found a best (quiet) move that improved alpha or caused cutoff
  if (bestMove != Move::NO_MOVE && bestMove_is_quiet && 
      (bestValue >= beta || bestValue > original_alpha)) {
    // Stockfish formula: min(121 * depth - 77, 1633) + bonus if TT move
    int bonus = std::min(121 * depth - 77, 1633);
    if (bestMove == tt_move && tt_move != Move::NO_MOVE) {
      bonus += 375; 
    }
    history.update(bestMove, side_to_move, bonus);
    
    // Penalize other quiet moves that were searched (only if we got a cutoff)
    // Limit to first 32 moves to avoid expensive loops
    if (bestValue >= beta && !quiets_searched.empty()) {
      int malus = std::min(825 * depth - 196, 2159) - 16 * move_count;
      int max_penalize = std::min(static_cast<int>(quiets_searched.size()), 32);
      for (int i = 0; i < max_penalize; i++) {
        const Move& quiet_move = quiets_searched[i];
        if (quiet_move != bestMove) {
          history.update(quiet_move, side_to_move, -malus);
        }
      }
    }
  }
 
  if (depth >= 1) {
    if (zobrist_key == 0) zobrist_key = board.hash();
    uint8_t flag = (bestValue <= original_alpha) ? 2 : (bestValue >= beta) ? 1 : 0;
    Move move_to_store = (bestMove != Move::NO_MOVE) ? bestMove : tt_move;
    int eval_to_store = (static_eval != INFINITY_SCORE && static_eval != -INFINITY_SCORE) ? static_eval : 30001;
    int score_to_store = bestValue;
    if (bestValue > MATE_SCORE - 1000) score_to_store += current_depth_from_root;
    else if (bestValue < -MATE_SCORE + 1000) score_to_store -= current_depth_from_root;
    tt.store(zobrist_key, depth, score_to_store, flag, move_to_store, eval_to_store);
  }
  
  return bestValue;
}

Move find_best_move(Board& board, int max_depth, int time_limit_ms = 0) {
    // For limiting purposes, hard cut off
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
   
    history.clear();
    
    Movelist movelist;
    movegen::legalmoves(movelist, board);
    
    if (movelist.empty()) {
        return Move::NO_MOVE;
    }
    
    std::vector<Move> all_moves;
    all_moves.reserve(movelist.size());
    for (const auto& m : movelist) {
        all_moves.push_back(m);
    }
    
    Move best_move = all_moves[0]; 
    int best_score = -INFINITY_SCORE;
    
    // Iterative deepening loop
    for (int depth = 1; depth <= max_depth; depth++) {

        // For limiting purposes, hard cut off
        if (time_limit_ms > 0) {
            auto current_time = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
            if (elapsed >= time_limit_ms) {
                std::cout << "Time limit reached at depth " << (depth - 1) << std::endl;
                break;
            }
        }
        
        movelist.clear();
        movegen::legalmoves(movelist, board);
        
        all_moves.clear();
        all_moves.reserve(movelist.size());
        for (const auto& m : movelist) {
            all_moves.push_back(m);
        }
        
        // Get TT move from root position for move ordering
        Move root_tt_move = Move::NO_MOVE;
        uint64_t root_key = board.hash();
        TTEntry* root_tt_entry = tt.probe(root_key);
        if (root_tt_entry->key == root_key && root_tt_entry->bestMove != Move::NO_MOVE) {
            if (std::find(movelist.begin(), movelist.end(), root_tt_entry->bestMove) != movelist.end()) {
                root_tt_move = root_tt_entry->bestMove;
            }
        }
        
        // Move previous best move to front 
        if (depth > 1 && best_move != Move::NO_MOVE) {
            auto it = std::find(all_moves.begin(), all_moves.end(), best_move);
            if (it != all_moves.end() && it != all_moves.begin()) {
                std::swap(all_moves[0], *it);
            }
        }
        
        // Get static evalution for move ordering
        int root_static_eval = INFINITY_SCORE;
        if (!board.inCheck()) {
          root_static_eval = evaluate(board);
        }
        
        order_moves(all_moves, board, depth, root_tt_move, root_static_eval);
        
        int alpha = -INFINITY_SCORE;
        int beta = INFINITY_SCORE;
        Move current_best_move = Move::NO_MOVE;
        int current_best_score = -INFINITY_SCORE;
        
        for (auto &move : all_moves) {
            if (time_limit_ms > 0) {
                auto current_time = high_resolution_clock::now();
                auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
                if (elapsed >= time_limit_ms) {
                    break;  
                }
            }
            
            board.makeMove(move);
            int score = -negamax(board, depth - 1, -beta, -alpha, 1, 0, INFINITY_SCORE);
            board.unmakeMove(move);
            
            if (score > current_best_score) {
                current_best_score = score;
                current_best_move = move;
            }
            
            alpha = std::max(alpha, current_best_score);
            
            // Beta cutoff
            if (alpha >= beta) {
                break;
            }
            
            // If we found a mate, we can stop early
            if (current_best_score >= MATE_SCORE - 1000) {
                break;
            }
        }
        
        if (current_best_move != Move::NO_MOVE) {
            best_move = current_best_move;
            best_score = current_best_score;
            
            std::cout << "Depth " << depth << ": " << chess::uci::moveToUci(best_move) 
                      << " (score: " << best_score << ")";
            
            if (time_limit_ms > 0) {
                auto current_time = high_resolution_clock::now();
                auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
                std::cout << " [time: " << elapsed << "ms]";
            }
            std::cout << std::endl;
            
            if (best_score >= MATE_SCORE - 1000 || best_score <= -MATE_SCORE + 1000) {
                std::cout << "Mate found, stopping search." << std::endl;
                break;
            }
        }
    }
    
    std::cout << "Final Score: " << best_score << std::endl;
    
    return best_move;
}

#ifndef UNIT_TESTS
int main() {
  attacks::initAttacks();

  Board board2 = 
      Board("2k2bnr/p1r2pp1/1pQp2q1/7p/4PPn1/2N1B3/PPP1BP1P/2KR3R w - - 1 16");

  Board board3 = Board("rnr5/p4p1k/bp1qp2p/3pP3/Pb1N1Q2/1P3NPB/5P2/R3R1K1 w - - 5 23");

  std::cout << "Initial Board:\n";
  std::cout << board2 << std::endl;
  
  auto start_time = std::chrono::high_resolution_clock::now();

  Move best_move = find_best_move(board3, 14, 0); 

  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "\nBest Move found: " << chess::uci::moveToUci(best_move) << std::endl;
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

}
#endif