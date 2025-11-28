#include "include/chess.hpp"
#include "constants.h"
#include "history.h"
#include "tt_parallel.h"
#include "thread_local_data.h"
#include <array>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <atomic>
#include <mutex>

// Platform specific popcount (GCC/Clang/MSVC)
#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT(x) __popcnt64(x)
#else
#define POPCOUNT(x) __builtin_popcountll(x)
#endif

using namespace chess;
using namespace chess_engine;

// --- DATA STRUCTURES ---
struct ScoredMove {
    Move move;
    int score;
    bool operator>(const ScoredMove& other) const {
        return score > other.score;
    }
};

// --- GLOBAL SHARED DATA ---
TTParallel tt(512); 
HistoryTable history; 

// --- THREAD LOCAL STORAGE ---
std::vector<ThreadLocalData> tld_store;

// --- EVALUATION ---
int evaluate(const Board &board) {
    int evaluation = 0;
    
    Bitboard wp = board.pieces(PieceType::PAWN, Color::WHITE);
    Bitboard bp = board.pieces(PieceType::PAWN, Color::BLACK);
    Bitboard wn = board.pieces(PieceType::KNIGHT, Color::WHITE);
    Bitboard bn = board.pieces(PieceType::KNIGHT, Color::BLACK);
    Bitboard wb = board.pieces(PieceType::BISHOP, Color::WHITE);
    Bitboard bb = board.pieces(PieceType::BISHOP, Color::BLACK);
    Bitboard wr = board.pieces(PieceType::ROOK, Color::WHITE);
    Bitboard br = board.pieces(PieceType::ROOK, Color::BLACK);
    Bitboard wq = board.pieces(PieceType::QUEEN, Color::WHITE);
    Bitboard bq = board.pieces(PieceType::QUEEN, Color::BLACK);
    Bitboard wk = board.pieces(PieceType::KING, Color::WHITE);
    Bitboard bk = board.pieces(PieceType::KING, Color::BLACK);

    int mat_score = 0;
    mat_score += (wp.count() - bp.count()) * piece_values[0];
    mat_score += (wn.count() - bn.count()) * piece_values[1];
    mat_score += (wb.count() - bb.count()) * piece_values[2];
    mat_score += (wr.count() - br.count()) * piece_values[3];
    mat_score += (wq.count() - bq.count()) * piece_values[4];
    
    evaluation += mat_score;

    evaluation += pstScore(wp, Color::WHITE, pawn_table)   - pstScore(bp, Color::BLACK, pawn_table);
    evaluation += pstScore(wn, Color::WHITE, knight_table) - pstScore(bn, Color::BLACK, knight_table);
    evaluation += pstScore(wb, Color::WHITE, bishop_table) - pstScore(bb, Color::BLACK, bishop_table);
    evaluation += pstScore(wr, Color::WHITE, rook_table)   - pstScore(br, Color::BLACK, rook_table);
    evaluation += pstScore(wq, Color::WHITE, queen_table)  - pstScore(bq, Color::BLACK, queen_table);
    evaluation += pstScore(wk, Color::WHITE, king_table)   - pstScore(bk, Color::BLACK, king_table);

    evaluation += (board.sideToMove() == Color::WHITE) ? 10 : -10;

    return evaluation;
}

inline Square findLeastValuableAttacker(Bitboard attackers, Color color, const Board &board) {
    if ((attackers & board.pieces(PieceType::PAWN, color))) 
        return (attackers & board.pieces(PieceType::PAWN, color)).lsb();
    if ((attackers & board.pieces(PieceType::KNIGHT, color))) 
        return (attackers & board.pieces(PieceType::KNIGHT, color)).lsb();
    if ((attackers & board.pieces(PieceType::BISHOP, color))) 
        return (attackers & board.pieces(PieceType::BISHOP, color)).lsb();
    if ((attackers & board.pieces(PieceType::ROOK, color))) 
        return (attackers & board.pieces(PieceType::ROOK, color)).lsb();
    if ((attackers & board.pieces(PieceType::QUEEN, color))) 
        return (attackers & board.pieces(PieceType::QUEEN, color)).lsb();
    if ((attackers & board.pieces(PieceType::KING, color))) 
        return (attackers & board.pieces(PieceType::KING, color)).lsb();
    return Square::NO_SQ;
}

int SEE(Move move, Board &board) {
    Square from_sq = move.from();
    Square to_sq = move.to();
    Bitboard occupied = board.occ();
    PieceType victim_type;

    if (move.typeOf() == Move::ENPASSANT) {
        victim_type = PieceType::PAWN;
        Square ep_square = board.enpassantSq();
        if (ep_square == Square::NO_SQ) return 0; 
        occupied ^= Bitboard::fromSquare(from_sq) ^ Bitboard::fromSquare(ep_square);
    } else {
        victim_type = board.at<PieceType>(to_sq);
        occupied ^= Bitboard::fromSquare(from_sq) ^ Bitboard::fromSquare(to_sq);
    }

    if (victim_type == PieceType::NONE) return 0;

    int gain[32] = {};
    gain[0] = piece_values[static_cast<int>(victim_type)];

    Bitboard attackers_white = attacks::attackers(board, Color::WHITE, to_sq);
    Bitboard attackers_black = attacks::attackers(board, Color::BLACK, to_sq);

    Color attacker_color = board.at(from_sq).color();
    if (attacker_color == Color::WHITE) attackers_white ^= Bitboard::fromSquare(from_sq);
    else attackers_black ^= Bitboard::fromSquare(from_sq);

    Color side = ~board.sideToMove();
    int depth = 1;

    while (depth < MAX_SEE_DEPTH) {
        Bitboard attackers = (side == Color::WHITE) ? attackers_white : attackers_black;
        if (attackers == 0) break;

        Square attacker_sq = findLeastValuableAttacker(attackers, side, board);
        if (attacker_sq == Square::NO_SQ) break;

        PieceType attacker_pt = board.at<PieceType>(attacker_sq);
        if (attacker_pt == PieceType::NONE) break;

        gain[depth] = -piece_values[static_cast<int>(attacker_pt)] + gain[depth - 1];
        occupied ^= Bitboard::fromSquare(attacker_sq);

        if (side == Color::WHITE) attackers_white ^= Bitboard::fromSquare(attacker_sq);
        else attackers_black ^= Bitboard::fromSquare(attacker_sq);

        Bitboard all_sliders = board.pieces(PieceType::BISHOP) | board.pieces(PieceType::QUEEN) | board.pieces(PieceType::ROOK);
        if (all_sliders & occupied) {
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
        }
        side = ~side;
        depth++;
    }

    for (int i = depth - 1; i > 0; i--) {
        gain[i - 1] = std::max(-gain[i], gain[i - 1]);
    }
    return gain[0];
}

int quiescence(Board &board, int alpha, int beta, int current_depth_from_root, ThreadLocalData* tld) {
    if (tld) tld->nodes_searched++;
    if (current_depth_from_root >= MAX_QUIESCENCE_DEPTH) return evaluate(board);

    int stand_pat = evaluate(board);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    Movelist captures;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(captures, board);
    if (captures.empty()) return stand_pat;

    std::vector<Move> caps;
    caps.reserve(captures.size());
    for(const auto& m : captures) caps.push_back(m);

    std::stable_sort(caps.begin(), caps.end(), [&board](const Move &move1, const Move &move2) {
        return SEE(move1, board) > SEE(move2, board);
    });

    for (const auto &move : caps) {
        int see_score = SEE(move, board);
        if (stand_pat + see_score + DELTA_MARGIN < alpha) continue;

        board.makeMove(move);
        int score = -quiescence(board, -beta, -alpha, current_depth_from_root + 1, tld);
        board.unmakeMove(move);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

void order_moves(std::vector<Move> &moves, Board &board, int depth, ThreadLocalData* tld, Move tt_move = Move::NO_MOVE) {
    Color side_to_move = board.sideToMove();
    std::vector<ScoredMove> scored_moves;
    scored_moves.reserve(moves.size());
    
    for (const auto& m : moves) {
        int s = 0;
        if (m == tt_move && tt_move != Move::NO_MOVE) s += 10000;
        else if (board.isCapture(m)) s += 2000 + SEE(m, board);
        else if (m.typeOf() == Move::PROMOTION) s += 1500;
        else if (tld && tld->killer_moves.isKiller(depth, m)) s += 1000; 
        else {
            int hist = history.get(m, side_to_move);
            s += hist / 32;
        }
        scored_moves.push_back({m, s});
    }

    std::sort(scored_moves.begin(), scored_moves.end(), [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });

    for (size_t i = 0; i < moves.size(); i++) {
        moves[i] = scored_moves[i].move;
    }
}

int negamax(Board &board, int depth, int alpha, int beta, int ply, 
            int extension_count, int prev_static_eval, ThreadLocalData* tld) {
    
    if (tld) tld->nodes_searched++;
    if (ply > MAX_SEARCH_DEPTH) return evaluate(board);

    uint64_t zobrist_key = 0;
    Move tt_move = Move::NO_MOVE;

    if (depth >= 1) {
        zobrist_key = board.hash();
        TTEntryParallel* entry = tt.probe(zobrist_key);
        if (entry->key == zobrist_key) {
            tt_move = entry->bestMove;
            if (entry->depth >= depth) {
                int tt_score = tt.retrieve_score(entry->score, ply);
                if (entry->flag == 0) return tt_score;
                if (entry->flag == 1 && tt_score >= beta) return tt_score;
                if (entry->flag == 2 && tt_score <= alpha) return tt_score;
            }
        }
    }

    Movelist movelist;
    movegen::legalmoves(movelist, board);

    if (tt_move != Move::NO_MOVE) {
        bool legal = false;
        for(const auto& m : movelist) if(m == tt_move) { legal = true; break; }
        if(!legal) tt_move = Move::NO_MOVE;
    }

    if (movelist.empty()) {
        if (board.inCheck()) return -MATE_SCORE + ply;
        return 0;
    }

    if (depth == 0) return quiescence(board, alpha, beta, ply, tld);

    bool in_check = board.inCheck();
    int material_count = board.occ().count();
    bool in_endgame = (material_count <= 6);

    int static_eval = INFINITY_SCORE;
    bool tt_hit_for_eval = false;
    bool improving = false;

    if (zobrist_key != 0 && !in_check) {
        TTEntryParallel* entry = tt.probe(zobrist_key);
        if (entry->key == zobrist_key) {
            tt_hit_for_eval = true;
            if (entry->staticEval != 30001) static_eval = entry->staticEval;
        }
    }

    if (!in_check) {
        if (static_eval == INFINITY_SCORE) static_eval = evaluate(board);
        improving = (prev_static_eval != INFINITY_SCORE) && (static_eval > prev_static_eval);

        if (depth <= 3 && static_eval < alpha - RAZOR_MARGIN_BASE - RAZOR_MARGIN_DEPTH * depth * depth) {
             return quiescence(board, alpha, beta, ply, tld);
        }
    } else {
        static_eval = -INFINITY_SCORE;
    }

    bool allow_null_move = depth >= 3 && !in_check && !in_endgame && beta < MATE_SCORE - 1000 &&
                           board.hasNonPawnMaterial(board.sideToMove()) && static_eval != -INFINITY_SCORE &&
                           (static_eval - 100) >= beta;

    if (allow_null_move) {
        const int R = (depth >= 6) ? 2 : 1;
        const int null_depth = depth - 1 - R;
        if (null_depth >= 2) {
            board.makeNullMove();
            int null_score = -negamax(board, null_depth, -beta, -beta + 1, ply + 1, extension_count, -static_eval, tld);
            board.unmakeNullMove();
            if (null_score >= beta) return null_score;
        }
    }

    std::vector<Move> moves_to_search;
    moves_to_search.reserve(movelist.size());
    for (const auto& m : movelist) moves_to_search.push_back(m);

    order_moves(moves_to_search, board, depth, tld, tt_move);

    int bestValue = -INFINITY_SCORE;
    int original_alpha = alpha;
    Move bestMove = Move::NO_MOVE;
    bool bestMove_is_quiet = false;
    
    bool first_move = true;
    Color side_to_move = board.sideToMove();
    std::vector<Move> quiets_searched;
    quiets_searched.reserve(moves_to_search.size());
    int move_count = 0;

    for (auto &move : moves_to_search) {
        bool is_capture = board.isCapture(move);
        bool is_promotion = (move.typeOf() == Move::PROMOTION);
        bool is_killer = tld->killer_moves.isKiller(depth, move);
        bool gives_check = false;

        if (depth >= CHECK_EXTENSION_DEPTH && extension_count < MAX_CHECK_EXTENSIONS) {
             gives_check = (board.givesCheck(move) != CheckType::NO_CHECK);
        }

        // --- CRITICAL FIX FOR 30000 SCORE BUG ---
        // We added (move_count > 0). 
        // This ensures we NEVER prune the first move.
        // If we prune ALL moves, bestValue remains -INFINITY, causing the bug.
        if (move_count > 0 && !in_check && static_eval != -INFINITY_SCORE && !is_capture && !is_promotion && !gives_check) {
            int futility_margin = FUTILITY_MARGIN_BASE * depth;
            if (tt_hit_for_eval) futility_margin -= FUTILITY_MARGIN_DEPTH_MULT;
            if (improving) futility_margin = futility_margin * 5 / 4;
            if (depth <= 6 && static_eval + futility_margin <= alpha) continue;
        }

        bool is_quiet = !is_capture && move.typeOf() != Move::PROMOTION && !gives_check;
        if (is_quiet) quiets_searched.push_back(move);

        board.makeMove(move);

        int search_depth;
        bool should_extend = gives_check && depth >= CHECK_EXTENSION_DEPTH && extension_count < MAX_CHECK_EXTENSIONS;
        int new_extension_count = should_extend ? extension_count + 1 : extension_count;
        bool full_depth_search = (move_count < LMR_FULL_DEPTH_MOVES) || (depth < LMR_MIN_DEPTH) || 
                                 is_capture || is_promotion || is_killer || in_check;

        if (full_depth_search) {
            search_depth = should_extend ? depth : depth - 1;
        } else {
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
            if (should_extend) search_depth = depth - reduction;
            if (search_depth < 0) search_depth = 0;
        }

        int score;
        int new_static_eval = INFINITY_SCORE;
        if (first_move && !board.inCheck()) new_static_eval = evaluate(board);

        if (first_move) {
            score = -negamax(board, search_depth, -beta, -alpha, ply + 1, new_extension_count, -new_static_eval, tld);
            first_move = false;
        } else {
            score = -negamax(board, search_depth, -alpha - 1, -alpha, ply + 1, new_extension_count, INFINITY_SCORE, tld);
            if (score > alpha && score < beta) {
                int re_search_depth = should_extend ? depth : depth - 1;
                score = -negamax(board, re_search_depth, -beta, -alpha, ply + 1, new_extension_count, INFINITY_SCORE, tld);
            }
        }

        board.unmakeMove(move);

        if (score > bestValue) {
            bestValue = score;
            bestMove = move;
            bestMove_is_quiet = is_quiet;
        }

        if (score > alpha) {
            alpha = score;
        }

        if (alpha >= beta) {
            if (!is_capture && move.typeOf() != Move::PROMOTION) {
                tld->killer_moves.addKiller(depth, move);
            }
            break; 
        }
        move_count++;
    }

    if (bestMove != Move::NO_MOVE && bestMove_is_quiet && (bestValue >= beta || bestValue > original_alpha)) {
        int bonus = std::min(121 * depth - 77, 1633);
        if (bestMove == tt_move && tt_move != Move::NO_MOVE) bonus += 375;
        
        history.update(bestMove, side_to_move, bonus);

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
        
        tt.store(zobrist_key, depth, bestValue, flag, move_to_store, eval_to_store, ply);
    }

    return bestValue;
}

Move find_best_move(Board& board, int max_depth, int time_limit_ms = 0) {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    
    static bool threads_init = false;
    if (!threads_init) {
        int max_threads = omp_get_max_threads();
        tld_store.resize(max_threads);
        for(int i=0; i<max_threads; ++i) tld_store[i] = ThreadLocalData(i);
        threads_init = true;
    }

    for(auto& t : tld_store) t.clear();
    tt.clear();
    tt.new_search();
    history.clear();

    Move best_move_overall = Move::NO_MOVE;
    int best_score_overall = -INFINITY_SCORE;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (time_limit_ms > 0) {
            auto current_time = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
            if (elapsed >= time_limit_ms) {
                std::cout << "Time limit reached." << std::endl;
                break;
            }
        }

        Movelist movelist;
        movegen::legalmoves(movelist, board);
        if (movelist.empty()) break;

        std::vector<Move> root_moves;
        for (const auto& m : movelist) root_moves.push_back(m);

        Move prev_best = (depth > 1) ? best_move_overall : Move::NO_MOVE;
        order_moves(root_moves, board, depth, &tld_store[0], prev_best);

        std::atomic<int> alpha(-INFINITY_SCORE);
        int beta = INFINITY_SCORE;
        
        std::vector<int> root_scores(root_moves.size(), -INFINITY_SCORE);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < (int)root_moves.size(); i++) {
            int tid = omp_get_thread_num();
            if (tid >= tld_store.size()) tid = 0; 
            ThreadLocalData* tld = &tld_store[tid];

            Board local_board = board;
            Move move = root_moves[i];
            
            local_board.makeMove(move);
            int local_alpha = alpha.load(std::memory_order_relaxed);
            int score = -negamax(local_board, depth - 1, -beta, -local_alpha, 1, 0, INFINITY_SCORE, tld);
            local_board.unmakeMove(move);

            root_scores[i] = score;

            int current_alpha = alpha.load(std::memory_order_relaxed);
            while (score > current_alpha) {
                if (alpha.compare_exchange_weak(current_alpha, score)) {
                    break;
                }
            }
        }

        int iteration_best_score = -INFINITY_SCORE;
        Move iteration_best_move = Move::NO_MOVE;

        for (int i = 0; i < (int)root_moves.size(); i++) {
            if (root_scores[i] > iteration_best_score) {
                iteration_best_score = root_scores[i];
                iteration_best_move = root_moves[i];
            }
        }

        best_score_overall = iteration_best_score;
        best_move_overall = iteration_best_move;

        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
        uint64_t total_nodes = 0;
        for(const auto& t : tld_store) total_nodes += t.nodes_searched;
        uint64_t nps = (elapsed > 0) ? (total_nodes * 1000 / elapsed) : 0;

        std::cout << "info depth " << depth 
                  << " score cp " << best_score_overall 
                  << " nodes " << total_nodes 
                  << " nps " << nps
                  << " time " << elapsed
                  << " pv " << chess::uci::moveToUci(best_move_overall) 
                  << std::endl;

        if (best_score_overall >= MATE_SCORE - 1000 || best_score_overall <= -MATE_SCORE + 1000) break;
    }

    return best_move_overall;
}

std::string run_engine(Board& board, int depth = 20) {
    attacks::initAttacks();
    
    static bool initialized = false;
    if (!initialized) {
        int num_threads = omp_get_max_threads();
        tld_store.resize(num_threads);
        for (int i = 0; i < num_threads; i++) {
            tld_store[i] = ThreadLocalData(i);
        }
        omp_set_num_threads(num_threads);
        initialized = true;
    }
    
    for (auto& td : tld_store) td.clear();
    tt.clear();
    tt.new_search(); 
    history.clear(); 
    
    std::cout << "Initial Board:\n" << board << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Move best_move = find_best_move(board, depth, 0); 
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    uint64_t total_nodes = 0;
    for (const auto& td : tld_store) total_nodes += td.nodes_searched;
    
    uint64_t nps = (duration.count() > 0) ? (total_nodes * 1000 / duration.count()) : 0;

    std::cout << "\nBest Move: " << chess::uci::moveToUci(best_move) << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Nodes: " << total_nodes << std::endl;
    std::cout << "NPS: " << nps << std::endl;
    std::cout << "Threads: " << tld_store.size() << std::endl;
    
    return chess::uci::moveToUci(best_move);
}

#ifndef UNIT_TESTS 
int main() {
    attacks::initAttacks();
    Board board("rnr5/p4p1k/bp1qp2p/3pP3/Pb1N1Q2/1P3NPB/5P2/R3R1K1 w - - 5 23");
    run_engine(board, 14);
    return 0;
}
#endif