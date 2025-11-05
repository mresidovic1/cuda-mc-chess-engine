#include "../include/chess.hpp"
#include <array>
#include <algorithm> // Za std::sort i std::max/min
#include <iostream>
#include <vector>    // Za vector<Move> za sortiranje poteza
#include <chrono> // Za merenje vremena


using namespace chess;

// --- Konstante ---
const int INFINITY_SCORE = 30000;
const int MATE_SCORE = 10000;

// --- Vrednosti figura (povećane za bolju rezoluciju) ---
constexpr std::array<int, 6> piece_values = {100, 300, 320, 500, 900, 0}; 

// --- Evaluaciona funkcija ---
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
  
  // Dodatni mali bonus za igrača čiji je red (Tempo Bonus)
  // Ovo je jednostavna poziciona heuristika.
  if (board.sideToMove() == chess::Color::WHITE) {
      evaluation += 10;
  } else {
      evaluation -= 10;
  }

  return evaluation;
}

// --- Pomoćna funkcija za sortiranje poteza (jednostavno Move Ordering) ---
// Sortira poteze tako da zahvati (captures) idu prvi.
void order_moves(std::vector<Move> &moves, const Board &board) {
    // Sortira listu poteza. Potezi zahvata će imati veću "vrednost" za sortiranje.
    std::sort(moves.begin(), moves.end(), [&board](const Move& a, const Move& b) {
        bool a_is_capture = board.isCapture(a);
        bool b_is_capture = board.isCapture(b);

        if (a_is_capture && !b_is_capture) return true;  // A je zahvat, B nije, A ide pre
        if (!a_is_capture && b_is_capture) return false; // B je zahvat, A nije, B ide pre

        // Ako su oba ili nijedan zahvati, nema preferencije u ovom jednostavnom sortiranju.
        // Složenija sortiranja bi ovde koristila MVV/LVA, history, killere itd.
        return false; 
    });
}


// --- Glavna Negamax funkcija sa Alfa-Beta odsecanjem ---
// Koristi Negamax, koji je ekvivalent Minimaxu, ali pojednostavljuje implementaciju.
// Parametar `current_depth_from_root` se koristi za bolju procenu mata.
int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root) {
  Movelist movelist;
  movegen::legalmoves(movelist, board);
  
  // Konvertujemo Movelist u std::vector<Move> za lakše sortiranje
  std::vector<Move> moves_to_search;
  for (const auto& m : movelist) {
      moves_to_search.push_back(m);
  }

  // Osnovni slučajevi:
  // 1. Nema legalnih poteza (mat ili pat)
  if (moves_to_search.empty()) {
    if (board.inCheck()) {
      // Mat: -MATE_SCORE za igrača na potezu (što je -INFINITY_SCORE za njega)
      // Dodajemo current_depth_from_root da bi preferirali brži mat
      return -MATE_SCORE + current_depth_from_root; 
    }
    return 0; // Pat (stalemate) je remi
  }

  // 2. Dostignuta dubina pretrage
  if (depth == 0) {
    return evaluate(board);
  }

  // --- Move Ordering ---
  order_moves(moves_to_search, board);

  int bestValue = -INFINITY_SCORE; // Inicijalizuj sa najgorom mogućom vrednošću
  
  // Prođi kroz sve moguće poteze
  for (auto &move : moves_to_search) {
    board.makeMove(move); // Napravi potez

    // Rekurzivni poziv za sledećeg igrača.
    // Negiramo rezultat i zamenjujemo alpha i beta jer je to Min igračev red.
    // current_depth_from_root se povećava za 1.
    int score = -negamax(board, depth - 1, -beta, -alpha, current_depth_from_root + 1);
    
    board.unmakeMove(move); // Poništi potez (vrati tablu u prethodno stanje)

    bestValue = std::max(bestValue, score); // Trenutni igrač želi da maksimizuje
    alpha = std::max(alpha, bestValue);     // Ažuriraj alfa

    if (alpha >= beta) { // Alfa-Beta odsecanje
      break;             // Prekini pretragu ovog podstabla
    }
  }
  return bestValue;
}

// --- Funkcija za pronalaženje najboljeg poteza ---
// Sada je ova funkcija jednostavnija jer nema Iterative Deepening
Move find_best_move(Board& board, int max_depth) {
    Movelist movelist;
    movegen::legalmoves(movelist, board);
    
    // Sortiraj početne poteze takođe
    std::vector<Move> initial_moves;
    for (const auto& m : movelist) {
        initial_moves.push_back(m);
    }
    order_moves(initial_moves, board);

    int best_score = -INFINITY_SCORE;
    Move best_move = Move::NO_MOVE;

    // Početni alfa i beta prozori
    int alpha = -INFINITY_SCORE;
    int beta = INFINITY_SCORE;

    for (auto &move : initial_moves) {
        board.makeMove(move);
        // Pozovi negamax za trenutni potez.
        // current_depth_from_root je 1 jer je ovo prvi polu-potez.
        int score = -negamax(board, max_depth - 1, -beta, -alpha, 1);
        board.unmakeMove(move);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        alpha = std::max(alpha, best_score); // Ažuriraj alpha na root nivou
    }
    
    std::cout << "Final Score: " << best_score << std::endl;

    return best_move;
}

int main() {
  attacks::initAttacks();
  
  Board board =
      Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  std::cout << "Initial Board:\n";
  std::cout << board << std::endl;
  
  // --- POČETAK MERENJA VREMENA ---
  auto start_time = std::chrono::high_resolution_clock::now();

  Move best_move = find_best_move(board, 9); // Pretražujemo do dubine 9

  // --- KRAJ MERENJA VREMENA ---
  auto end_time = std::chrono::high_resolution_clock::now();

  // Izračunavanje trajanja
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // Možeš koristiti i nanoseconds, microseconds, seconds, itd.
  // Za milisekunde je duration_cast<std::chrono::milliseconds>
  // Za sekunde je duration_cast<std::chrono::seconds>

  std::cout << "\nBest Move found: " << chess::uci::moveToUci(best_move) << std::endl;
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl; // Ispis trajanja u milisekundama

  return 0;
}