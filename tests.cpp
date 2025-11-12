#include "include/chess.hpp"
#include <cassert>
#include <iostream>
#include <string>

int total = 0;
int passed = 0;

// Declare the symbols defined in chess_parallelization.cpp
extern chess::Move find_best_move(chess::Board& board, int max_depth);
extern std::string run_engine(chess::Board& board, int depth = 8);

// Optional: if you need attacks::initAttacks() explicitly
using namespace chess;

static bool run_best_move_test(const char* name,
                               const char* fen,
                               const char* expected_uci,
                               int depth) {
  std::cout << "Running: " << name << "\n";
  chess::Board b(fen);

  std::string got = run_engine(b, 9); // Default depth is 8

  std::cout << "FEN: " << fen << "\nExpected: " << expected_uci
            << "\nGot: " << got << "\n\n";

  return got == std::string(expected_uci);
}

void run_test(std::string board, std::string expected_result){
  total++;
  std::string test_title_string = "Test " + std::to_string(total) + " - expected " + expected_result;
  if (run_best_move_test(
          test_title_string.c_str(),
          board.c_str(),
          expected_result.c_str(),
          8)) {
    std::cout << "[PASS] " << test_title_string << "\n";
    passed++;
  } else {
    std::cout << "[FAIL] " << test_title_string << "\n";
  }
}

int main() {
  // Test cases
  std::string board;
  // Test case 1
  board = "4r1k1/r4p1p/p3bRpQ/q2pP3/2pP4/Bpn1R3/6PP/1B4K1 w - - 0 1";
  run_test(board, "b1g6");

  // Test case 2
  board = "5r2/Rpp4p/3k2p1/3N4/2P1P1b1/1PQ3P1/1P5r/4KB1q w - - 0 1";
  run_test(board, "c3b4");

  // Test case 3
  board = "2RR1K2/1B3PPq/5Q2/4P3/7n/Ppp1b3/p5pp/3rk2N b - - 0 1";
  run_test(board, "d1d8");

  // Test case 4
  board = "3R1K2/1B3PPq/5Q2/2R1P3/7n/Ppp5/p5pp/3rk2N b - - 0 1";
  run_test(board, "d1d8");

  // Test case 5
  board = "3R3q/1B2KPP1/5Q2/2R1P3/7n/Ppp5/p5pp/3rk2N b - - 0 1";
  run_test(board, "h8d8");

  // Test case 6
  board = "3q4/1B3PP1/4KQ2/2R1P3/7n/Ppp5/p5pp/3rk2N b - - 0 1";
  run_test(board, "d8d7");

  // Test case 7
  board = "6k1/3r2p1/R4p2/1p2qP1Q/4P3/1P1P3P/1Pr2BP1/6K1 w - - 0 1";
  run_test(board, "a6a8");

  // Test case 8 - cheese blunder
  board = "r1bqk2r/pppp1p1p/5p2/2b1p2Q/2B1P3/8/PP1PKPPP/n1B3NR w kq - 0 1";
  run_test(board, "h5f7");
  // Will add more tests here as needed 

  std::cout << "Passed " << passed << " / " << total << " tests.\n";
  return passed == total ? 0 : 1;
}