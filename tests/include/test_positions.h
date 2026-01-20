// test_positions.h - Standard chess test positions for benchmarking
// Contains tactical puzzles, famous positions, and varied complexity levels

#ifndef TEST_POSITIONS_H
#define TEST_POSITIONS_H

#include <string>
#include <vector>

// ============================================================================
// Test Position Structure
// ============================================================================

struct TestPosition {
    std::string name;
    std::string fen;
    std::string best_move;  // UCI notation (if known)
    std::string difficulty; // "easy", "medium", "hard"
    std::string category;   // "tactical", "endgame", "positional", etc.
};

// ============================================================================
// Bratko-Kopec Test Suite (24 positions)
// Classic engine testing positions
// ============================================================================

inline std::vector<TestPosition> get_bratko_kopec_suite() {
    return {
        {"BK01", "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", "d6d1", "medium", "tactical"},
        {"BK02", "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1", "d4d5", "hard", "positional"},
        {"BK03", "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1", "f6f5", "medium", "tactical"},
        {"BK04", "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1", "e5e6", "easy", "tactical"},
        {"BK05", "r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1", "d4g7", "medium", "tactical"},
        {"BK06", "2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1", "g5g6", "medium", "tactical"},
        {"BK07", "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1", "h5f6", "hard", "tactical"},
        {"BK08", "4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - - 0 1", "f4f5", "medium", "endgame"},
        {"BK09", "2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - - 0 1", "f4f5", "medium", "tactical"},
        {"BK10", "3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1", "c6e5", "hard", "tactical"},
        {"BK11", "2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - - 0 1", "f2f4", "hard", "positional"},
        {"BK12", "r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1", "d7f5", "easy", "tactical"},
        {"BK13", "r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - - 0 1", "b2b4", "hard", "positional"},
        {"BK14", "rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - - 0 1", "b3c4", "medium", "positional"},
        {"BK15", "2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1", "g4g7", "easy", "tactical"},
        {"BK16", "r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1 w kq - 0 1", "g5e7", "medium", "tactical"},
        {"BK17", "r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - - 0 1", "h7h5", "hard", "positional"},
        {"BK18", "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1", "c6b4", "medium", "tactical"},
        {"BK19", "3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - - 0 1", "g6e5", "medium", "tactical"},
        {"BK20", "r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - - 0 1", "g3g4", "hard", "tactical"},
        {"BK21", "3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - - 0 1", "f5h6", "medium", "tactical"},
        {"BK22", "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1", "b7e4", "hard", "tactical"},
        {"BK23", "r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1", "f7f6", "medium", "tactical"},
        {"BK24", "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1", "f2f4", "hard", "positional"}
    };
}

// ============================================================================
// Win At Chess (WAC) - Selected positions
// ============================================================================

inline std::vector<TestPosition> get_wac_suite() {
    return {
        {"WAC001", "r1b1kb1r/pppp1ppp/5n2/4q3/4P3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "d3b5", "easy", "tactical"},
        {"WAC002", "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", "d6d1", "medium", "tactical"},
        {"WAC003", "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1", "d4d5", "hard", "positional"},
        {"WAC004", "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1", "f6f5", "medium", "tactical"},
        {"WAC005", "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1", "e5e6", "easy", "tactical"}
    };
}

// ============================================================================
// Performance Testing Positions
// Varied complexity for throughput benchmarks
// ============================================================================

inline std::vector<TestPosition> get_performance_suite() {
    return {
        // Simple middlegame positions
        {"Simple01", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "", "easy", "opening"},
        {"Simple02", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", "", "easy", "opening"},
        {"Simple03", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "", "easy", "opening"},
        
        // Complex middlegame
        {"Complex01", "r2qkb1r/pb1n1p2/2p1pn1p/1p2N1pP/3PP1P1/2N5/PPP2PB1/R1BQK2R w KQkq - 0 1", "", "hard", "middlegame"},
        {"Complex02", "r1b1kb1r/1p1n1ppp/p1n1pq2/3p4/Q2P4/2NBPN2/PP3PPP/R1B2RK1 b kq - 0 1", "", "hard", "middlegame"},
        
        // Endgames
        {"Endgame01", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "", "medium", "endgame"},
        {"Endgame02", "8/8/1p1k4/3Pp3/1PP1K3/8/8/8 b - - 0 1", "", "medium", "endgame"},
        
        // High branching factor
        {"HighBranching", "r1bqk2r/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 0 1", "", "hard", "middlegame"}
    };
}

// ============================================================================
// All Positions Combined
// ============================================================================

inline std::vector<TestPosition> get_all_positions() {
    auto all = get_bratko_kopec_suite();
    auto wac = get_wac_suite();
    auto perf = get_performance_suite();
    
    all.insert(all.end(), wac.begin(), wac.end());
    all.insert(all.end(), perf.begin(), perf.end());
    
    return all;
}

// ============================================================================
// Filter by difficulty
// ============================================================================

inline std::vector<TestPosition> filter_by_difficulty(const std::vector<TestPosition>& positions, 
                                                       const std::string& difficulty) {
    std::vector<TestPosition> filtered;
    for (const auto& pos : positions) {
        if (pos.difficulty == difficulty) {
            filtered.push_back(pos);
        }
    }
    return filtered;
}

#endif // TEST_POSITIONS_H
