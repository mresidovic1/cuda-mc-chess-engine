#ifndef TEST_POSITIONS_H
#define TEST_POSITIONS_H

#include <string>
#include <vector>

struct TestPosition {
    std::string fen;
    std::string expected_move;
    std::string description;
    int difficulty;
};

constexpr int EASY = 1;
constexpr int MEDIUM = 2;
constexpr int HARD = 3;

const std::vector<TestPosition> EASY_POSITIONS = {
    {"6k1/5ppp/p7/P7/5b2/7P/1r3PP1/3R2K1 w - - 0 1", "d1d8", "Mate in 1", EASY},
    {"1r2R3/8/2p2k1p/p5p1/Pp1n4/6Pq/QP3P2/4R1K1 b - - 0 1", "d4f3", "Mate in 1", EASY},
    {"3r2rk/p5pp/1p4n1/3p2N1/2pP4/2P1R3/qPBQ1PPP/6K1 w - - 0 1", "g5f7", "Mate in 1", EASY},
    {"r1b2rk1/pp3pp1/2nbpq1p/2pp2N1/3P3P/2P1P3/PPQ2PP1/RN2KB1R w KQ - 0 1", "c2h7", "Mate in 1", EASY},
    {"r1bqkb1r/pppp1ppp/2n2n2/3Q4/2B1P3/8/PB3PPP/RN2K1NR w KQkq - 0 1", "d5f7", "Mate in 1", EASY},
};

const std::vector<TestPosition> MEDIUM_POSITIONS = {
    {"r4r1k/1R1R2p1/7p/8/8/3Q1Ppq/P7/6K1 w - - 0 1", "d3h7", "Mate in 4", MEDIUM},
    {"2N5/5p2/6pp/7k/4N3/p3P1KP/1p6/3b4 w - - 0 1", "h3h4", "Mate in 5", MEDIUM},
    {"6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1", "h4f4", "Mate in 5", MEDIUM},
    {"2q1nk1r/4Rp2/1ppp1P2/6Pp/3p1B2/3P3P/PPP1Q3/6K1 w - - 0 1", "e7e8", "Mate in 5", MEDIUM},
    {"6r1/p3p1rk/1p1pPp1p/q3n2R/4P3/3BR2P/PPP2QP1/7K w - - 0 1", "h5h6", "Mate in 5", MEDIUM},
};

const std::vector<TestPosition> HARD_POSITIONS = {
    {"r1bq2nr/pp2k2p/3bB3/5pPP/1n1P4/4Q3/PBP1N1P1/R4RK1 w - - 1 20", "e6f5", "Mate in 8", HARD},
    {"rn4k1/5rbn/1p1p4/1p1q1p1p/3P4/P1B1P2P/Q2N1PR1/1K4R1 w - - 2 27", "a2d5", "Mate in 12", HARD},
    {"r4k2/pb6/1p2qP1p/P2p3r/5Q2/1Pn1RN1P/5PB1/6K1 w - - 1 33", "e3e6", "Mate in 8", HARD},
    {"1r1qk1nr/pp2b2p/4p3/1b1P2BP/1PB3Pn/1R4N1/P1P5/3Q1RK1 w - - 1 24", "c4b5", "Mate in 8-12", HARD},
    {"3qb2k/1p6/1P5p/p1P1pBrn/P1Nb2p1/B2Q4/7K/5R2 b - - 4 54", "g5f5", "Black is winning", HARD},
};

inline std::vector<TestPosition> get_positions_by_difficulty(int difficulty) {
    switch (difficulty) {
        case EASY: return EASY_POSITIONS;
        case MEDIUM: return MEDIUM_POSITIONS;
        case HARD: return HARD_POSITIONS;
        default: return {};
    }
}

inline std::vector<TestPosition> get_all_positions() {
    std::vector<TestPosition> all;
    all.insert(all.end(), EASY_POSITIONS.begin(), EASY_POSITIONS.end());
    all.insert(all.end(), MEDIUM_POSITIONS.begin(), MEDIUM_POSITIONS.end());
    all.insert(all.end(), HARD_POSITIONS.begin(), HARD_POSITIONS.end());
    return all;
}

inline std::string difficulty_name(int diff) {
    switch (diff) {
        case EASY: return "EASY";
        case MEDIUM: return "MEDIUM";
        case HARD: return "HARD";
        default: return "UNKNOWN";
    }
}

#endif

