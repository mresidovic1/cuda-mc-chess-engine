#ifndef TEST_POSITIONS_H
#define TEST_POSITIONS_H

#include <string>

// Test Position Structure
struct TestPosition {
    const char* name;           
    const char* fen;            
    const char* expected_move;  //  UCI notation ("e2e4")
    int suggested_depth;        
    int suggested_iterations;   
    const char* category;       // "easy", "medium", "hard"
    const char* description;    
};

// Easy Positions (Mate in 1-2, obvious tactics)
// Positions verified from CPU tests
const TestPosition EASY_TESTS[] = {
    {
        "Mate in 1",
        "6k1/5ppp/p7/P7/5b2/7P/1r3PP1/3R2K1 w - - 0 1",
        "d1d8",
        2,
        5000,
        "easy",
        "Mate in 1"
    },
    {
        "Mate in 1",
        "1r2R3/8/2p2k1p/p5p1/Pp1n4/6Pq/QP3P2/4R1K1 b - - 0 1",
        "d4f3",
        2,
        5000,
        "easy",
        "Mate in 1"
    },
    {
        "Mate in 1",
        "3r2rk/p5pp/1p4n1/3p2N1/2pP4/2P1R3/qPBQ1PPP/6K1 w - - 0 1",
        "g5f7",
        2,
        5000,
        "easy",
        "Mate in 1"
    },
    {
        "Mate in 1",
        "r1b2rk1/pp3pp1/2nbpq1p/2pp2N1/3P3P/2P1P3/PPQ2PP1/RN2KB1R w KQ - 0 1",
        "c2h7",
        2,
        5000,
        "easy",
        "Mate in 1"
    },
    {
        "Mate in 1",
        "r1bqkb1r/pppp1ppp/2n2n2/3Q4/2B1P3/8/PB3PPP/RN2K1NR w KQkq - 0 1",
        "d5f7",
        2,
        5000,
        "easy",
        "Mate in 1"
    }
};

const int NUM_EASY_TESTS = sizeof(EASY_TESTS) / sizeof(TestPosition);

// Medium Positions (Mate in 4-5, tactical combinations)

const TestPosition MEDIUM_TESTS[] = {
    {
        "Mate in 4",
        "r4r1k/1R1R2p1/7p/8/8/3Q1Ppq/P7/6K1 w - - 0 1",
        "d3h7",
        4,
        15000,
        "medium",
        "Mate in 4"
    },
    {
        "Mate in 5",
        "2N5/5p2/6pp/7k/4N3/p3P1KP/1p6/3b4 w - - 0 1",
        "h3h4",
        5,
        15000,
        "medium",
        "Mate in 5"
    },
    {
        "Mate in 5",
        "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
        "h4f4",
        5,
        15000,
        "medium",
        "Mate in 5"
    },
    {
        "Mate in 5",
        "2q1nk1r/4Rp2/1ppp1P2/6Pp/3p1B2/3P3P/PPP1Q3/6K1 w - - 0 1",
        "e7e8",
        5,
        15000,
        "medium",
        "Mate in 5"
    },
    {
        "Mate in 5",
        "6r1/p3p1rk/1p1pPp1p/q3n2R/4P3/3BR2P/PPP2QP1/7K w - - 0 1",
        "h5h6",
        5,
        15000,
        "medium",
        "Mate in 5"
    }
};

const int NUM_MEDIUM_TESTS = sizeof(MEDIUM_TESTS) / sizeof(TestPosition);

// Hard Positions (Deep tactics, mate in 8-12)

const TestPosition HARD_TESTS[] = {
    {
        "Mate in 8",
        "r1bq2nr/pp2k2p/3bB3/5pPP/1n1P4/4Q3/PBP1N1P1/R4RK1 w - - 1 20",
        "e6f5",
        8,
        50000,
        "hard",
        "Mate in 8"
    },
    {
        "Mate in 12",
        "rn4k1/5rbn/1p1p4/1p1q1p1p/3P4/P1B1P2P/Q2N1PR1/1K4R1 w - - 2 27",
        "a2d5",
        12,
        100000,
        "hard",
        "Mate in 12"
    },
    {
        "Mate in 8",
        "r4k2/pb6/1p2qP1p/P2p3r/5Q2/1Pn1RN1P/5PB1/6K1 w - - 1 33",
        "e3e6",
        8,
        75000,
        "hard",
        "Mate in 8"
    },
    {
        "Mate in 8-12",
        "1r1qk1nr/pp2b2p/4p3/1b1P2BP/1PB3Pn/1R4N1/P1P5/3Q1RK1 w - - 1 24",
        "c4b5",
        10,
        80000,
        "hard",
        "Mate in 8-12"
    },
    {
        "Black is winning",
        "3qb2k/1p6/1P5p/p1P1pBrn/P1Nb2p1/B2Q4/7K/5R2 b - - 4 54",
        "g5f5",
        8,
        60000,
        "hard",
        "Black is winning"
    }
};

const int NUM_HARD_TESTS = sizeof(HARD_TESTS) / sizeof(TestPosition);

// Perft Positions (for move generation validation)

struct PerftPosition {
    const char* name;
    const char* fen;
    int depth;
    unsigned long long expected_nodes;
};

const PerftPosition PERFT_TESTS[] = {
    // Starting position tests
    {
        "Starting Position Depth 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        1,
        20
    },
    {
        "Starting Position Depth 2",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        2,
        400
    },
    {
        "Starting Position Depth 3",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        3,
        8902
    },
    {
        "Starting Position Depth 4",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        4,
        197281
    },

    // Kiwipete - complex position with many tactics
    {
        "Kiwipete Depth 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        1,
        48
    },
    {
        "Kiwipete Depth 2",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        2,
        2039
    },
    {
        "Kiwipete Depth 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        3,
        97862
    },

    // Position 3 - tests en passant and king movement
    {
        "Position 3 Depth 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        1,
        14
    },
    {
        "Position 3 Depth 2",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        2,
        191
    },
    {
        "Position 3 Depth 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        3,
        2812
    },

    // Position 4 - tests castling with rook captures
    {
        "Position 4 Depth 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        1,
        6
    },
    {
        "Position 4 Depth 2",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        2,
        264
    },
    {
        "Position 4 Depth 3",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        3,
        9467
    },

    // Position 5 - promotion tests
    {
        "Position 5 (Promotions) Depth 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        1,
        44
    },
    {
        "Position 5 (Promotions) Depth 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        2,
        1486
    },
    {
        "Position 5 (Promotions) Depth 3",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        3,
        62379
    }
};

const int NUM_PERFT_TESTS = sizeof(PERFT_TESTS) / sizeof(PerftPosition);

// FEN Validation Tests

struct FENValidationTest {
    const char* fen;
    bool should_be_valid;
    const char* description;
};

const FENValidationTest FEN_TESTS[] = {
    {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        true,
        "Valid starting position"
    },
    {
        "8/8/8/8/8/8/8/8 w - - 0 1",
        false,
        "Invalid: No kings"
    },
    {
        "k7/8/8/8/8/8/8/K7 w - - 0 1",
        true,
        "Valid: Just kings"
    },
    {
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        true,
        "Valid: After 1.e4 with en passant"
    },
    {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        false,
        "Invalid: Bad side to move"
    },
    {
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        true,
        "Valid: Both sides can castle"
    },
    {
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        true,
        "Valid: Complex endgame position"
    }
};

const int NUM_FEN_TESTS = sizeof(FEN_TESTS) / sizeof(FENValidationTest);

#endif 
