#include "../include/puct_mcts.h"
#include "../include/chess_types.cuh"
#include "../include/cpu_movegen.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

// External initialization functions
extern void init_attack_tables();
extern void init_startpos(BoardState* pos);

// Move notation helpers

const char* SQUARE_NAMES[64] = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"
};

const char PIECE_CHARS[] = "PNBRQK";
const char PROMO_CHARS[] = "nbrq";

std::string move_to_string(Move m) {
    if (m == 0) return "(none)";

    Square from = move_from(m);
    Square to = move_to(m);
    uint8_t flags = move_flags(m);

    std::string result = std::string(SQUARE_NAMES[from]) + SQUARE_NAMES[to];

    // Add promotion piece
    if (flags >= MOVE_PROMO_N) {
        result += PROMO_CHARS[flags & 0x3];
    }

    return result;
}

// Board printing

char piece_char(const BoardState* pos, int sq) {
    Bitboard bb = C64(1) << sq;

    for (int c = 0; c <= 1; c++) {
        for (int p = 0; p <= 5; p++) {
            if (pos->pieces[c][p] & bb) {
                char ch = PIECE_CHARS[p];
                return (c == WHITE) ? ch : (ch + 32);  
            }
        }
    }
    return '.';
}

void print_board(const BoardState* pos) {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int rank = 7; rank >= 0; rank--) {
        std::cout << (rank + 1) << " |";
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            std::cout << " " << piece_char(pos, sq) << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
    std::cout << "Side to move: " << (pos->side_to_move == WHITE ? "White" : "Black") << "\n";
    std::cout << "Castling: ";
    if (pos->castling & CASTLE_WK) std::cout << "K";
    if (pos->castling & CASTLE_WQ) std::cout << "Q";
    if (pos->castling & CASTLE_BK) std::cout << "k";
    if (pos->castling & CASTLE_BQ) std::cout << "q";
    if (pos->castling == 0) std::cout << "-";
    std::cout << "\n";
    if (pos->ep_square >= 0) {
        std::cout << "En passant: " << SQUARE_NAMES[pos->ep_square] << "\n";
    }
    std::cout << "Halfmove clock: " << (int)pos->halfmove << "\n";
    std::cout << std::endl;
}

// GPU Info

void print_gpu_info() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "========================================\n";
    std::cout << "GPU-Accelerated MCTS Chess Engine\n";
    std::cout << "========================================\n\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "CUDA cores: " << prop.multiProcessorCount * 128 << " (approx)\n";  // SM * 128 for Turing
    std::cout << "Global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    std::cout << "Shared memory per block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n\n";
}

// Removed old MCTSEngine functions - using PUCT only now

// Main

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --play          Play a self-play game (default)\n";
    std::cout << "  --benchmark     Run benchmark on starting position\n";
    std::cout << "  --sims N        Simulations per move (default: 5000)\n";
    std::cout << "  --batch N       Batch size for GPU (default: 512)\n";
    std::cout << "  --moves N       Max moves in self-play game (default: 200)\n";
    std::cout << "  --help          Show this help\n";
}

int main(int argc, char** argv) {
    // Parse arguments
    bool benchmark_mode = false;
    int simulations = 5000;
    int batch_size = 512;
    int max_moves = 200;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--benchmark") {
            benchmark_mode = true;
        } else if (arg == "--play") {
            benchmark_mode = false;
        } else if (arg == "--sims" && i + 1 < argc) {
            simulations = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--moves" && i + 1 < argc) {
            max_moves = std::stoi(argv[++i]);
        }
    }

    // Print GPU info
    print_gpu_info();

    // PUCT MCTS (Heuristic AlphaZero)
    std::cout << "\n=== PUCT MCTS Engine (Heuristic AlphaZero-style) ===\n";
    std::cout << "NO Neural Networks - Pure tactical heuristics\n\n";

    PUCTConfig config;
    config.num_simulations = simulations;
    config.batch_size = batch_size;
    config.verbose = true;
    config.info_interval = simulations / 10;

    PUCTEngine engine(config);
    engine.init();
    std::cout << "PUCT Engine ready!\n\n";

    // Initialize position
    BoardState pos;
    init_startpos(&pos);

    if (benchmark_mode) {
        std::cout << "Running PUCT benchmark...\n";
        print_board(&pos);

        auto start = std::chrono::high_resolution_clock::now();
        Move best_move = engine.search(pos);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();

        std::cout << "\n=== PUCT Results ===\n";
        std::cout << "Best move: " << move_to_string(best_move) << "\n";
        std::cout << "Total visits: " << engine.get_total_visits() << "\n";
        std::cout << "Root value: " << engine.get_root_value() << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << " s\n";
        std::cout << "Sims/sec: " << (int)(simulations / elapsed) << "\n";

        // Print PV
        std::vector<Move> pv = engine.get_pv(5);
        std::cout << "PV: ";
        for (Move m : pv) {
            std::cout << move_to_string(m) << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "Playing game with PUCT MCTS...\n";
        for (int move_num = 0; move_num < max_moves; move_num++) {
            Move best_move = engine.search(pos);

            if (best_move == 0) {
                std::cout << "Game over at move " << move_num << "\n";
                break;
            }

            std::cout << "Move " << (move_num + 1) << ": " << move_to_string(best_move) << "\n";
            cpu_movegen::make_move_cpu(&pos, best_move);

            if (move_num % 10 == 9) {
                print_board(&pos);
            }
        }
    }

    return 0;
}
