# Sledeći koraci - Kako nastaviti sa razvojem

## Kratkoročni prioriteti (1-2 nedelje)

### 1. Testiraj trenutnu implementaciju

**Koraci:**

```bash
cd gpu
build.bat  # ili build.sh

# Testiranje sa različitim brojem simulacija
monte_carlo_chess_gpu.exe 1000
monte_carlo_chess_gpu.exe 10000
monte_carlo_chess_gpu.exe 50000
monte_carlo_chess_gpu.exe 100000

# Testiraj sa custom pozicijom
monte_carlo_chess_gpu.exe 50000 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
```

**Šta proveriti:**

- Da li se kompajlira bez grešaka?
- Koliko je brzo (simulacije/sekund)?
- Da li rezultati imaju smisla (približno 33-35% za svaku stranu na starting poziciji)?

### 2. Implementiraj potpunu generaciju poteza

**Trenutna ograničenja:**

- En passant nije potpuno implementiran
- Promocija nije potpuno implementirana
- Nema checkmate detekcije

**Zadatak:** Proširiti `monte_carlo_kernel.cu`

**Fajlovi za izmenu:**

- `gpu/monte_carlo_kernel.cu` - dodaj funkcije za en passant, promociju, checkmate

**Primer koda za promociju:**

```cpp
// U generate_pseudo_legal_moves(), dodaj za pione:
if (piece_type == PAWN) {
    int target_rank = (my_color == WHITE_MASK) ? 7 : 0;

    if (rank == target_rank - direction) {
        // Promotion moves
        int forward_sq = square_index(rank + direction, file);
        if (is_valid_square(forward_sq) && GET_PIECE_TYPE(board->squares[forward_sq]) == EMPTY) {
            // Dodaj 4 poteza (Q, R, B, N)
            moves[move_count++] = {(uint8_t)sq, (uint8_t)forward_sq, QUEEN};
            moves[move_count++] = {(uint8_t)sq, (uint8_t)forward_sq, ROOK};
            moves[move_count++] = {(uint8_t)sq, (uint8_t)forward_sq, BISHOP};
            moves[move_count++] = {(uint8_t)sq, (uint8_t)forward_sq, KNIGHT};
        }
    }
}
```

### 3. Dodaj checkmate detekciju

**Zadatak:** Implementiraj funkciju `is_checkmate()`

**Potrebne funkcije:**

```cpp
__device__ bool is_square_attacked(GPUBoard* board, int sq, uint8_t attacker_color);
__device__ bool is_in_check(GPUBoard* board);
__device__ bool is_checkmate(GPUBoard* board, int move_count);
```

**Primer:**

```cpp
__device__ bool is_checkmate(GPUBoard* board, int move_count) {
    if (move_count > 0) return false;  // Ima poteza

    // Pronađi kralja
    uint8_t my_color = board->side_to_move == 0 ? WHITE_MASK : BLACK_MASK;
    int king_sq = -1;
    for (int sq = 0; sq < 64; sq++) {
        if (GET_PIECE_TYPE(board->squares[sq]) == KING &&
            GET_COLOR(board->squares[sq]) == my_color) {
            king_sq = sq;
            break;
        }
    }

    // Proveri da li je kralj u šahu
    return is_square_attacked(board, king_sq, ~my_color);
}
```

### 4. Benchmarkuj performanse

**Napravi benchmark skriptu:**

`gpu/benchmark.sh` (Linux) ili `gpu/benchmark.bat` (Windows):

```bash
#!/bin/bash
echo "=== Monte Carlo GPU Benchmark ==="
echo ""

for sims in 1000 5000 10000 50000 100000 200000; do
    echo "Running $sims simulations..."
    ./monte_carlo_chess_gpu $sims | grep "Simulations per second"
    echo ""
done
```

**Uporedi sa CPU verzijom:**

- Pokreni CPU version benchmark
- Uporedi throughput (pozicije/s vs simulacije/s)
- Dokumentuj rezultate

## Srednjoročni prioriteti (1-2 meseca)

### 5. Implementiraj MCTS Selection fazu

**Cilj:** Umesto purely random playouts, koristi UCB formulu za odabir poteza

**Šta treba:**

1. Struktura za MCTS čvor:

```cpp
struct MCTSNode {
    int visits;
    float value;
    int parent;
    int first_child;
    int num_children;
    SimpleMove move;
};
```

2. UCB formula:

```cpp
__device__ float ucb_score(MCTSNode* node, int parent_visits) {
    float exploitation = node->value / (float)node->visits;
    float exploration = sqrtf(2.0f * logf(parent_visits) / (float)node->visits);
    return exploitation + exploration;
}
```

3. Selection funkcija:

```cpp
__device__ int select_child(MCTSNode* tree, int node_idx, int parent_visits);
```

### 6. Dodaj heuristike za brže playouts

**Ideje:**

- Preference za captures
- Preference za checks
- Koristi piece-square tabele za evaluaciju

**Primer:**

```cpp
__device__ int score_move(GPUBoard* board, SimpleMove move) {
    int score = 0;

    // Capture bonus
    uint8_t captured = GET_PIECE_TYPE(board->squares[move.to]);
    if (captured != EMPTY) {
        score += 100 * piece_values[captured];
    }

    // Position bonus (piece-square tables)
    uint8_t piece = GET_PIECE_TYPE(board->squares[move.from]);
    score += pst[piece][move.to] - pst[piece][move.from];

    return score;
}
```

### 7. Integriraj sa chess.hpp bibliotekom

**Cilj:** Koristiti postojeću chess biblioteku umesto pojednostavljene reprezentacije

**Plan:**

1. Ekstraktuj relevantne delove chess.hpp u CUDA-kompatibilne header-e
2. Izbaci STL zavisnosti
3. Koristi bitboard operacije umesto unit8_t array

**Prednosti:**

- Puna implementacija šahovskih pravila
- Bitboard operacije (brže)
- Konzistentnost sa CPU verzijom

### 8. Dodaj "best move" selekciju

**Cilj:** Ne samo statistika, već i konkretni best move

**Implementacija:**

```cpp
struct MoveWithStats {
    SimpleMove move;
    int simulations;
    int wins;
    int losses;
    int draws;

    float win_rate() const {
        return (float)wins / simulations;
    }
};

// Testiraj svaki legalni potez iz root pozicije
MCTSResults run_for_each_move(GPUBoard& board, int sims_per_move);
```

### 9. Transpozicijska tabla na GPU

**Cilj:** Cache često viđenih pozicija

**Struktura:**

```cpp
struct TTEntryGPU {
    uint64_t hash;
    int visits;
    float value;
};

__device__ TTEntryGPU* tt_probe(uint64_t hash);
__device__ void tt_store(uint64_t hash, int visits, float value);
```

## Dugoročni prioriteti (3-6 meseci)

### 10. Neural Network evaluacija

**Ideja:** Umesto random playouts, koristi NN za evaluaciju pozicije

**Tehnologije:**

- CUDA + cuDNN za NN inference
- PyTorch/TensorFlow za treniranje NN
- ONNX za export modela

**Arhitektura NN:**

- Input: Board representation (8x8x12 - piece types)
- Hidden: Conv layers (kao AlphaZero)
- Output: Value head (-1 do 1) + Policy head (probability distribution over moves)

**Workflow:**

1. Prikupi self-play partije sa trenutnom implementacijom
2. Treniraj mali NN (npr. 3 conv layers, 128 filters)
3. Export u ONNX
4. Load u CUDA kernel i koristi za evaluaciju

### 11. Hibridni CPU+GPU pristup

**Ideja:** Najbolje od oba sveta

**Arhitektura:**

```
CPU: MCTS tree management
  |
  ├─> Leaf expansion
  |
  └─> GPU: Batch playouts (1000s simultano)
       |
       └─> Results back to CPU for backpropagation
```

**Prednosti:**

- CPU efikasan za stablo (mala memorija, dynamic)
- GPU efikasan za simulacije (masovna paralelizacija)

### 12. Distributed MCTS preko mreže

**Ideja:** Koristi više računara sa GPU-ima

**Tehnologija:**

- MPI (Message Passing Interface)
- gRPC za komunikaciju
- Root parallelization ili tree parallelization

## Kako testirati i validirati

### Unit testovi

**Napravi `gpu/tests/` folder:**

```cpp
// test_move_generation.cu
__global__ void test_pawn_moves() {
    GPUBoard board;
    // Setup board
    // Generate moves
    // Assert expected moves
}
```

### Integration testovi

**Test protiv poznateih pozicija:**

```cpp
// Test mate in 1
test_position("6k1/5ppp/p7/P7/5b2/7P/1r3PP1/3R2K1 w - - 0 1");
// Očekujem da white_wins > 95%
```

### Poređenje sa Stockfish

**Koristi Stockfish kao baseline:**

```bash
# Pokreni Stockfish evaluaciju
stockfish << EOF
position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
go depth 10
EOF

# Uporedi sa Monte Carlo rezultatima
./monte_carlo_chess_gpu 100000
```

## Dokumentacija razvoja

**Napravi development log:**

`gpu/DEVELOPMENT_LOG.md`:

```markdown
# Development Log

## 2026-01-01: Initial Implementation

- Created basic Monte Carlo kernel
- Simplified move generation
- Random playouts
- Performance: 50K sims/s on RTX 3080

## 2026-01-XX: Add Promotion

- Implemented pawn promotion
- Added 4 promotion types (Q, R, B, N)
- Performance: ...

...
```

## Debugging tips

### Koristiti CUDA-GDB

```bash
# Compile sa debug
nvcc -g -G monte_carlo_kernel.cu

# Run sa cuda-gdb
cuda-gdb ./monte_carlo_chess_gpu
(cuda-gdb) break monte_carlo_playout_kernel
(cuda-gdb) run 1000
(cuda-gdb) cuda thread (0,0,0)  # Switch to specific thread
```

### Print debugging

```cpp
__global__ void debug_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug: value = %d\n", value);
    }
}
```

### Compute Sanitizer

```bash
# Detect memory errors
compute-sanitizer ./monte_carlo_chess_gpu 1000

# Detect race conditions
compute-sanitizer --tool racecheck ./monte_carlo_chess_gpu 1000
```

## Resursi za učenje

### CUDA programiranje:

1. [CUDA by Example](https://developer.nvidia.com/cuda-example) - knjiga
2. [CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/) - video tutoriali
3. [CUDA Samples](https://github.com/NVIDIA/cuda-samples) - primeri koda

### Monte Carlo Tree Search:

1. [MCTS Survey](https://ieeexplore.ieee.org/document/6145622) - akademski paper
2. [AlphaGo Paper](https://www.nature.com/articles/nature16961) - Nature članak
3. [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - arXiv

### Chess programming:

1. [Chess Programming Wiki](https://www.chessprogramming.org/)
2. [Stockfish Source](https://github.com/official-stockfish/Stockfish)
3. [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) - NN-based chess engine

## Community i podrška

### Gde postaviti pitanja:

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [Stack Overflow - CUDA tag](https://stackoverflow.com/questions/tagged/cuda)
- [r/CUDA subreddit](https://www.reddit.com/r/CUDA/)
- [Computer Chess Discord](https://discord.gg/computerchess)

### Gde pokazati projekat:

- GitHub (open-source)
- Reddit (r/chess, r/programming, r/CUDA)
- NVIDIA Developer Blog (možda featured project)

## Zaključak

Implementacija je solidna osnova za dalje unapređenje. Prioriteti su:

1. **Kratkoročno:** Testiraj, dodaj checkmate, benchmark
2. **Srednjoročno:** Implementiraj MCTS selection, integriši sa chess.hpp
3. **Dugoročno:** Neural network evaluacija, hibridni pristup

**Srećno sa razvojem! 🚀**
