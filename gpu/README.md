# Monte Carlo Chess Engine - GPU Version

## Pregled

Ova implementacija koristi Monte Carlo simulacije na GPU-u (CUDA) za evaluaciju šahovskih pozicija. Za razliku od CPU verzije koja koristi Minimax/Negamax algoritam, GPU verzija pokreće hiljade nasumičnih simulacija paralelno za procenu koja strana ima prednost.

## Implementacija

### Arhitektura

1. **monte_carlo_kernel.cu** - CUDA kernel koji simulira partije

   - Svaki GPU thread simulira jednu kompletnu partiju od trenutne pozicije
   - Koristi nasumične poteze (random playouts)
   - Vraća rezultat: pobeda belog, pobeda crnog, ili remi

2. **monte_carlo_gpu.hpp/cpp** - C++ wrapper za CUDA kernel

   - Upravlja CUDA memory alokacijom
   - Konvertuje FEN notaciju u GPU-friendly format
   - Agregira rezultate iz svih simulacija

3. **main_gpu.cpp** - Glavni program za testiranje
   - Testira različite pozicije
   - Prikazuje statistiku (% pobeda, remija)
   - Meri performanse (simulacije/sekund)

### Razlike od CPU verzije

| Aspekt         | CPU (Minimax/Negamax)           | GPU (Monte Carlo)        |
| -------------- | ------------------------------- | ------------------------ |
| Algoritam      | Deterministička pretraga stabla | Stohastičke simulacije   |
| Paralelizacija | OpenMP (thread-level)           | CUDA (hiljada threadova) |
| Evaluacija     | Piece-square tabele, heuristike | Win/loss statistika      |
| Dubina         | Ograničena (10-20 ply)          | Puna partija (do kraja)  |
| Performanse    | 100K-1M pozicija/s              | 10K-100K simulacija/s    |

## Kompilacija

### Preduslovi

- **NVIDIA GPU** sa CUDA podrškom (Compute Capability 7.0+)
- **CUDA Toolkit** (12.x ili noviji)
- **g++** kompajler (GCC 9+ ili MSVC)

### Windows

```bash
cd gpu
build.bat
```

**Napomena:** Možda ćeš morati da prilagodiš CUDA_PATH u build.bat ako CUDA nije instalirana na default lokaciji.

### Linux

```bash
cd gpu
chmod +x build.sh
./build.sh
```

**Napomena:** Možda ćeš morati da prilagodiš CUDA arhitekturu (sm_70, sm_75, sm_80, itd.) u build.sh za tvoju GPU.

## Pokretanje

### Osnovni test (10,000 simulacija)

```bash
# Windows
monte_carlo_chess_gpu.exe

# Linux
./monte_carlo_chess_gpu
```

### Sa većim brojem simulacija

```bash
# Windows
monte_carlo_chess_gpu.exe 100000

# Linux
./monte_carlo_chess_gpu 100000
```

### Sa custom FEN pozicijom

```bash
# Windows
monte_carlo_chess_gpu.exe 50000 "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Linux
./monte_carlo_chess_gpu 50000 "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

## Očekivani izlaz

```
=== Monte Carlo Chess Engine - GPU Version ===
Using CUDA device: NVIDIA GeForce RTX 3080
Compute capability: 8.6

=== Testing Position: Starting Position ===
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Number of simulations: 10000

=== Monte Carlo Simulation Results ===
Total simulations: 10000
White wins: 3520 (35.20%)
Black wins: 3380 (33.80%)
Draws: 3100 (31.00%)
=======================================
Time taken: 245 ms
Simulations per second: 40816.33
```

## Performanse

| GPU Model | Simulacije/s (aprox.) |
| --------- | --------------------- |
| RTX 3060  | 30,000 - 50,000       |
| RTX 3080  | 50,000 - 80,000       |
| RTX 4090  | 100,000 - 150,000     |
| A100      | 150,000 - 250,000     |

_Napomena: Performanse zavise od kompleksnosti pozicije i dužine simulacija._

## Ograničenja trenutne implementacije

1. **Jednostavna generacija poteza**: Trenutno se koristi pojednostavljena implementacija bez svih šahovskih pravila (npr. bez en passant u nekim slučajevima, promocija nije kompletna)

2. **Bez prave evaluacije**: Simulacije koriste samo nasumične poteze, što nije optimalno. Poboljšanja:

   - Dodati heuristike za odabir poteza
   - Implementirati selection/expansion faze MCTS-a
   - Koristiti neural network za evaluaciju (AlphaZero stil)

3. **Bez checkmate detekcije**: Trenutna implementacija ne proverava da li je pozicija checkmate, već samo da li ima legalnih poteza

## Sledeći koraci

1. **Implementirati potpunu MCTS strukturu**:

   - Selection (UCB formula)
   - Expansion
   - Simulation (trenutno implementirano)
   - Backpropagation

2. **Dodati checkpoint/caching**:

   - Sačuvaj MCTS stablo između poteza
   - Transpozicione tabele na GPU

3. **Integracija sa postojećom evaluacijom**:

   - Koristiti piece-square tabele iz CPU verzije
   - Hibridni pristup: Monte Carlo za taktičke pozicije, Minimax za mirne

4. **Neural network evaluacija**:
   - Treniraj mali NN za pozicijsku evaluaciju
   - Koristi ga umesto random playouts

## Poređenje sa CPU verzijom

Za pokretanje CPU verzije iz root foldera:

```bash
# Build CPU version
cd ..
meson setup build
meson compile -C build

# Run tests
./build/test_suite_parallel --mode=depth --depth=10 --level=easy
```

## Troubleshooting

### "No CUDA-capable device found"

- Proveri da li imaš NVIDIA GPU sa CUDA podrškom
- Ažuriraj GPU drajvere

### "nvcc not found"

- Instaliraj CUDA Toolkit
- Dodaj CUDA bin folder u PATH

### Compile errors

- Proveri da li CUDA_PATH u build skripti pokazuje na ispravnu lokaciju
- Proveri da li imaš kompatibilnu verziju g++ kompajlera (GCC 9-11 preporučeno za CUDA 12.x)

### Performanse su lošije od očekivanih

- Poveća broj simulacija (više paralelizma)
- Proveri da li GPU nije opterećen drugim procesima
- Prilagodi CUDA arhitekturu u build skripti za tvoju GPU

## Licenca

MIT License - same as the parent project
