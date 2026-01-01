# Monte Carlo Chess GPU - Quick Reference Card

## Kompilacija

### Windows

```powershell
cd gpu
build.bat
```

### Linux

```bash
cd gpu
chmod +x build.sh
./build.sh
```

### Make (oba sistema)

```bash
cd gpu
make                    # Build
make clean              # Clean
make rebuild            # Clean + Build
make run                # Build + Run (10K sims)
make run-benchmark      # Build + Run (100K sims)
```

## Pokretanje

```bash
# Default test pozicije (10K simulacija)
./monte_carlo_chess_gpu

# Custom broj simulacija
./monte_carlo_chess_gpu 50000

# Custom FEN pozicija
./monte_carlo_chess_gpu 50000 "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

## CUDA provera

```bash
# Proveri GPU
nvidia-smi

# Proveri CUDA toolkit
nvcc --version

# Proveri compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Build sa custom CUDA arhitekturom

```bash
# Makefile
make CUDA_ARCH=sm_86

# Ručno (nvcc)
nvcc -O3 -arch=sm_86 -c monte_carlo_kernel.cu
```

## CUDA arhitekture

| GPU      | Arch  |
| -------- | ----- |
| RTX 20xx | sm_75 |
| RTX 30xx | sm_86 |
| RTX 40xx | sm_89 |

## Debugovanje

```bash
# Compile sa debug info
nvcc -g -G -O0 monte_carlo_kernel.cu -c

# Run sa compute-sanitizer
compute-sanitizer ./monte_carlo_chess_gpu 1000

# Profajliranje
nsys profile ./monte_carlo_chess_gpu 50000
```

## Ključni fajlovi

| Fajl                       | Opis                     |
| -------------------------- | ------------------------ |
| `monte_carlo_kernel.cu`    | CUDA kernel (simulacije) |
| `monte_carlo_gpu.cpp/.hpp` | C++ wrapper              |
| `main_gpu.cpp`             | Main program             |
| `build.bat` / `build.sh`   | Build skripte            |
| `Makefile`                 | Alternative build        |

## Performanse (očekivane)

| GPU      | Sims/sec |
| -------- | -------- |
| GTX 1660 | 30K      |
| RTX 3060 | 50K      |
| RTX 3080 | 80K      |
| RTX 4090 | 150K     |

## Troubleshooting

| Problem                      | Rešenje                                |
| ---------------------------- | -------------------------------------- |
| `nvcc not found`             | Dodaj CUDA bin u PATH                  |
| `No CUDA device`             | Instaliraj NVIDIA drajvere             |
| `architecture not supported` | Prilagodi -arch flag                   |
| Loše performanse             | Poveći broj simulacija, prilagodi arch |

## Fajl struktura

```
gpu/
├── monte_carlo_kernel.cu      # CUDA kernel
├── monte_carlo_gpu.cpp/hpp    # C++ wrapper
├── main_gpu.cpp               # Main program
├── build.bat / build.sh       # Build skripte
├── Makefile                   # Build sistem
├── README.md                  # Glavna dokumentacija
├── QUICKSTART.md              # Brzi start
├── IMPLEMENTATION_SUMMARY.md  # Tehnički detalji
├── CUDA_SETUP.md              # CUDA instalacija
├── NEXT_STEPS.md              # Razvoj planova
└── REFERENCE_CARD.md          # Ovaj fajl
```

## CPU vs GPU komande

### CPU verzija

```bash
cd build
./test_suite_parallel --mode=depth --depth=10 --level=easy
```

### GPU verzija

```bash
cd gpu
./monte_carlo_chess_gpu 50000
```

## Dokumentacija lokacije

| Dokument                    | Tema                            |
| --------------------------- | ------------------------------- |
| `README.md`                 | Pregled, instalacija, upotreba  |
| `QUICKSTART.md`             | Brze komande za pokretanje      |
| `IMPLEMENTATION_SUMMARY.md` | Tehnički detalji implementacije |
| `CUDA_SETUP.md`             | CUDA instalacija i setup        |
| `NEXT_STEPS.md`             | Plan za dalje unapređenje       |
| `REFERENCE_CARD.md`         | Ovaj quick reference            |

## Važni linkovi

- CUDA Downloads: https://developer.nvidia.com/cuda-downloads
- GPU Compute Capabilities: https://developer.nvidia.com/cuda-gpus
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Chess Programming Wiki: https://www.chessprogramming.org/

## Primer workflow

```bash
# 1. Setup
cd gpu
make clean

# 2. Build
make CUDA_ARCH=sm_86

# 3. Quick test
./monte_carlo_chess_gpu 1000

# 4. Benchmark
./monte_carlo_chess_gpu 100000

# 5. Custom pozicija
./monte_carlo_chess_gpu 50000 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
```

## Očekivani output

```
=== Monte Carlo Chess Engine - GPU Version ===
Using CUDA device: NVIDIA GeForce RTX 3080
Compute capability: 8.6

=== Testing Position: Starting Position ===
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Number of simulations: 50000

=== Monte Carlo Simulation Results ===
Total simulations: 50000
White wins: 17520 (35.04%)
Black wins: 16890 (33.78%)
Draws: 15590 (31.18%)
=======================================
Time taken: 612 ms
Simulations per second: 81699.35
```

## Sledeći koraci

1. ✅ Build i testiraj
2. ⏳ Dodaj checkmate detekciju
3. ⏳ Implementiraj MCTS selection
4. ⏳ Integriši sa chess.hpp
5. ⏳ Dodaj neural network evaluaciju

---

**Napomena:** Ova referentna karta sadrži najčešće korišćene komande i informacije. Za detaljnije uputstvo, pogledaj ostale .md fajlove u `gpu/` folderu.
