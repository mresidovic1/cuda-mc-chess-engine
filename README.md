# Chess Engine - CPU & GPU Parallelization

Ovaj projekat sadrži implementacije šahovskog engine-a optimizovane za paralelizaciju na CPU i GPU.

## Struktura projekta

- **CPU verzija** (`src/`, `include/`): Minimax/Negamax algoritam sa OpenMP paralelizacijom
- **GPU verzija** (`gpu/`): Monte Carlo Tree Search (MCTS) implementacija sa CUDA

## CPU Verzija (Minimax/Negamax)

Kompletan chess library nalazi se u `/include/chess.hpp`.

Za korištenje u projektu:

```cpp
#include '../include/chess.hpp'
```

Ovaj library koristi [Meson build sistem](https://mesonbuild.com/Quick-guide.html) za kompajliranje.

### Koraci za instalaciju CPU verzije:

```bash
   meson setup build
```

```bash
   meson compile -C build
```

```bash
   # Pokretanje testova
   build/chess-tests
   build/test_suite_parallel --mode=depth --depth=10 --level=easy
```

## GPU Verzija (Monte Carlo)

GPU verzija koristi CUDA za masivnu paralelizaciju Monte Carlo simulacija.

### Preduslovi za GPU verziju:

- NVIDIA GPU sa CUDA podrškom (Compute Capability 7.0+)
- CUDA Toolkit 12.x ili noviji
- g++ kompajler

### Kompilacija GPU verzije:

**Windows:**

```bash
cd gpu
build.bat
```

**Linux:**

```bash
cd gpu
chmod +x build.sh
./build.sh
```

### Pokretanje GPU verzije:

```bash
# Windows
gpu\monte_carlo_chess_gpu.exe 50000

# Linux
./gpu/monte_carlo_chess_gpu 50000
```

Za detaljnije uputstvo, pogledaj [gpu/README.md](gpu/README.md).

## Poređenje CPU vs GPU pristupa

| Aspekt         | CPU (Minimax)             | GPU (Monte Carlo)          |
| -------------- | ------------------------- | -------------------------- |
| Algoritam      | Deterministička pretraga  | Stohastičke simulacije     |
| Paralelizacija | OpenMP (niti)             | CUDA (hiljade threadova)   |
| Evaluacija     | Heuristike + piece-square | Win/loss statistika        |
| Brzina         | ~1M pozicija/s            | ~50K simulacija/s          |
| Tačnost        | Visoka na maloj dubini    | Raste sa brojem simulacija |

## Licence

MIT License
