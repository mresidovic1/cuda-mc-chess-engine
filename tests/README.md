# Chess Engine Benchmarks

Poređenje CPU (Negamax) i GPU (PUCT-MCTS) chess engine-a.

## Kompajliranje

```bash
cd tests
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Pokretanje Testova

```bash
cd tests/build

# Throughput test (brzina)
./benchmark_throughput --output results/throughput.csv --time 3000

# Quality test (tačnost poteza)
./benchmark_fixed_time --output results/quality.csv --times 100,500,1000 --suite bratko-kopec

# Head-to-head test (direktno poređenje)
./benchmark_matches --output results/matches.csv --games 30 --time 2000
```

## Opcije

### Throughput
- `--difficulty easy|medium|hard|all` - Težina pozicija
- `--cpu-depth N` - Dubina CPU pretrage (default: 15)
- `--gpu-sims N` - GPU simulacije (default: 5000)
- `--time N` - Vreme po poziciji (ms)

### Fixed-Time Quality
- `--times MS,MS,...` - Vremenski budžeti (ms)
- `--suite bratko-kopec|wac|performance|all` - Test suite

### Matches
- `--games N` - Broj partija (default: 30)
- `--time N` - Vreme po potezu (ms)

## Rezultati

Svi testovi generišu CSV fajlove u `results/` direktorijumu koji se mogu analizirati sa pandas/matplotlib.
