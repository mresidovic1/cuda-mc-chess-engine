# CPU Chess Engine

CPU engine using a parallelized minimax/negamax search with OpenMP. The CPU build produces test executables that run the engine against tactical suites.

## Prerequisites

- C++17 compiler with OpenMP support
- Meson 1.3+

## Build

```bash
cd cpu
meson setup build
meson compile -C build
```

## Run

```bash
cd cpu
build\chess-tests
build\test_suite_parallel --mode=depth --depth=10 --level=easy
build\test_suite_sequential --mode=depth --depth=10 --level=easy
```
