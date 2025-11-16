# OPTIMIZATION_REPORT.md

## Šta je urađeno?

- Napravljen novi engine (`chess_engine_optimized.cpp`) sa svim modernim Stockfish tehnikama:
  - Paralelizacija na root nivou (OpenMP, shared alpha)
  - Veća i brža transposition table (512 MB)
  - Napredni move ordering (hash move, killer moves, PV ordering)
  - Null move pruning i late move reductions (LMR)
  - Principal Variation Search (PVS)
  - Killer move heuristika i thread-local podaci
  - Inline evaluacija i branchless kod
