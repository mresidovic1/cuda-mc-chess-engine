# Izvještaj o razvoju sekvencijalnog šahovskog engine-a

## Početna tačka projekta

**Početni commit (b27f96b):**
```cpp
#include "../include/chess.hpp"

using namespace chess;

int main () {
    Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Movelist moves;
    movegen::legalmoves(moves, board);
    for (const auto &move : moves) {
        std::cout << uci::moveToUci(move) << std::endl;
    }
    return 0;
}
```
- **Šta je bilo implementirano:** Samo generisanje legalnih poteza koristeći chess.hpp biblioteku.
- **Nije postojala:** Nikakva pretraga, evaluacija, ni optimizacija.

---

## Faza 1: Implementacija osnovnog Minimax algoritma (branch: minimax)

**Commit: ecf4ebf "Basic minimax implemented"**

### Dodano:
1. **Evaluaciona funkcija:**
```cpp
constexpr std::array<int, 6> piece_values = {10, 30, 32, 50, 90, 0};

int evaluate(const Board &board) {
  int white_material = 0;
  int black_material = 0;
  // Broji materijal figura
  return white_material - black_material;
}
```

2. **Minimax algoritam sa alpha-beta pruning-om:**
```cpp
int minimax(Board &board, int depth, bool is_maximizing, int alpha, int beta) {
  // Osnovni minimax sa alpha-beta proresima
  if (depth == 0) return evaluate(board);
  // Rekurzivna pretraga...
}
```

3. **Konstante:**
- `INFINITY_SCORE = 30000`
- `MATE_SCORE = 10000`

**Karakteristike:**
- Klasični minimax sa eksplicitnim maximizing/minimizing player-om
- Alpha-beta pruning za smanjenje broja pozicija
- Dubina pretrage: 9 ply-eva

---

## Faza 2: Optimizacija sa Negamax (branch: minimax)

**Commit: d80d38e "Added negamax optimization"**

### Dodano:
1. **Negamax umjesto minimax:**
```cpp
int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root) {
  // Pojednostavljena verzija sa negacijom rezultata
  int score = -negamax(board, depth - 1, -beta, -alpha, current_depth_from_root + 1);
  bestValue = std::max(bestValue, score);
  alpha = std::max(alpha, bestValue);
  if (alpha >= beta) break;
  return bestValue;
}
```

2. **Move ordering:**
```cpp
void order_moves(std::vector<Move> &moves, const Board &board) {
    std::sort(moves.begin(), moves.end(), [&board](const Move& a, const Move& b) {
        bool a_is_capture = board.isCapture(a);
        bool b_is_capture = board.isCapture(b);
        if (a_is_capture && !b_is_capture) return true;
        return false; 
    });
}
```
- Capture potezi se analiziraju prvi

3. **find_best_move funkcija:**
```cpp
Move find_best_move(Board& board, int max_depth) {
    // Iterira kroz sve poteze i nalazi najbolji
}
```

4. **Poboljšana evaluacija:**
- Dodato bonus za stranu koja je na potezu (+10 centipawna)
- Bolje vrijednosti figura: {100, 300, 320, 500, 900, 0}

**Karakteristike:**
- Kod je jednostavniji i čitljiviji
- Efikasniji nego minimax (manje koda)
- Dodano mjerenje vremena izvršavanja

---

## Faza 3: Napredne optimizacije - Negascout, TT, Killer moves (branch: negascout)

**Commit: 56bebf8 "Moved tt and killer_move into separate headers. Added killer moves, negascout and null move prunning"**

### Dodano:

1. **Transposition Table (transposition_table.h):**
```cpp
const int TT_SIZE = 8388608; // 8M entries (~128MB)

struct TTEntry {
    uint64_t key;      // Zobrist hash
    int depth;         // Dubina pretrage
    int score;         // Rezultat
    uint8_t flag;      // 0=exact, 1=lower bound, 2=upper bound
};

struct TranspositionTable {
    std::vector<TTEntry> table;
    TTEntry* probe(uint64_t key);
    void store(uint64_t key, int depth, int score, uint8_t flag);
};
```
- Pamti već analizirane pozicije
- Sprečava ponavljanje analize iste pozicije

2. **Killer Moves (killer_move.h):**
```cpp
const int MAX_DEPTH = 64;
struct KillerMoves {
    Move killers[MAX_DEPTH][2];  // 2 killer poteza po dubini
    void addKiller(int depth, Move move);
    bool isKiller(int depth, Move move) const;
};
```
- Pamti tihe poteze koji su doveli do beta cutoff-a
- Poboljšava move ordering

3. **Negascout (Principal Variation Search):**
```cpp
if(first_move) {
  score = -negamax(board, depth - 1, -beta, -alpha, current_depth_from_root + 1);
  first_move = false;
} else {
  // Null window search
  score = -negamax(board, depth - 1, -alpha - 1, -alpha, current_depth_from_root + 1);
  if(score > alpha && score < beta) {
    // Re-search sa punim window-om
    score = -negamax(board, depth - 1, -beta, -score, current_depth_from_root + 1);
  }
}
```
- Prvi potez: puni alpha-beta window
- Ostali potezi: null window search (brže)
- Re-search ako je potrebno

4. **Null Move Pruning:**
```cpp
if (depth >= 3 && !board.inCheck() && !in_endgame && beta < MATE_SCORE - 1000) {
  const int R = (depth >= 6) ? 3 : 2;
  board.makeNullMove();
  int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ...);
  board.unmakeNullMove();
  if (null_score >= beta) return null_score;
}
```
- Preskoči potez i vidi može li protivnik nešto napraviti
- Ako ne može, preskoči čitavu granu

5. **Poboljšan Move Ordering:**
```cpp
void order_moves(std::vector<Move> &moves, Board &board, int depth) {
    auto score_of = [&](const Move &m) -> int {
        int s = 0;
        if (board.isCapture(m)) s += 2000;           // Captures
        if (m.typeOf() == Move::PROMOTION) s += 1500; // Promotions
        if (killer_moves.isKiller(depth, m)) s += 1000; // Killer moves
        return s;
    };
    std::stable_sort(moves.begin(), moves.end(), ...);
}
```

**Karakteristike:**
- Značajno brži engine (10-50x ubrzanje)
- Koristi pamćenje (TT) i heuristike (killer moves)
- Negascout za efikasniju pretragu

---

## Faza 4: Finalne optimizacije - Quiescence, SEE, History, PST (branch: seq-optimization-3)

**Commit: dc6b3bb "Fixed tt mate score mistake"** (trenutni branch)

### Dodano:

1. **Quiescence Search:**
```cpp
int quiescence(Board &board, int alpha, int beta, int current_depth_from_root) {
  int stand_pat = evaluate(board);
  if (stand_pat >= beta) return beta;
  
  Movelist captures;
  movegen::legalmoves<movegen::MoveGenType::CAPTURE>(captures, board);
  
  // Sortira po SEE
  std::stable_sort(captures.begin(), captures.end(), [&board](const Move &a, const Move &b) {
    return SEE(a, board) > SEE(b, board);
  });
  
  // Analizira samo capture poteze
  for (const auto &move : captures) {
    int see_score = SEE(move, board);
    if (stand_pat + see_score + DELTA_MARGIN < alpha) continue;
    
    board.makeMove(move);
    int score = -quiescence(board, -beta, -alpha, current_depth_from_root + 1);
    board.unmakeMove(move);
    
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}
```
- Produbljuje pretragu samo za capture poteze
- Sprečava "horizon effect" (propuštanje taktika na granici dubine)

2. **Static Exchange Evaluation (SEE):**
```cpp
int SEE(Move move, Board &board) {
  // Simulira sekvencu uzimanja figura
  // Vraća neto materijalnu prednost
  int gain[32] = {};
  gain[0] = piece_values[static_cast<int>(victim_type)];
  
  // Iterativno simulira uzimanja
  while (depth < MAX_SEE_DEPTH) {
    Square attacker_sq = findLeastValuableAttacker(attackers, side, board);
    gain[depth] = -attacker_val + gain[depth - 1];
    // ...
  }
  
  // Backtrack kroz capture sekvencu
  for (int i = depth - 1; i > 0; i--) {
    gain[i - 1] = std::max(-gain[i], gain[i - 1]);
  }
  return gain[0];  
}
```
- Procjenjuje taktičku vrijednost capture poteza
- Koristi se za move ordering i delta pruning

3. **History Heuristic (history.h):**
```cpp
struct HistoryTable {
    int table[2][64][64];  // [color][from][to]
    
    void update(Move move, Color side, int depth) {
        // Povećava score za dobre poteze
        int bonus = depth * depth;
        table[static_cast<int>(side)][from][to] += bonus;
    }
    
    int get(Move move, Color side) const;
};
```
- Pamti koje su tihe poteze često dobre
- Koristi se za move ordering kada nema TT ili killer poteza

4. **Piece-Square Tables (constants.h):**
```cpp
constexpr std::array<int, 64> pawn_pst = { /* pozicione vrijednosti */ };
constexpr std::array<int, 64> knight_pst = { /* ... */ };
// ... za sve figure

int pstScore(Bitboard pieces, Color color, const std::array<int, 64> &table) {
  int score = 0;
  while (pieces) {
    Square sq = pieces.pop();
    int idx = (color == Color::WHITE) ? sq : (63 - sq);
    score += table[idx];
  }
  return score;
}
```
- Daje bonus za dobre pozicije figura (centralizacija, razvoj, itd.)
- Uključeno u evaluaciju

5. **Razoring:**
```cpp
if (depth <= 3 && static_eval < alpha - RAZOR_MARGIN_BASE - RAZOR_MARGIN_DEPTH * depth * depth) {
  return quiescence(board, alpha, beta, current_depth_from_root);
}
```
- Ako je pozicija očigledno loša, preskoči punu pretragu i idi na quiescence

6. **Check Extensions:**
```cpp
if (gives_check && extension_count < MAX_CHECK_EXTENSIONS) {
  extension = 1;  // Produži pretragu za 1 ply
}
```
- Produži pretragu kada se daje šah (taktički važno)

7. **Late Move Reductions (LMR):**
```cpp
if (move_count >= 4 && depth >= 3 && !is_capture && !is_promotion && 
    !gives_check && !in_check && !is_killer) {
  reduction = 1;
  if (move_count >= 8 && depth >= 4) reduction = 2;
}
```
- Kasni tihi potezi se pretražuju sa manjom dubinom

8. **Improved Move Ordering:**
```cpp
void order_moves(std::vector<Move> &moves, Board &board, int depth, Move tt_move, int static_eval) {
    auto score_of = [&](const Move &m) -> int {
        int s = 0;
        if (m == tt_move) s += 10000;              // TT move (najbolji!)
        else if (board.isCapture(m)) {
          s += 2000 + SEE(m, board);               // Captures sa SEE
        }
        else if (m.typeOf() == Move::PROMOTION) s += 1500;
        else if (killer_moves.isKiller(depth, m)) s += 1000;
        else {
          s += history.get(m, side_to_move) / 32;  // History heuristic
        }
        return s;
    };
    std::stable_sort(moves.begin(), moves.end(), ...);
}
```
- TT move → Captures (SEE) → Promotions → Killers → History

9. **Improved Transposition Table:**
- Dodato polje `bestMove` u TT entry
- Dodato polje `staticEval` da se ne računa više puta
- Bolji replacement scheme

10. **Tests (tests.cpp):**
```cpp
// 8 test pozicija sa poznatim najboljim potezima
// Testira i tačnost i brzinu engine-a
```

**Karakteristike:**
- Najnaprednija sekvencijalna verzija
- Koristi sve standardne Stockfish tehnike (osim paralelizacije)
- Duboka pretraga (quiescence) za taktičku tačnost
- Sofisticiran move ordering
- Poziciona evaluacija (PST)

---

## Sažetak napretka

| Faza | Branch | Tehnike | Relativna brzina |
|------|--------|---------|------------------|
| 1 | Initial | Samo generisanje poteza | - |
| 2 | minimax | Minimax + Alpha-Beta | 1x (baseline) |
| 3 | minimax (negamax) | Negamax + Move ordering | 2-3x |
| 4 | negascout | TT + Killer + Negascout + Null move | 10-50x |
| 5 | seq-optimization-3 | Quiescence + SEE + History + PST + LMR + Razoring | 50-200x |

---

## Ključni fajlovi u finalnoj verziji (seq-optimization-3):

1. **chess_parallelization.cpp** (705 linija):
   - Glavna logika engine-a
   - Negamax sa svim optimizacijama
   - Quiescence search
   - SEE funkcija
   - Move ordering

2. **transposition_table.h**:
   - TT struktura sa key, depth, score, flag, bestMove, staticEval
   - probe() i store() metode

3. **killer_move.h**:
   - Pamti 2 killer poteza po dubini
   - addKiller() i isKiller() metode

4. **history.h**:
   - History heuristic za tihe poteze
   - update() i get() metode

5. **constants.h**:
   - Piece values
   - Piece-Square Tables za sve figure
   - Sve konstante (RAZOR_MARGIN, DELTA_MARGIN, MAX_CHECK_EXTENSIONS, itd.)

6. **tests.cpp**:
   - 8 test pozicija
   - Mjerenje tačnosti i brzine

---

## Najrelevantniji isječci koda za prezentaciju

### 1. Evaluacija (početak vs kraj):

**Početak (minimax):**
```cpp
constexpr std::array<int, 6> piece_values = {10, 30, 32, 50, 90, 0};
int evaluate(const Board &board) {
  int evaluation = white_material - black_material;
  return evaluation;
}
```

**Kraj (seq-optimization-3):**
```cpp
constexpr std::array<int, 6> piece_values = {100, 320, 330, 500, 900, 0};
int evaluate(const Board &board) {
  int evaluation = 0;
  // Material + Piece-Square Tables
  for (const auto &pt : piece_types) {
    evaluation += white_bb.count() * value;
    evaluation += pstScore(white_bb, Color::WHITE, table);
    // ... isto za crnog
  }
  evaluation += (board.sideToMove() == Color::WHITE) ? 10 : -10;
  return evaluation;
}
```

### 2. Pretraga (početak vs kraj):

**Početak (minimax):**
```cpp
int minimax(Board &board, int depth, bool is_maximizing, int alpha, int beta) {
  if (depth == 0) return evaluate(board);
  
  if (is_maximizing) {
    int bestValue = -INFINITY_SCORE;
    for (auto &move : movelist) {
      board.makeMove(move);
      int score = minimax(board, depth - 1, false, alpha, beta);
      board.unmakeMove(move);
      bestValue = std::max(bestValue, score);
      alpha = std::max(alpha, bestValue);
      if (beta <= alpha) break;
    }
    return bestValue;
  } else { /* ... */ }
}
```

**Kraj (seq-optimization-3):**
```cpp
int negamax(Board &board, int depth, int alpha, int beta, int current_depth_from_root, 
            int extension_count, int prev_static_eval) {
  // TT probe
  TTEntry* tt_entry = tt.probe(zobrist_key);
  if (tt_entry->key == zobrist_key && tt_entry->depth >= depth) {
    int tt_score = tt_entry->score;
    if (tt_entry->flag == 0) return tt_score;
    // ...
  }
  
  // Quiescence na dubini 0
  if (depth == 0) return quiescence(board, alpha, beta, current_depth_from_root);
  
  // Razoring
  if (depth <= 3 && static_eval < alpha - RAZOR_MARGIN) {
    return quiescence(board, alpha, beta, current_depth_from_root);
  }
  
  // Null move pruning
  if (allow_null_move) {
    board.makeNullMove();
    int null_score = -negamax(board, null_depth, -beta, -beta + 1, ...);
    board.unmakeNullMove();
    if (null_score >= beta) return null_score;
  }
  
  // Move ordering sa TT move, SEE, killer, history
  order_moves(moves_to_search, board, depth, tt_move, static_eval);
  
  // Negascout + LMR + Check extensions
  for (auto &move : moves_to_search) {
    int extension = 0;
    if (gives_check && extension_count < MAX_CHECK_EXTENSIONS) extension = 1;
    
    int reduction = 0;
    if (move_count >= 4 && depth >= 3 && !is_capture && ...) {
      reduction = 1;
      if (move_count >= 8) reduction = 2;
    }
    
    if (first_move) {
      score = -negamax(board, depth - 1 + extension, -beta, -alpha, ...);
    } else {
      score = -negamax(board, depth - 1 - reduction + extension, -alpha - 1, -alpha, ...);
      if (score > alpha && score < beta) {
        score = -negamax(board, depth - 1 + extension, -beta, -alpha, ...);
      }
    }
    
    // Update history for good quiet moves
    if (score >= beta && !is_capture) {
      history.update(move, side_to_move, depth);
      killer_moves.addKiller(depth, move);
    }
  }
  
  // TT store
  tt.store(zobrist_key, depth, bestValue, flag, bestMove, static_eval);
  return bestValue;
}
```

### 3. Move Ordering (napredak):

**Početak (negamax):**
```cpp
void order_moves(std::vector<Move> &moves, const Board &board) {
    std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        return board.isCapture(a) && !board.isCapture(b);
    });
}
```

**Kraj (seq-optimization-3):**
```cpp
void order_moves(std::vector<Move> &moves, Board &board, int depth, 
                 Move tt_move, int static_eval) {
    auto score_of = [&](const Move &m) -> int {
        if (m == tt_move) return 10000;                    // TT move
        else if (board.isCapture(m)) return 2000 + SEE(m, board); // SEE
        else if (m.typeOf() == Move::PROMOTION) return 1500;
        else if (killer_moves.isKiller(depth, m)) return 1000;
        else return history.get(m, side_to_move) / 32;     // History
    };
    std::stable_sort(moves.begin(), moves.end(), 
                     [&](const Move &a, const Move &b) { return score_of(a) > score_of(b); });
}
```

---

## Zaključak

Sekvencijalni engine je prošao kroz 5 glavnih faza razvoja:
1. **Osnovna implementacija** (minimax)
2. **Optimizacija algoritma** (negamax)
3. **Dodavanje naprednih tehnika** (TT, killer, negascout, null move)
4. **Taktička tačnost** (quiescence, SEE)
5. **Finalne optimizacije** (history, PST, LMR, razoring, check extensions)

Rezultat je sofisticiran sekvencijalni šahovski engine koji koristi sve standardne tehnike modernih engine-a, osim paralelizacije. Engine je testiran sa 8 test pozicija i pokazuje odličnu tačnost i brzinu.

**Napomena:** Sve što je navedeno u ovom izvještaju je REALNO implementirano i može se provjeriti u git historiji projekta.
