# Pregled Testova i Poređenja

## 🎯 Šta Se Testira?

Ovaj direktorijum sadrži 4 tipa benchmarka koji porede **CPU** i **GPU** chess engine.

---

## 1. 📊 Throughput Test

**Šta mjeri**: Koliko brzo engine radi (operacija po sekundi)

**Fajl**: `benchmark_throughput.cpp`

**Metrike**:

- **CPU**: Nodes/sec (koliko čvorova pretraži u sekundi)
- **GPU**: Playouts/sec (koliko simulacija izvrši u sekundi)

**Poređenje**:

- Testira različite pozicije (easy/medium/hard)
- Mjeri vrijeme i broj operacija
- Računa throughput za svaki engine

**Rezultat**: CSV sa kolonama `engine,position_name,fen,time_ms,operations,throughput,depth`

**Što dobijamo**: Odgovor na pitanje "Ko je brži?"

---

## 2. 🎯 Fixed-Time Quality Test

**Šta meri**: Koliko kvalitetne poteze prave enginei pod vremenskim ograničenjem

**Fajl**: `benchmark_fixed_time.cpp`

**Metrike**:

- Koji potez odabere engine za različite vremenske budžete (100ms, 500ms, 1s, itd.)
- Kolika je evaluacija pozicije
- Koliko duboko je pretraživao
- Broj nodes/playouts

**Poređenje**:

- Obe engine testira sa ISTIM vremenom
- Uporedi koji potez svaki izabere
- Može se uporediti sa "ground truth" pozicijama (Bratko-Kopec, WAC)

**Rezultat**: CSV sa kolonama `engine,position_name,fen,time_budget_ms,actual_time_ms,move_uci,eval_cp,depth,nodes`

**Što dobijamo**: Odgovor na pitanje "Ko igra bolje u istom vremenu?"

---

## 3. 🤝 Head-to-Head Matches

**Šta meri**: Direktno poređenje - ko pobedi u pravoj šahovskoj partiji

**Fajl**: `benchmark_matches.cpp`

**Metrike**:

- Broj pobeda/nerijesenih/poraza
- Elo rating razlika
- Prosečna dužina partije
- Vrsta pozicija gdje ko dominira

**Poređenje**:

- CPU vs GPU u pravim partijama
- Alternira boje (CPU beli/crni)
- Fiksirano vrijeme po potezu

**Rezultat**: CSV sa kolonama `game_id,white_engine,black_engine,result,moves,time_control,final_fen`

**Što dobijamo**: Odgovor na pitanje "Ko ZAPRAVO pobedi?"

---

## 4. 📈 Stockfish Agreement (Napredni Test)

**Šta meri**: Koliko se slažu sa Stockfish-om (2800+ Elo engine)

**Fajl**: `benchmark_stockfish.cpp`

**Metrike**:

- % poteza koji se poklapaju sa Stockfish najboljim potezom
- Razlika u evaluaciji pozicije
- Korelacija sa Stockfish ocenama

**Poređenje**:

- Analizira Stockfish poziciju na depth 15+
- Obe engine testira istu poziciju
- Uporedi koliko su blizu "objektivno najboljem"

**Rezultat**: CSV sa kolonama `engine,position_name,stockfish_move,engine_move,match,stockfish_eval,engine_eval,eval_diff`

**Što dobijamo**: Odgovor na pitanje "Ko je bliži optimalnoj igri?"

---

## 📊 Kompletan Pregled Poređenja

| Test           | CPU Prednost      | GPU Prednost   | Ključna Metrika |
| -------------- | ----------------- | -------------- | --------------- |
| **Throughput** | Dubina pretrage   | Brzina (600x)  | Ops/sec         |
| **Quality**    | Taktičke pozicije | Strategija (?) | Accuracy        |
| **Matches**    | ?                 | ?              | Win rate        |
| **Stockfish**  | Preciznost        | ?              | Agreement %     |

---

## 🔬 Zašto 4 Tipa Testova?

### Throughput ≠ Kvalitet

- GPU može biti 600x brži ALI igrati lošije
- Brzina je bitna samo ako se pretvara u dobre poteze

### Quality ≠ Pobede

- Engine može igrati "dobre" poteze na poznatim pozicijama
- Ali gubiti u pravim partijama zbog stila igre

### Matches = Ultimativna Istina

- Head-to-head pokazuje ko ZAPRAVO pobedi
- Ali ne objašnjava ZAŠTO

### Stockfish = Objektivna Procena

- Pokazuje koliko su blizu "savršenoj" igri
- Ali Stockfish igra drugačije od oba enginea

---

## 🚀 Kako Pokrenuti Sve Testove

```bash
cd tests/build

# 1. Throughput (5 min)
./benchmark_throughput --output results/throughput.csv

# 2. Quality (10 min)
./benchmark_fixed_time --output results/quality.csv

# 3. Matches (20 min)
./benchmark_matches --output results/matches.csv

# 4. Stockfish (15 min - ako imaš Stockfish)
./benchmark_stockfish --output results/stockfish.csv
```

---

## ✅ Konačna Svrha

**Cilj**: Razumjeti KADA koristiti koji engine (taktika vs strategija, blitz vs long games, itd.)
