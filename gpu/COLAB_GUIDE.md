# Vodič za pokretanje Monte Carlo Chess Engine na Google Colabu

## 🎯 BrziStart

### Korak 1: Otvori Google Colab
1. Idi na https://colab.research.google.com/
2. Klikni **File → New notebook**

### Korak 2: Aktiviraj GPU
1. **Runtime → Change runtime type**
2. **Hardware accelerator → T4 GPU** (ili L4/A100 ako imaš pristup)
3. Klikni **Save**

### Korak 3: Proveri GPU (u prvoj ćeliji)
```python
!nvidia-smi
```
Trebalo bi da vidiš NVIDIA Tesla T4 ili sličnu karticu.

### Korak 4: Kloniraj projekat (u sledećoj ćeliji)
```bash
!git clone https://github.com/<tvoj-username>/chess-parallelization.git
%cd chess-parallelization/gpu
```
**Zameni `<tvoj-username>` sa tvojim GitHub username-om!**

### Korak 5: Kompajliraj projekat (nova ćelija)
```bash
# Kompilacija CUDA kernela
!nvcc -O3 -arch=sm_75 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o

# Kompilacija C++ wrapper-a
!g++ -O3 -std=c++17 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I/usr/local/cuda/include

# Kompilacija main programa
!g++ -O3 -std=c++17 -c main_gpu.cpp -o main_gpu.o -I.. -I/usr/local/cuda/include

# Linkovanje
!g++ -O3 main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu \
    -L/usr/local/cuda/lib64 -lcudart -lcurand
```

**Napomena o arhitekturi:**
- Tesla T4 → `sm_75` ✅ (većina Colab-ova)
- V100 → `sm_70`
- A100 → `sm_80`

Proveri sa `!nvidia-smi` koju GPU imaš i prilagodi `-arch=sm_XX` ako je potrebno.

### Korak 6: Pokreni engine (nova ćelija)

#### Pronađi najbolji potez sa 5000 simulacija po potezu (default):
```bash
!./monte_carlo_chess_gpu
```

#### Sa većim brojem simulacija (bolja preciznost):
```bash
!./monte_carlo_chess_gpu 10000
```

#### Za custom poziciju:
```bash
!./monte_carlo_chess_gpu 5000 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
```

## 📊 Očekivani izlaz

```
=== Monte Carlo Chess Engine - GPU Version ===
Using CUDA device: Tesla T4
Compute capability: 7.5

=== Finding Best Move ===
Position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

=== Analyzing 20 possible moves ===
Simulations per move: 5000
Side to move: White

Move 1/20: a2a3 - Score: 48.2% (W:1205 D:1590 B:1205)
Move 2/20: a2a4 - Score: 49.8% (W:1245 D:1510 B:1245)
Move 3/20: b2b3 - Score: 47.5% (W:1188 D:1624 B:1188)
...
Move 17/20: e2e4 - Score: 52.1% (W:1305 D:1390 B:1305)
...

=== BEST MOVE ===
Move: e2 -> e4
Score: 52.1%
Total time: 15234 ms
=================
```

## 🔧 Troubleshooting

### Problem: "nvcc not found"
Colab već ima CUDA Toolkit instaliran. Ako ovo dobiješ, restart runtime-a:
- **Runtime → Restart runtime**

### Problem: "No CUDA-capable device found"
Nisi aktivirao GPU. Vrati se na **Korak 2** i aktiviraj T4 GPU.

### Problem: Kompilacija pada
Proveri da si u pravom direktorijumu:
```bash
%cd /content/chess-parallelization/gpu
!ls -la
```

### Problem: Sve pobede/remiji/porazi su 0
To je stara verzija koda. Pull-uj najnovije izmene:
```bash
%cd /content/chess-parallelization
!git pull origin main
%cd gpu
```
Zatim ponovo kompajliraj.

## ⚡ Optimizacija performansi

### Više simulacija = Bolja preciznost (ali sporije)
- **5000 simulacija**: Brzo, okej preciznost (~15-30 sec za poziciju)
- **10000 simulacija**: Srednjo, dobra preciznost (~30-60 sec)
- **50000 simulacija**: Sporo, odlična preciznost (nekoliko minuta)

### Pametno testiranje
```bash
# Prvo testiraj sa manje simulacija za brzi pregled
!./monte_carlo_chess_gpu 1000

# Onda sa više za tačniji rezultat
!./monte_carlo_chess_gpu 10000
```

## 📝 Napredne opcije

### Kreiranje bash skripte za lakše pokretanje
```bash
%%writefile run_gpu.sh
#!/bin/bash
nvcc -O3 -arch=sm_75 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o
g++ -O3 -std=c++17 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I/usr/local/cuda/include
g++ -O3 -std=c++17 -c main_gpu.cpp -o main_gpu.o -I.. -I/usr/local/cuda/include
g++ -O3 main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu \
    -L/usr/local/cuda/lib64 -lcudart -lcurand
./monte_carlo_chess_gpu $@
```

Zatim:
```bash
!chmod +x run_gpu.sh
!./run_gpu.sh 10000
```

### Testiranje različitih pozicija
```bash
# Testovi različitih pozicija
!./monte_carlo_chess_gpu 5000 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Početna pozicija
!./monte_carlo_chess_gpu 5000 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"  # Italian Game
!./monte_carlo_chess_gpu 5000 "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"  # Kralj i pešak endgame
```

## 🌐 Alternative za Colab

Ako imaš problema sa Colab-om ili trebaš više GPU vremena:

1. **Kaggle Notebooks** (30h GPU sedmično besplatno)
   - https://www.kaggle.com/code
   - Slično kao Colab, takođe ima T4 GPU

2. **Google Cloud Free Trial** ($300 kredita)
   - https://cloud.google.com/free

3. **AWS SageMaker Studio Lab** (besplatno sa T4)
   - https://studiolab.sagemaker.aws/

## 🎓 Kako radi Monte Carlo metoda?

1. **Za svaki legalan potez** iz trenutne pozicije:
   - Napravi novu poziciju nakon tog poteza
   - Iz te pozicije pokreni hiljade nasumičnih partija do kraja
   - Zabelež koliko partija je bilo pobeda/remija/poraza

2. **Najbolji potez** je onaj koji ima najviše pobeda (ili najbolji "score")

3. **Score formula**: 
   - Za belog: `score = (white_wins + 0.5 * draws) / total_simulations`
   - Za crnog: `score = (black_wins + 0.5 * draws) / total_simulations`

## 📈 Performanse

Na Tesla T4:
- ~200,000 - 500,000 simulacija/sekund
- ~20 legalnih poteza po poziciji (prosek)
- ~15-60 sekundi za pronalaženje najboljeg poteza (zavisi od broja simulacija)

## 🐛 Debug mod

Ako želiš da vidiš detaljnije šta se dešava:
```bash
# Kompajliraj sa debug flagovima
!nvcc -G -g -arch=sm_75 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o
!g++ -g -std=c++17 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I/usr/local/cuda/include
!g++ -g -std=c++17 -c main_gpu.cpp -o main_gpu.o -I.. -I/usr/local/cuda/include
!g++ -g main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu \
    -L/usr/local/cuda/lib64 -lcudart -lcurand

!./monte_carlo_chess_gpu 1000
```

---

**Sreća u testiranju! 🚀**
