# Kako pokrenuti GPU verziju - Quick Start

## Windows (Preporučeni metod)

### Metod 1: Build.bat skripta (najlakše)

```bash
cd gpu
build.bat
monte_carlo_chess_gpu.exe 50000
```

### Metod 2: Makefile (ako imaš make instaliran)

```bash
cd gpu
make
make run
```

### Metod 3: Ručna kompilacija

```bash
cd gpu

# Kompajliranje CUDA kernela
nvcc -O3 -arch=sm_70 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o

# Kompajliranje C++ wrapper-a
g++ -O3 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"

# Kompajliranje main programa
g++ -O3 -c main_gpu.cpp -o main_gpu.o -I.. -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"

# Linkovanje
g++ -O3 main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu.exe -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64" -lcudart -lcurand

# Pokretanje
monte_carlo_chess_gpu.exe 50000
```

## Linux

### Metod 1: Build.sh skripta

```bash
cd gpu
chmod +x build.sh
./build.sh
./monte_carlo_chess_gpu 50000
```

### Metod 2: Makefile

```bash
cd gpu
make
make run
```

### Metod 3: Ručna kompilacija

```bash
cd gpu

# Kompajliranje CUDA kernela
nvcc -O3 -arch=sm_70 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o

# Kompajliranje C++ wrapper-a
g++ -O3 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I/usr/local/cuda/include

# Kompajliranje main programa
g++ -O3 -c main_gpu.cpp -o main_gpu.o -I.. -I/usr/local/cuda/include

# Linkovanje
g++ -O3 main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu -L/usr/local/cuda/lib64 -lcudart -lcurand

# Pokretanje
./monte_carlo_chess_gpu 50000
```

## Troubleshooting

### Problem: "nvcc not found"

**Rešenje:** Instaliraj CUDA Toolkit sa [NVIDIA sajta](https://developer.nvidia.com/cuda-downloads) i dodaj u PATH:

- Windows: Dodaj `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` u PATH
- Linux: `export PATH=/usr/local/cuda/bin:$PATH`

### Problem: "No CUDA-capable device found"

**Rešenje:**

1. Proveri da li imaš NVIDIA GPU: `nvidia-smi`
2. Ažuriraj GPU drajvere
3. Proveri da li GPU podržava CUDA (GeForce GTX 900 series ili noviji)

### Problem: Kompilacija pada na linkovanje

**Rešenje:** Proveri CUDA_PATH u build skripti:

- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
- Linux: `/usr/local/cuda`

### Problem: Performance je loš

**Rešenje:**

1. Povećaj broj simulacija: `monte_carlo_chess_gpu.exe 100000`
2. Prilagodi CUDA arhitekturu za tvoju GPU:
   - RTX 20xx: `-arch=sm_75`
   - RTX 30xx: `-arch=sm_80` ili `-arch=sm_86`
   - RTX 40xx: `-arch=sm_89`

## Parametri

```bash
# Format
monte_carlo_chess_gpu.exe [num_simulations] [optional_fen]

# Primeri
monte_carlo_chess_gpu.exe 10000                           # 10K simulacija na default pozicijama
monte_carlo_chess_gpu.exe 100000                          # 100K simulacija
monte_carlo_chess_gpu.exe 50000 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Custom pozicija
```

## Provera instalacije

Nakon kompilacije, pokreni:

```bash
# Windows
monte_carlo_chess_gpu.exe 1000

# Linux
./monte_carlo_chess_gpu 1000
```

Očekivani izlaz:

```
=== Monte Carlo Chess Engine - GPU Version ===
Using CUDA device: [Your GPU Name]
Compute capability: X.X
...
```

## Upoređivanje sa CPU verzijom

Za poređenje performansi, pokreni obe verzije:

**CPU verzija:**

```bash
cd ..
build/test_suite_parallel --mode=depth --depth=10 --level=easy
```

**GPU verzija:**

```bash
cd gpu
monte_carlo_chess_gpu.exe 50000
```

## Optimizacije

Za najbolje performanse:

1. Koristi što veći broj simulacija (100K+)
2. Prilagodi CUDA arhitekturu za tvoju GPU
3. Zatvori druge GPU procese (igre, rendering, itd.)
4. Na multi-GPU sistemima, možeš eksplicitno odabrati GPU (TODO: dodati parametar)
