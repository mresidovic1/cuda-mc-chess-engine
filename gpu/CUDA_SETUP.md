# CUDA Setup i Provera Instalacije

## Provera CUDA instalacije

### 1. Proveri da li imaš NVIDIA GPU

**Windows:**

```powershell
nvidia-smi
```

**Linux:**

```bash
nvidia-smi
```

Očekivani izlaz:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.6   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    15W / 350W |    512MiB / 12288MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
```

### 2. Proveri CUDA Toolkit instalaciju

**Windows:**

```powershell
nvcc --version
```

**Linux:**

```bash
nvcc --version
```

Očekivani izlaz:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.6, V12.6.xxx
Build cuda_12.6.r12.6/compiler.xxxxx_0
```

Ako `nvcc` nije pronađen:

**Windows:**

- Proveri instalaciju: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\`
- Dodaj u PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`

**Linux:**

- Proveri instalaciju: `/usr/local/cuda`
- Dodaj u PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- Dodaj u LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

### 3. Proveri CUDA kompajler

Kreiraj test fajl `test_cuda.cu`:

```cpp
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Kompajliraj i pokreni:

```bash
nvcc test_cuda.cu -o test_cuda
./test_cuda
```

Očekivani izlaz:

```
Hello from GPU thread 0
Hello from GPU thread 1
Hello from GPU thread 2
Hello from GPU thread 3
Hello from GPU thread 4
```

## Instalacija CUDA Toolkit

### Windows

1. Preuzmi sa: https://developer.nvidia.com/cuda-downloads
2. Odaberi:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: Windows 10/11
   - Installer Type: exe (local)
3. Pokreni installer i prati uputstva
4. Restartuj računar
5. Proveri instalaciju: `nvcc --version`

### Linux (Ubuntu/Debian)

```bash
# Preuzmi CUDA Toolkit (12.6 primer)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_535.104.05_linux.run
sudo sh cuda_12.6.0_535.104.05_linux.run

# Dodaj u PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Proveri
nvcc --version
```

## Pronalaženje CUDA Arhitekture tvoje GPU

### Metod 1: nvidia-smi

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Metod 2: Proveri online

1. Pronađi model tvoje GPU: `nvidia-smi`
2. Proveri na: https://developer.nvidia.com/cuda-gpus
3. Pronađi "Compute Capability"

### Mapiranje Compute Capability na CUDA Arhitekturu

| Compute Capability | CUDA Arch | Primer GPU-a               |
| ------------------ | --------- | -------------------------- |
| 7.0                | sm_70     | Tesla V100                 |
| 7.5                | sm_75     | RTX 2060, 2070, 2080       |
| 8.0                | sm_80     | A100                       |
| 8.6                | sm_86     | RTX 3060, 3070, 3080, 3090 |
| 8.9                | sm_89     | RTX 4060, 4070, 4080, 4090 |
| 9.0                | sm_90     | H100                       |

**Kako prilagoditi u build skriptama:**

**build.bat (Windows):**

```batch
REM Promeni sm_70 u odgovarajuću arhitekturu
nvcc -O3 -arch=sm_86 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o
```

**build.sh (Linux):**

```bash
# Promeni sm_70 u odgovarajuću arhitekturu
CUDA_ARCH="sm_86"
nvcc -O3 -arch=$CUDA_ARCH -c monte_carlo_kernel.cu -o monte_carlo_kernel.o
```

**Makefile:**

```makefile
# Promeni default vrednost
make CUDA_ARCH=sm_86
```

## Debugging CUDA koda

### Compile sa debug info

```bash
nvcc -g -G -O0 monte_carlo_kernel.cu -c -o monte_carlo_kernel.o
```

### Provera CUDA errors

Dodaj ovo u kod:

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Koristi ovako:
CUDA_CHECK(cudaMalloc(&d_data, size));
```

### Profajliranje

**NVIDIA Nsight Systems:**

```bash
nsys profile ./monte_carlo_chess_gpu 50000
```

**NVIDIA Nsight Compute:**

```bash
ncu --set full ./monte_carlo_chess_gpu 50000
```

## Česte greške i rešenja

### Error: "CUDA_ERROR_NO_DEVICE"

**Rešenje:** Nemaš NVIDIA GPU ili drajver nije instaliran

```bash
# Proveri
nvidia-smi

# Instaliraj najnovije drajvere sa:
# https://www.nvidia.com/Download/index.aspx
```

### Error: "undefined reference to `cudaMalloc'"

**Rešenje:** Nisi linkirao CUDA biblioteke

```bash
# Dodaj -lcudart
g++ ... -L/usr/local/cuda/lib64 -lcudart
```

### Error: "architecture compute_XX not supported"

**Rešenje:** Tvoja GPU je prestara ili previše nova za CUDA Toolkit verziju

```bash
# Za stare GPU (Compute Capability < 7.0), koristi stariji CUDA Toolkit
# Za nove GPU, ažuriraj CUDA Toolkit
```

### Error: Compilation fails with "host compiler too new"

**Rešenje:** CUDA verzija ne podržava tvoju GCC verziju

```bash
# Proveri kompatibilnost:
# CUDA 12.6: GCC 9.x - 12.x
# CUDA 12.0: GCC 9.x - 11.x
# CUDA 11.8: GCC 8.x - 11.x

# Downgrade-uj GCC ili upgrade-uj CUDA
```

### Performance je loša

**Rešenje:**

1. Proveri da li GPU nije opterećena drugim procesima:

   ```bash
   nvidia-smi
   # Proveri GPU-Util %
   ```

2. Povećaj broj simulacija:

   ```bash
   ./monte_carlo_chess_gpu 100000  # Umesto 10000
   ```

3. Prilagodi CUDA arhitekturu:

   ```bash
   # Build sa tvojom specifičnom arhitekturom
   make clean
   make CUDA_ARCH=sm_86
   ```

4. Proveri memory bandwidth:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```

## Optimizacija build procesa

### Korišćenje ccache za brže recompile

```bash
# Instaliraj ccache
sudo apt install ccache  # Linux
# ili
choco install ccache     # Windows (sa Chocolatey)

# Koristi sa nvcc
ccache nvcc -O3 -arch=sm_70 -c monte_carlo_kernel.cu
```

### Paralelno kompajliranje

```bash
# Sa make
make -j8  # Koristi 8 paralelnih procesa
```

## Testiranje instalacije Monte Carlo Chess GPU

Nakon što si siguran da sve radi, testiraj projekat:

```bash
cd gpu

# Build
./build.sh  # ili build.bat na Windows

# Quick test (1000 simulacija)
./monte_carlo_chess_gpu 1000

# Benchmark (100K simulacija)
./monte_carlo_chess_gpu 100000
```

Očekivano vreme izvršavanja:

- GTX 1660: ~3-4 sekunde za 100K simulacija
- RTX 3080: ~1-2 sekunde za 100K simulacija
- RTX 4090: ~0.5-1 sekunda za 100K simulacija

Ako traje značajno duže, proveri:

1. Da li koristiš release build (-O3)?
2. Da li je GPU arhitektura pravilno podešena?
3. Da li GPU nije opterećena drugim procesima?

## Dodatni resursi

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
