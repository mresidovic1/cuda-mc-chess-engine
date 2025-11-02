# Uputstvo za instalaciju

Kompletan chess library nalazi se u `/include/chess.hpp`. 

Za korištenje u projektu:
```cpp
#include '../include/chess.hpp'
```

Ovaj library koristi [Meson build sistem](https://mesonbuild.com/Quick-guide.html) za kompajliranje.

### Koraci za instalaciju:
```bash
   mkdir build
   cd build
```
```bash
   meson setup build
```
```bash
   meson compile
```
```bash
   cd ..
   build/chess-parallelization
```
