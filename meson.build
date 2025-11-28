project(
  'chess-parallelization',
  'cpp',
  meson_version : '>= 1.3.0',
  version : '0.1',
  default_options : ['cpp_std=c++17'],
)

omp_dep = dependency('openmp')

dependencies = [
  omp_dep,
]

# --- OPTIMIZED ARGS ---
optimized_args = ['-O3', '-march=native', '-DNDEBUG', '-DUNIT_TESTS']

# --- LINKER ARGS (16MB Stack) ---
# This prevents silent exits due to recursion depth
stack_args = ['-Wl,--stack,16777216'] 

test_exe = executable(
  'chess-tests',
  ['tests.cpp', 'chess_engine_parallelized.cpp'], 
  dependencies: dependencies,
  cpp_args: optimized_args,
  link_args: stack_args  # <--- CRITICAL ADDITION
)

test('engine-tests', test_exe)