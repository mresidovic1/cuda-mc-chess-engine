project(
  'chess-parallelization',
  'cpp',
  meson_version : '>= 1.3.0',
  version : '0.1',
  default_options : ['cpp_std=c++17'],
)

dependencies = [
  dependency('openmp', required: true),
]

exe = executable(
  'chess-parallelization',
  'chess_engine_optimized.cpp',
  'tests.cpp',
  dependencies : dependencies,
  install : true,
  cpp_args: ['-O3', '-march=native', '-DNDEBUG', '-fopenmp'],
  link_args: ['-fopenmp']
)

test('basic', exe)
