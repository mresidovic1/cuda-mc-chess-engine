project(
  'chess-parallelization',
  'cpp',
  meson_version : '>= 1.3.0',
  version : '0.1',
  default_options : ['cpp_std=c++17'],
)

dependencies = [
]

exe = executable(
  'chess-parallelization',
  'chess_parallelization.cpp',
  'tests.cpp',
  dependencies : dependencies,
  install : true,
  cpp_args: ['-O3', '-march=native', '-DNDEBUG']
)

test('basic', exe)
