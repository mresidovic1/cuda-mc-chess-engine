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
  dependencies : dependencies,
  install : true,
)

test('basic', exe)
