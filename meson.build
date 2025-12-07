project(
  'chess-parallelization',
  'cpp',
  meson_version : '>= 1.3.0',
  version : '0.1',
  default_options : ['cpp_std=c++17', 'buildtype=debugoptimized'],
)

omp_dep = dependency('openmp')

dependencies = [omp_dep]

inc_dirs = include_directories('include')

optimized_args = ['-O3', '-march=native', '-DNDEBUG', '-DUNIT_TESTS']
stack_args = ['-Wl,-stack_size,0x1000000']

test_exe = executable(
  'chess-tests',
  ['tests/tests.cpp', 'src/chess_engine_parallelized.cpp'],
  dependencies: dependencies,
  include_directories: inc_dirs,
  cpp_args: optimized_args,
  link_args: stack_args
)

test('engine-tests', test_exe)

test_suite_exe = executable(
  'test_suite',
  ['tests/test_suite.cpp', 'src/chess_engine_parallelized.cpp'],
  dependencies: dependencies,
  include_directories: inc_dirs,
  cpp_args: optimized_args,
  link_args: stack_args
)

test('suite-depth-easy', test_suite_exe,
     args: ['--mode=depth', '--depth=20', '--level=easy'],
     timeout: 120)

test('suite-depth-medium', test_suite_exe,
     args: ['--mode=depth', '--depth=20', '--level=medium'],
     timeout: 180)

test('suite-depth-hard', test_suite_exe,
     args: ['--mode=depth', '--depth=20', '--level=hard'],
     timeout: 300)

test('suite-time-easy', test_suite_exe,
     args: ['--mode=time', '--time=30', '--level=easy'],
     timeout: 300)

test('suite-time-medium', test_suite_exe,
     args: ['--mode=time', '--time=30', '--level=medium'],
     timeout: 300)

test('suite-time-hard', test_suite_exe,
     args: ['--mode=time', '--time=30', '--level=hard'],
     timeout: 400)
