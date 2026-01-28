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

optimized_args = ['/O2', '-DNDEBUG', '-DUNIT_TESTS']
stack_args = ['/STACK:16777216']

test_exe = executable(
  'chess-tests',
  ['tests/tests.cpp', 'src/chess_engine_parallelized.cpp'],
  dependencies: dependencies,
  include_directories: inc_dirs,
  cpp_args: optimized_args,
  link_args: stack_args
)

test('engine-tests', test_exe)

test_suite_parallel = executable(
  'test_suite_parallel',
  ['tests/test_suite.cpp', 'src/chess_engine_parallelized.cpp'],
  dependencies: dependencies,
  include_directories: inc_dirs,
  cpp_args: optimized_args,
  link_args: stack_args
)

test_suite_sequential = executable(
  'test_suite_sequential',
  ['tests/test_suite.cpp', 'src/chess_sequential.cpp'],
  dependencies: dependencies,
  include_directories: inc_dirs,
  cpp_args: optimized_args,
  link_args: stack_args
)

test('parallel-easy', test_suite_parallel,
     args: ['--mode=depth', '--depth=20', '--level=easy'],
     timeout: 120)

test('parallel-medium', test_suite_parallel,
     args: ['--mode=depth', '--depth=20', '--level=medium'],
     timeout: 180)

test('parallel-hard', test_suite_parallel,
     args: ['--mode=depth', '--depth=20', '--level=hard'],
     timeout: 400)

test('sequential-easy', test_suite_sequential,
     args: ['--mode=depth', '--depth=20', '--level=easy'],
     timeout: 120)

test('sequential-medium', test_suite_sequential,
     args: ['--mode=depth', '--depth=20', '--level=medium'],
     timeout: 180)

test('sequential-hard', test_suite_sequential,
     args: ['--mode=depth', '--depth=20', '--level=hard'],
     timeout: 400)
