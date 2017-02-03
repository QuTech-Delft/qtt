"""Unified qcodes test runners."""

import sys


def _test_core(test_pattern='test*.py', **kwargs):
    import unittest

    import qtt.tests as qctest
    import qtt

    suite = unittest.defaultTestLoader.discover(
        qctest.__path__[0], top_level_dir=qtt.__path__[0],
        pattern=test_pattern)
    if suite.countTestCases() == 0:
        print('found no tests')
        print('dirs: %s %s' % (qctest.__path__[0], qtt.__path__[0]))
        sys.exit(1)
    print('testing %d cases' % suite.countTestCases())

    result = unittest.TextTestRunner(**kwargs).run(suite)
    return result.wasSuccessful()


def test_part(name):
    """
    Run part of the qcodes core test suite.

    Args:
        name (str): a name within the qcodes.tests directory. May be:
            - a module ('test_loop')
            - a TestCase ('test_loop.TestLoop')
            - a test method ('test_loop.TestLoop.test_nesting')
    """
    import unittest
    fullname = 'qcodes.tests.' + name
    suite = unittest.defaultTestLoader.loadTestsFromName(fullname)
    return unittest.TextTestRunner().run(suite).wasSuccessful()

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    try:
        import coverage
        _ = coverage.Coverage
        coverage_missing = False
    except (ImportError, AttributeError):
        coverage_missing = True

    # make sure coverage looks for .coveragerc in the right place
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description=('Core test suite for Qcodes, '
                     'covering everything except instrument drivers'))

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase verbosity')

    parser.add_argument('-q', '--quiet', action='store_true',
                        help='reduce verbosity (opposite of --verbose)')

    parser.add_argument('-s', '--skip-coverage', default=True, type=int,
                        help='skip coverage reporting')

    parser.add_argument('-t', '--test_pattern', type=str, default='test*.py',
                        help=('regexp for test name to match, '
                              'default "test*.py"'))

    parser.add_argument('-f', '--failfast', action='store_true',
                        help='halt on first error/failure')

    parser.add_argument('-m', '--mp-spawn', action='store_true',
                        help=('force "spawn" method of starting child '
                              'processes to emulate Win behavior on Unix'))

    args = parser.parse_args()

    if args.mp_spawn:
        mp.set_start_method('spawn')

    args.skip_coverage |= coverage_missing

    if not args.skip_coverage:
        cov = coverage.Coverage(source=['qcodes'])
        cov.start()

    success = _test_core(verbosity=(1 + args.verbose - args.quiet),
                         failfast=args.failfast,
                         test_pattern=args.test_pattern)

    if not args.skip_coverage:
        cov.stop()
        cov.save()
        cov.report()

    # restore unix-y behavior
    # exit status 1 on fail
    if not success:
        sys.exit(1)
