import argparse
from helper import RenderScore, RelMSE
import compute

parser = argparse.ArgumentParser(
    description='Compute "render score" distance between MC renderings (via Mitsuba).'
)

parser.add_argument('refc',
                    metavar='ref-num-iter',
                    type=int,
                    help='Defines the number of individual renderings computed for the reference.',
                    )

parser.add_argument('refp',
                    metavar='ref-scene-path',
                    type=str,
                    help='Defines the path to the Mitsuba XML scene file used to compute the reference data.'
                    )

parser.add_argument('testc',
                    metavar='test-num-iter',
                    type=int,
                    help='Defines the number of individual renderings computed for the test scenes.',
                    )

parser.add_argument('testps',
                    metavar='test-scene-paths',
                    type=str,
                    nargs='+',
                    help='Defines the paths to the Mitsuba XML scene file used to compute the test data.'
                    )

args = parser.parse_args()
compute.compute_test((args.refc, args.refp), args.testc, args.testps, [RenderScore, RelMSE])