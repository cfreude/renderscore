import os, pprint
import numpy as np
import matplotlib.pyplot as plt

from py_mitsuba import PyMitsuba as PyMts
from helper import SampleStats, RenderScore, RelMSE


def init_figure(_rows, _cols):

    fig, axs = plt.subplots(_rows, _cols, sharex=True, sharey=True)
    axs = axs.flatten()

    hs = []

    rnd = np.random.rand(1, 1, 3)
    for ax in axs:
        hs.append(ax.imshow(rnd))

    plt.ion()
    plt.show()

    return fig, axs, hs


def build_views(_tasks, _ref_mean, _ref_coov, _cols):

    # fixed views
    views = [
        ('Ref. mean', None, None, lambda i, k: _ref_mean),
        ('Ref. coov', None, None, lambda i, k: _ref_coov),
    ]

    views += [('', None, None, lambda i, k: np.ones((1,1)))] * (_cols - 2)

    # procedural views
    proc_views = []
    for tk, (_, inst_arr) in _tasks.items():
        for df in inst_arr:
            for vk, _ in df.get_values().items():
                desc = '%s: %s' % (tk, vk)
                val = (desc, df, vk, lambda i, k: i.get_values()[k])
                proc_views.append(val)

    views += proc_views

    return views


def compute_test(_reference_parameters, _test_count, _test_paths, _cpu_count=None, _show=False):

    # define views
    rows = 1 + len(scene_dict.keys())
    cols = sum([len(i().get_values().keys()) for i in sample_processors])
    print rows, cols

    fig, axs, hs = init_figure(rows, cols)
    views = build_views(tasks, ref_mean, ref_coov, cols)

    # TODO add tone mapping

    for i in range(_test_count):

        # render
        mts.render([scene for k, (scene, _) in tasks.items()])

        # process samples
        for _, (scene, dist) in tasks.items():
            sample = PyMts.get_img(scene)
            for d in dist:
                d.push(sample)

        # plot data
        for axi, (desc, inst, key, dfunc) in enumerate(views):
            ax = axs[axi]
            hs[axi].set_data(dfunc(inst, key))
            ax.set_title(desc)
        fig.suptitle('Sample iteration: %d/%d' % (i+1, _test_count))
        plt.pause(.05)

    plt.show(block=True)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Compute distance between MC renderings (via Mitsuba).'
    )

    parser.add_argument('refc',
                        metavar='test-count',
                        type=int,
                        help='Defines the number of individual renderings computed for the test.',
                        )

    parser.add_argument('ref',
                        metavar='test-paths',
                        type=str,
                        help='Defines the paths to the Mitsuba XML scene file used to generate the test rendering data.'
                        )

    #args = parser.parse_args()
    #compute_reference(args.refc, args.ref)

    compute_test((32, '../data/veach_ajar/bdpt.xml'), 16, ['../data/veach_ajar/path.xml', '../data/veach_ajar/erpt.xml'], 6, True)
    #compute_test(16, ['../data/veach_bidir/path.xml', '../data/veach_bidir/erpt.xml'], 6, True)

