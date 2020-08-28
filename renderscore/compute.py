import os
import numpy as np
from py_mitsuba import PyMitsuba as PyMts
from helper import SampleStats
import common

CPU_COUNT = None


def print_results(_i, _tasks, _distance_measures):
    col_width = 15
    col_names = [d().get_values().keys()[0] for d in _distance_measures]
    row_format = ('{:<%d}' % col_width) * (len(col_names) + 1)
    print '#' * col_width * (len(col_names) + 1)
    print row_format.format('ITERATION #%s' % _i, *col_names)
    for scene_name, (scene, dist) in _tasks.items():
        avg_values = []
        for d in dist:
            if type(d) != SampleStats:
                for value_name, value in d.get_values().items():
                    avg = np.nanmean(value)
                    avg_values.append(avg)
        print row_format.format(scene_name, *avg_values)
    print '#' * col_width * (len(col_names) + 1)


@common.cache_to_disk('./cache/')
def compute_reference(_num_iterations, _ref_scene_path):

    print 'Computing reference data ...'
    mts = PyMts(CPU_COUNT)
    scn = mts.load_scene(_ref_scene_path)

    stats = common.OnlineStats()

    for i in range(_num_iterations):
        mts.render([scn])
        rend_a = PyMts.get_img(scn)
        stats.push(rend_a)

    mts.shutdown_mitsuba()

    return stats.mean(), stats.standard_deviation()


def compute_test(_reference_parameters, _num_iterations, _scene_paths, _distance_measures, _iteration_callback=None):

    # prepare reference
    ref_mean, ref_std = compute_reference(*_reference_parameters)
    ref_coov = np.copy(ref_std)
    np.divide(ref_std, ref_mean, out=ref_coov, where=ref_mean > 0.0)

    mts = PyMts(CPU_COUNT)

    # config
    scene_dict = {}
    for path in _scene_paths:
        name = os.path.basename(os.path.splitext(path)[0])
        scene_dict[name] = path

    sample_processors = [SampleStats] + _distance_measures

    # load scenes and create distance class instances
    tasks = {}
    for name, path in scene_dict.items():
        scene = mts.load_scene(path)
        inst_arr = []
        for classtype in sample_processors:
            instance = classtype()
            instance.set_reference(ref_mean, ref_std)
            inst_arr.append(instance)
        tasks[name] = (scene, inst_arr)

    # iterate
    for i in range(_num_iterations):

        # render
        mts.render([scene for k, (scene, _) in tasks.items()])

        # process samples
        for _, (scene, dist) in tasks.items():
            sample = PyMts.get_img(scene)
            for d in dist:
                d.push(sample)

        print_results(i+1, tasks, _distance_measures)
