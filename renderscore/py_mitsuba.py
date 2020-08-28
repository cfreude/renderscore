import os
import sys
import multiprocessing
import numpy as np

if os.name == 'nt':

    mts_path = 'D:/bin/mts/'
    # mts_path = 'C:/data/bin/mitsuba/'

    # NOTE: remember to specify paths using FORWARD slashes (i.e. '/' instead of
    # '\' to avoid pitfalls with string escaping)

    # Configure the search path for the Python extension module
    sys.path.append(mts_path + '/python/2.7')

    # Ensure that Python will be able to find the Mitsuba core libraries
    os.environ['PATH'] = mts_path + os.pathsep + os.environ['PATH']

from mitsuba.core import *
from mitsuba.render import SceneHandler, RenderQueue, RenderJob


class PyMitsuba:

    def __init__(self, _cpu_count=1, _debug_log=False):

        max_workers = multiprocessing.cpu_count()
        worker_count = max_workers if _cpu_count is None else min(_cpu_count, max_workers)
        print 'Mitsuba is using %d cpus.' % worker_count

        self.scheduler = Scheduler.getInstance()
        for i in range(0, worker_count):
            self.scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
        self.scheduler.start()
        self.queue = RenderQueue()
        self.pmgr = PluginManager.getInstance()

        if _debug_log:
            logger = Thread.getThread().getLogger()
            logger.setLogLevel(EWarn)

    def shutdown_mitsuba(self):
        self.queue.join()
        self.scheduler.stop()

    def load_scene(self, _scene_file_path):

        scene_path, scene_file = os.path.split(_scene_file_path)

        # Get a reference to the thread's file resolver
        file_resolver = Thread.getThread().getFileResolver()

        # Register any searchs path needed to load scene resources (optional)
        file_resolver.appendPath(scene_path)

        # Optional: supply parameters that can be accessed
        # by the scene (e.g. as $myParameter)
        param_map = StringMap()

        # Load the scene from an XML file
        scene = SceneHandler.loadScene(file_resolver.resolve(_scene_file_path), param_map)

        return scene

    def load_scene_custom(self, _scene_file_path, _spp=None, _img_w=None, _img_h=None, _seed=None, _npy=False):

        scene_path, scene_file = os.path.split(_scene_file_path)

        # Get a reference to the thread's file resolver
        file_resolver = Thread.getThread().getFileResolver()

        # Register any searchs path needed to load scene resources (optional)
        file_resolver.appendPath(scene_path)

        # Optional: supply parameters that can be accessed
        # by the scene (e.g. as $myParameter)
        param_map = StringMap()
        if _spp is not None:
            param_map['samplesPerPixel'] = str(_spp)
        if _img_w is not None:
            param_map['imgWidth'] = str(_img_w)
        if _img_h is not None:
            param_map['imgHeight'] = str(_img_h)
        if _seed is not None:
            param_map['samplerSeed'] = str(_seed)  # needs custom mitsuba bin

        # Load the scene from an XML file
        scene = SceneHandler.loadScene(file_resolver.resolve(_scene_file_path), param_map)

        pmgr = PluginManager.getInstance()

        if _npy:

            src_film_props = scene.getFilm().getProperties()

            film_props = Properties('mfilm')
            film_props['width'] = src_film_props['width']
            film_props['height'] = src_film_props['height']
            film_props['fileFormat'] = 'numpy'
            film_props['digits'] = 4
            film_props['pixelFormat'] = 'rgb'

            film = pmgr.createObject(film_props)
            film.configure()

            sprop = scene.getSensor().getProperties()
            sensor = pmgr.createObject(sprop)
            sensor.addChild(film)
            sensor.configure()

            scene.setSensor(sensor)

        return scene

    def render(self, _scenes):
        for scene in _scenes:
            self.start_render(scene)
        return self.stop_render()

    def start_render(self, scene):
        job = RenderJob('myRenderJob', scene, self.queue)
        job.start()

    def stop_render(self):
        self.queue.waitLeft(0)

    @staticmethod
    def get_img(_scene):
        film = _scene.getFilm()
        fsize = film.getSize()
        bitmap = Bitmap(Bitmap.ERGB, Bitmap.EFloat64, fsize)
        film.develop(Point2i(0, 0), fsize, Point2i(0, 0), bitmap)
        raw = np.frombuffer(bitmap.toByteArray(), dtype=np.float64)
        return np.reshape(raw, (fsize[1], fsize[0], 3,))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mts = PyMitsuba(8)
    scn = mts.load_scene('../data/veach_bidir/path.xml')
    scn.setDestinationFile('')
    mts.render([scn])
    plt.imshow(PyMitsuba.get_img(scn) ** (1.0 / 2.2))
    plt.show()
    mts.shutdown_mitsuba()
