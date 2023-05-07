from libs import *


class Camera:
    def __init__(self, pos=np.array([0.0, 0.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]), fov=60,
                world_up=np.array([0.0, 1.0, 0.0])):
        self.pos = pos
        self.forward = normalize(forward)
        self.right = normalize(-np.cross(self.forward, world_up))
        self.up = normalize(np.cross(self.right, self.forward))
        self.fov = fov
