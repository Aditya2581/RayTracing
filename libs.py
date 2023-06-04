import numpy as np


def normalize(vect):
    norm = np.linalg.norm(vect)
    if norm == 0:
        return vect
    return vect / norm

def reflect_dir(ray_dir, normal):
    return normalize(-2 * np.dot(normal, ray_dir)*normal + ray_dir)

def lerp(vec1, vec2, factor):
    result = vec1 * (1-factor) + vec2 * factor
    return result


class Ray:
    def __init__(self, origin=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 0.0])):
        self.origin = origin
        self.direction = normalize(direction)


class hitInfo:
    def __init__(self, didHit=False, dist=0.0, hitPoint=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 0.0])):
        self.didHit = didHit
        self.dist = dist
        self.hitPoint = hitPoint
        self.normal = normalize(normal)


class Object:
    def __init__(self):
        self.material()
    # self.colour = np.array([1.0, 0.0, 0.0])
    # self.emission_colour = np.array([1.0, 1.0, 1.0])
    # self.emission_strength = 0.0
    # self.smoothness = 0.0

    def material(self, colour=np.array([0.0, 0.0, 0.0]), emission_colour=np.array([0.0, 0.0, 0.0]),
                 emission_strength=0.0, smoothness=0.0):
        # self.colour = colour/255.0  # object colour
        self.emission_colour = emission_colour/255.0  # emitting colour
        self.emission_strength = emission_strength  # emitting light strength
        if emission_strength > (10**(-3)):
            self.colour = np.array([0.0, 0.0, 0.0])
        else:
            self.colour = colour/255.0
        self.smoothness = smoothness  # specular probability reflection

    def collision(self, ray):
        hit = hitInfo()
        return hit
