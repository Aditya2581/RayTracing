import numpy as np


# This function normalizes any linear vector
def normalize(vect):
    norm = np.linalg.norm(vect)
    if norm == 0:
        return vect
    return vect / norm


# calculated reflected direction using laws of reflection
def reflect_dir(ray_dir, normal):
    return normalize(-2 * np.dot(normal, ray_dir) * normal + ray_dir)


# linear interpolator between two vectors
def lerp(vec1, vec2, factor):
    result = vec1 * (1 - factor) + vec2 * factor
    return result


# Ray class to create ray object with properties such as origin and direction
class Ray:
    # properties of this class
    # origin: starting point of the ray
    # direction: direction in which ray is pointing
    def __init__(self, origin=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 0.0])):
        self.origin = origin
        self.direction = normalize(direction)


# creates and object to store information about ray collision point
class hitInfo:
    # properties of this class:
    # didHit: true or false whether collision happened or not
    # dist: distance between ray origin and point of collision
    # hitPoint: coordinated of the point of collision
    # normal: normal direction to the surface at the point of collision
    def __init__(self, didHit=False, dist=0.0, hitPoint=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 0.0])):
        self.didHit = didHit
        self.dist = dist
        self.hitPoint = hitPoint
        self.normal = normalize(normal)


# Parent class to any physical object such as spheres, planes etc.
class Object:
    # necessary functions in any object are:
    # __init__: this hold the definition of the object such as its location, orientation and size
    # material: this defines its base colour (colour), the emission colour (emission_colour) and its strength (emission_strength), glossiness or smoothness (smooothness)
    # collision: it should store the method to calculate intersection by a given ray, and it returns hitInfo
    def __init__(self):
        self.material()

    def material(self, colour=np.array([0.0, 0.0, 0.0]), emission_colour=np.array([0.0, 0.0, 0.0]),
                 emission_strength=0.0, smoothness=0.0):
        self.emission_colour = emission_colour / 255.0  # emitting colour
        self.emission_strength = emission_strength  # emitting light strength
        if emission_strength > (10 ** (-3)):
            self.colour = np.array([0.0, 0.0, 0.0])
        else:
            self.colour = colour / 255.0
        self.smoothness = smoothness  # specular probability reflection

    def collision(self, ray):
        hit = hitInfo()     # stores hit information
        return hit


# Sphere object class to define spheres at certain location of some radius and also calculated intersection with a given ray
class Sphere(Object):
    # x: the x coordinate of center of the sphere
    # y: the y coordinate of center of the sphere
    # z: the z coordinate of center of the sphere
    # radius: radius of the sphere
    def __init__(self, x, y, z, radius):
        self.center = np.array([x, y, z], dtype=np.float64)
        self.radius = radius
        # super().__init__()

    # calculate collision point as distance of the collision with the given ray
    def collision(self, ray):
        hit = hitInfo()
        L = self.center - ray.origin
        tca = L.dot(ray.direction)
        d2 = L.dot(L) - tca * tca
        if d2 > self.radius * self.radius:
            hit.didHit = False
            return hit
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            hit.didHit = False
            return hit
        hit.didHit = True
        hit.dist = t0
        hit.hitPoint = ray.origin + t0 * ray.direction
        hit.normal = normalize(hit.hitPoint - self.center)
        return hit


# Plane object which is defined with a point on it and a normal to it
class Plane(Object):
    def __init__(self, point, normal):
        self.point = point
        self.normal = normalize(normal)
        # super().__init__()

    # calculating collision of plane with given ray
    def collision(self, ray):
        hit = hitInfo()
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 1e-6:
            t = self.normal.dot(self.point - ray.origin) / denom
            if t >= 0:
                hit.didHit = True
                hit.hitPoint = ray.origin + t * ray.direction
                hit.dist = t
                hit.normal = normalize(-denom * self.normal / abs(denom))
                return hit
        return hit


# Camera class creates a camera with properties such as :
# pos: position of the camera
# forward: forward direction of the camera
# right: cross product of world up direction and camera forward direction
# up: cross product right and forward direction
# fov: field of view of camera in degrees
class Camera:
    def __init__(self, pos=np.array([0.0, 0.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]), fov=60,
                 world_up=np.array([0.0, 1.0, 0.0])):
        self.pos = pos
        self.forward = normalize(forward)
        self.right = normalize(-np.cross(self.forward, world_up))
        self.up = normalize(np.cross(self.right, self.forward))
        self.fov = fov
