from libs import *


class Sphere(Object):
    def __init__(self, x, y, z, radius):
        self.center = np.array([x, y, z], dtype=np.float64)
        self.radius = radius
        # super().__init__()

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

