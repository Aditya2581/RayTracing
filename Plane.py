from libs import *


class Plane(Object):
    def __init__(self, point, normal):
        self.point = point
        self.normal = normalize(normal)
        # super().__init__()

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
