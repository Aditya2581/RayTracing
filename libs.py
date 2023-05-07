import random

import numpy as np


# def randomValue(state):
#     state[0] = (state[0] + 195439) * (state[0] + 124395) * (state[0] + 845921)
#     return abs(state[0] / 4294967295)

def randomValue(state):
    state[0] = state[0] * 747796405 + 2891336453
    result = np.array([5], dtype=np.uint32)
    result[0] = ((state[0] >> ((state[0] >> 28) + 4)) ^ state[0]) * 277803737
    result[0] = (result[0] >> 22) ^ result[0]
    return result[0]/4294967295.0

def randomValueNormal(state):
    theta = 2*np.pi*randomValue(state)
    rho = np.sqrt(-2*np.log(randomValue(state)))
    return rho*np.cos(theta)

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


# def GetEnvironmentLight(EnvironmentEnabled, ray):
#     if not EnvironmentEnabled:
#         return 0
#     skyGradientT = pow(smoothstep(0, 0.4, ray.dir.y), 0.35)
#     groundToSkyT = smoothstep(-0.01, 0, ray.dir.y)
#     skyGradient = lerp(SkyColourHorizon, SkyColourZenith, skyGradientT)
#     sun = pow(max(0, dot(ray.dir, _WorldSpaceLightPos0.xyz)), SunFocus) * SunIntensity
#     # Combine ground, sky, and sun
#     return composite
#     composite = lerp(GroundColour, skyGradient, groundToSkyT) + sun * (groundToSkyT >= 1)

def randomDirection(state):
    # x = random.normalvariate(0.5, 1.0)
    # y = random.normalvariate(0.5, 1.0)
    # z = random.normalvariate(0.5, 1.0)
    # x = randomValue(state)
    # y = randomValue(state)
    # z = randomValue(state)
    x = randomValueNormal(state)
    y = randomValueNormal(state)
    z = randomValueNormal(state)
    return normalize(np.array([x, y, z]))


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

