import Sphere
import Plane
import numpy as np


def scene1():
    s0 = Sphere.Sphere(25, -11, 70, 10)
    s0.material(colour=np.array([255.0, 255.0, 255.0]))
    s1 = Sphere.Sphere(10, -13.5, 60, 7)
    s1.material(colour=np.array([255.0, 0.0, 0.0]))
    s2 = Sphere.Sphere(0, -16.5, 50, 3)
    s2.material(colour=np.array([0.0, 255.0, 0.0]))
    sun = Sphere.Sphere(0, 0, 1000, 500)
    sun.material(colour=np.array([255.0, 255.0, 255.0]))
    s3 = Sphere.Sphere(0, -1018, 0, 1000)
    s3.material(colour=np.array([0.0, 0.0, 255.0]))
    s4 = Sphere.Sphere(-500, 400, 1000, 100)
    s4.material(colour=np.array([255.0, 255.0, 255.0]),
                emission_colour=np.array([255.0, 255.0, 255.0]),
                emission_strength=100000.0)
    p0 = Plane.Plane(np.array([0.0, -20.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    p0.material(colour=np.array([0.0, 0.0, 255.0]))
    p1 = Plane.Plane(np.array([1000.0, 1000.0, 1000.0]), np.array([1.0, 1.0, 1.0]))
    # p1.material(colour=np.array([133.0, 243.0, 255.0]))
    p1.material(colour=np.array([133.0, 243.0, 255.0]),
                emission_colour=np.array([133.0, 243.0, 255.0]),
                emission_strength=100.0)
    scene = [s0, s1, s2, s3, s4]
    return scene
