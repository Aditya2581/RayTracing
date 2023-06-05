from libs import *


def coordinate_test():
    s0 = Sphere(0, 0, 100, 5)
    s0.material(colour=np.array([255.0, 255.0, 255.0]))
    s1 = Sphere(20, 20, 100, 5)
    s1.material(colour=np.array([255.0, 0.0, 0.0]))
    s2 = Sphere(-20, 20, 100, 5)
    s2.material(colour=np.array([0.0, 255.0, 0.0]))
    s3 = Sphere(20, -20, 100, 5)
    s3.material(colour=np.array([0.0, 0.0, 255.0]))
    s4 = Sphere(-20, -20, 100, 5)
    s4.material(colour=np.array([163.0, 7.0, 178.0]))
    scene = [s0, s1, s2, s3, s4]
    camera = Camera()
    return scene, camera


def sphere_scene():
    s0 = Sphere(12.31, -6.76, 33.1, 5)
    s0.material(colour=np.array([255.0, 255.0, 255.0]))
    s1 = Sphere(5.64, -8.42, 31, 3.5)
    s1.material(colour=np.array([255.0, 0.0, 0.0]))
    s2 = Sphere(0.8, -9.51, 25.42, 1.5)
    s2.material(colour=np.array([0.0, 255.0, 0.0]))
    s3 = Sphere(0, -510.4, -0.1, 500)
    s3.material(colour=np.array([119.0, 77.0, 168.0]))
    s4 = Sphere(-210, 193.5, 559.4, 500)
    s4.material(colour=np.array([255.0, 255.0, 255.0]),
                emission_colour=np.array([255.0, 255.0, 255.0]),
                emission_strength=2.0)
    scene = [s0, s1, s2, s3, s4]
    camera = Camera()
    return scene, camera


def rgb_box():
    p_bottom = Plane(np.array([0.0, -4.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    p_left = Plane(np.array([-4.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    p_right = Plane(np.array([4.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    p_top = Plane(np.array([0.0, 4.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    p_back = Plane(np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.0, -1.0]))
    p_bottom.material(colour=np.array([0.0, 255.0, 0.0]))
    p_left.material(colour=np.array([255.0, 0.0, 0.0]))
    p_right.material(colour=np.array([0.0, 0.0, 255.0]))
    p_top.material(colour=np.array([255.0, 255.0, 255.0]), emission_colour=np.array([255.0, 255.0, 255.0]), emission_strength=1.0)
    p_back.material(colour=np.array([150.0, 150.0, 150.0]))

    s0 = Sphere(0, 0, 0, 1)
    s1 = Sphere(2, 2, 2, 1)
    s2 = Sphere(-2, 2, 2, 1)
    s3 = Sphere(2, -2, -2, 1)
    s4 = Sphere(-2, -2, -2, 1)
    s0.material(colour=np.array([200.0, 200.0, 200.0]), smoothness=1.0)
    s1.material(colour=np.array([200.0, 0.0, 0.0]), smoothness=0.6)
    s2.material(colour=np.array([0.0, 200.0, 0.0]), smoothness=0.4)
    s3.material(colour=np.array([0.0, 0.0, 200.0]), smoothness=0.2)
    s4.material(colour=np.array([163.0, 7.0, 178.0]), smoothness=0.1)

    scene = [p_bottom, p_left, p_top, p_back, p_right, s0, s1, s2, s3, s4]
    camera = Camera(pos=np.array([0.0, 0.0, -10.0]))
    return scene, camera
