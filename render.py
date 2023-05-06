from libs import *
import Sphere
import Plane
from PIL import Image
from time import time
import Scenes
from sys import stdout
# import warnings
import matplotlib.pyplot as plt

# Turn the warning into an error
# warnings.filterwarnings("error", category=RuntimeWarning)

world_up = np.array([0.0, 1.0, 0.0])

# image properties
image_width = 480
image_height = 360

# camera properties
camera_pos = np.array([0.0, 0.0, 0.0])
camera_forward = normalize(np.array([0.0, 0.0, 1.0]))
camera_right = -np.cross(camera_forward, world_up)
camera_up = np.cross(camera_right, camera_forward)
camera_fov = 60

plane_height = 10.0 * np.tan(np.deg2rad(camera_fov * 0.5)) * 2
plane_width = plane_height * (image_width / image_height)

bottom_left_local = np.array([-plane_width / 2, -plane_height / 2, 10.0])

# scene
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
s4 = Sphere.Sphere(-500, 400, 1000, 1000)
s4.material(colour=np.array([255.0, 255.0, 255.0]),
            emission_colour=np.array([255.0, 255.0, 255.0]),
            emission_strength=1.0)
p0 = Plane.Plane(np.array([0.0, -20.0, 0.0]), np.array([0.0, 1.0, 0.0]))
p0.material(colour=np.array([0.0, 0.0, 255.0]))
p1 = Plane.Plane(np.array([1000.0, 1000.0, 1000.0]), np.array([1.0, 1.0, 1.0]))
# p1.material(colour=np.array([133.0, 243.0, 255.0]))
p1.material(colour=np.array([133.0, 243.0, 255.0]),
            emission_colour=np.array([133.0, 243.0, 255.0]),
            emission_strength=100.0)

scene = [s0, s1, s2, s3, s4]

# final array and RT properties
image = np.zeros((image_height, image_width, 3), dtype=np.float64)
bounce_limit = 2
recursion_ray_per_pixel = 2
num_ray_per_pixels = recursion_ray_per_pixel**bounce_limit


def CalculateRyCollision(ray, origin_obj=None):
    closest_hit = hitInfo(dist=10 ** 6)
    closest_obj = Object()
    for obj in scene:
        if obj is not origin_obj:
            hit = obj.collision(ray)
            if hit.didHit:
                if closest_hit.dist > hit.dist:
                    closest_hit = hit
                    closest_obj = obj
    return closest_hit, closest_obj


def Trace(ray, rngState, ray_colour, remaining_bounce, ray_origin_obj=None):
    incoming_light = np.array([0.0, 0.0, 0.0])
    # if remaining_bounce == -1:
    #     return incoming_light
    hit, hit_obj = CalculateRyCollision(ray, ray_origin_obj)
    if hit.didHit:
        if hit_obj.emission_strength != 0.0:
            emitted_light = hit_obj.emission_colour * hit_obj.emission_strength
            incoming_light += emitted_light * ray_colour
            return incoming_light
        if remaining_bounce == 0:
            return incoming_light
        remaining_bounce -= 1
        ray_colour *= hit_obj.colour
        ray.origin = hit.hitPoint
        for i in range(recursion_ray_per_pixel):
            ray.direction = normalize(hit.normal + randomDirection(rngState))
            incoming_light += Trace(ray, rngState, ray_colour, remaining_bounce, hit_obj)
        return incoming_light/recursion_ray_per_pixel
    # incoming_light += GetEnvironmentLight(ray) * ray_colour
    return incoming_light


def Raster(ray):
    closest_dist = 10**6
    closest_obj = Object()
    for obj in scene:
        hit = obj.collision(ray)
        if hit.didHit:
            if closest_dist > hit.dist:
                closest_dist = hit.dist
                closest_obj = obj
    return closest_obj.colour


def RT(x, y, ray):
    rngState = np.array([y * image_width + x], dtype=int)
    # d = num_ray_per_pixels
    ray_colour = np.array([1.0, 1.0, 1.0])
    total_incoming_light = Trace(ray, rngState, ray_colour, bounce_limit)
    # for _ in range(num_ray_per_pixels):
    #     # incoming_light = Trace(ray, rngState)
    #     # print(incoming_light)
    #     # if np.sum(incoming_light) == 0.0:
    #     #  d -= 1
    #     ray_colour = np.array([1.0, 1.0, 1.0])
    #     total_incoming_light += Trace(ray, rngState, ray_colour, bounce_limit)
    # # if d == 0:
    # #     d = 1
    return total_incoming_light


start_time = time()
row_time = 0.0
# analysis = np.zeros((image_height, 1), dtype=float)
for y in range(image_height):
    start_row_time = time()
    for x in range(image_width):
        tx = x / (image_width - 1)
        ty = y / (image_height - 1)
        point_local = bottom_left_local + np.array([plane_width * tx, plane_height * ty, 0.0])
        point = camera_pos + camera_right * point_local[0] + camera_up * point_local[1] + camera_forward * point_local[
            2]
        ray = Ray(origin=camera_pos, direction=normalize(point - camera_pos))

        # Rasterization part for scene visualization
        # image[y, x] = Raster(ray)

        # Serial Ray Traced for final image
        image[y, x] = RT(x, y, ray)
        # print(f"time for each pixel: {end_pixel_time - start_pixel_time}")
    end_row_time = time()
    row_time += end_row_time-start_row_time
    # time_to_finish = ((image_height-(y+1))*row_time)/60
    # analysis[y] = time_to_finish
    # stdout.flush()
    # stdout.write(f"\ry: {y}, time to finish: {time_to_finish} min")
    # print(f"estimated time to finish: {time_to_finish}", flush=True)

end_time = time()
# plt.plot(analysis)
# plt.show()
# print(image)
image = np.clip(image, 0.0, 1.0)*255
image = image.astype(np.uint8)
# print(f"shape of final image: {np.shape(image)}\nfinal image:\n{image}")

print(f"\nfor loop time: {end_time - start_time}")
Image.fromarray(image).save(f"./outputs/rt-{end_time - start_time}.png")
