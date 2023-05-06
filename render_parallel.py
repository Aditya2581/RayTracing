import copy

import numpy as np

from libs import *
import Sphere
import Plane
from PIL import Image
from time import time
from sys import stdout
# import warnings
# import matplotlib.pyplot as plt
from mpi4py import MPI
import Scenes

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Turn the warning into an error
# warnings.filterwarnings("error", category=R
# untimeWarning)

world_up = np.array([0.0, 1.0, 0.0])
directional_light_dir = -normalize(np.array([-1.0, -1.0, 1.0]))
directional_light_colour = np.array([1.0, 1.0, 1.0])
directional_light_intensity = 1.0

frame_averaging = False

# image properties
image_width = 480
image_height = 360
if (image_height % size) != 0:
    exit()
rank_image_height = int(image_height / size)

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
s2.material(colour=np.array([70.0, 241.0, 17.0]))
sun = Sphere.Sphere(0, 0, 1000, 500)
sun.material(colour=np.array([255.0, 255.0, 255.0]))
s3 = Sphere.Sphere(0, -1018, 0, 1000)
s3.material(colour=np.array([119.0, 78.0, 169.0]))
s4 = Sphere.Sphere(-500, 400, 1000, 1000)
s4.material(colour=np.array([255.0, 255.0, 255.0]),
            emission_colour=np.array([255.0, 255.0, 255.0]),
            emission_strength=0.5)
p0 = Plane.Plane(np.array([0.0, -20.0, 0.0]), np.array([0.0, 1.0, 0.0]))
p0.material(colour=np.array([0.0, 0.0, 255.0]))
p1 = Plane.Plane(np.array([1000.0, 1000.0, 1000.0]), np.array([1.0, 1.0, 1.0]))
# p1.material(colour=np.array([133.0, 243.0, 255.0]))
p1.material(colour=np.array([133.0, 243.0, 255.0]),
            emission_colour=np.array([133.0, 243.0, 255.0]),
            emission_strength=1000.0)

scene = [s0, s1, s2, s3, s4]

# final array and RT properties
image = np.zeros((rank_image_height, image_width, 3), dtype=np.float64)
bounce_limit = 1
recursion_ray_per_pixel = 20
num_ray_per_pixels = recursion_ray_per_pixel**bounce_limit
old_num_ray_per_pixels = 10
num_frames = 60

def CalculateRyCollision(ray, origin_obj = None):
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


def Trace(ray, rngState, ray_colour, remaining_bounce):
    incoming_light = np.array([0.0, 0.0, 0.0])
    # if remaining_bounce == -1:
    #     return incoming_light
    hit, hit_obj = CalculateRyCollision(ray)
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
            incoming_light += Trace(ray, rngState, ray_colour, remaining_bounce)
        return incoming_light/recursion_ray_per_pixel
    # incoming_light += GetEnvironmentLight(ray) * ray_colour
    return incoming_light


def oldTrace(ray, rngState):
    incoming_light = np.array([0.0, 0.0, 0.0])
    ray_colour = np.array([1.0, 1.0, 1.0])
    hit_obj = None
    for _ in range(bounce_limit+1):
        hit, hit_obj = CalculateRyCollision(ray, hit_obj)
        if hit.didHit:
            ray.origin = hit.hitPoint
            ray.direction = normalize(hit.normal + randomDirection(rngState))
            # if hit_obj.emission_strength != 0.0:
            #     break
            emitted_light = hit_obj.emission_colour * hit_obj.emission_strength
            incoming_light += emitted_light * ray_colour
            ray_colour *= hit_obj.colour
        else:
            # incoming_light += directional_light_intensity * directional_light_colour * ray_colour * np.dot(ray.direction, directional_light_dir)
            break
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


def RT(ray, rngState):
    ray_colour = np.array([1.0, 1.0, 1.0])
    total_incoming_light = Trace(ray, rngState, ray_colour, bounce_limit)
    return total_incoming_light


def oldRT(ray, rngState):
    total_incoming_light = np.array([0.0, 0.0, 0.0])
    for _ in range(old_num_ray_per_pixels):
        total_incoming_light += oldTrace(ray, rngState)
    total_incoming_light = total_incoming_light / old_num_ray_per_pixels
    return total_incoming_light


start_time = time()
row_time = 0.0
analysis = np.zeros((image_height, 1), dtype=float)

start_rank_time = time()
if not frame_averaging:
    for y_rank in range(rank_image_height):
        y = y_rank + rank*rank_image_height
        for x in range(image_width):
            rngState = np.array([y * image_width + x], dtype=np.uint32)
            tx = x / (image_width - 1)
            ty = y / (image_height - 1)
            point_local = bottom_left_local + np.array([plane_width * tx, plane_height * ty, 0.0])
            point = camera_pos + camera_right * point_local[0] + camera_up * point_local[1] + camera_forward * point_local[2]
            ray = Ray(origin=camera_pos, direction=normalize(point - camera_pos))

            # Rasterization part for scene visualization
            # image[y_rank, x] = Raster(ray)

            # Parallel Ray Traced for final image
            # image[y_rank, x] = RT(ray, rngState)
            image[y_rank, x] = oldRT(ray, rngState)
else:
    # mutiple frames averaging
    av_image = image.copy()
    for frame in range(num_frames):
        for y_rank in range(rank_image_height):
            y = y_rank + rank * rank_image_height
            for x in range(image_width):
                rngState = np.array([y * image_width + x + frame*719393], dtype=np.uint32)
                tx = x / (image_width - 1)
                ty = y / (image_height - 1)
                point_local = bottom_left_local + np.array([plane_width * tx, plane_height * ty, 0.0])
                point = camera_pos + camera_right * point_local[0] + camera_up * point_local[1] + camera_forward * point_local[2]
                ray = Ray(origin=camera_pos, direction=normalize(point - camera_pos))

                # Rasterization part for scene visualization
                # image[y_rank, x] = Raster(ray)

                # Parallel Ray Traced for final image
                # image[y_rank, x] = RT(ray, rngState)
                image[y_rank, x] = oldRT(ray, rngState)
        av_image += image
    image = av_image/num_frames

end_rank_time = time()
# row_time += end_row_time-start_row_time
# time_to_finish = ((image_height-(y+1))*row_time)/60
# analysis[y] = time_to_finish
# stdout.flush()
# stdout.write(f"\ry: {y}, time to finish: {time_to_finish} min")
# print(f"estimated time to finish: {time_to_finish}", flush=True)

# end_time = time()
# plt.plot(analysis)
rank_time = end_rank_time - start_rank_time
print(f"rank: {rank}, time taken: {rank_time}")
# print(image)
# image = np.ones(np.shape(image), dtype=float) * rank
# image = np.random.random(np.shape(image))
image = np.clip(image, 0.0, 1.0)*255
image = image.astype(np.uint8)
# print(f"rank: {rank}, image data:\n{image}")
# Image.fromarray(image).save(f"./outputs/{rank}prt-{rank_time}.png")

collected_image = comm.gather(image, root=0)
max_time = comm.reduce(rank_time, MPI.MAX, root=0)
# print(f"collected image: {collected_image}\nsize of collected image array: {np.shape(collected_image)}\n")
if rank == 0:
    final_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    # print(f"\n\n final image array:\n{final_image}\nshape of final image: {np.shape(final_image)}\n\n")
    for i in range(size):
        final_image[i*rank_image_height:(i+1)*rank_image_height] = collected_image[i]
    # print(f"size after filling: {np.shape(final_image)}\nfinal image after filling data:\n{final_image}\n\n")
    print(f"\nmax row time: {max_time}")
    Image.fromarray(final_image).save(f"./outputs/prt-{max_time} {recursion_ray_per_pixel},{bounce_limit}.png")
