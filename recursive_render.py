import numpy as np

import copy
from libs import *
from PIL import Image
from time import time
import Scenes
from mpi4py import MPI
from playsound import playsound

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

world_up = np.array([0.0, 1.0, 0.0])
# image properties
image_width = 480
image_height = 360
if (image_height % size) != 0:
    exit()
rank_image_height = int(image_height / size)

scene, camera = Scenes.rgb_box()

plane_dist = 0.1
plane_height = plane_dist * np.tan(np.deg2rad(camera.fov * 0.5)) * 2
plane_width = plane_height * (image_width / image_height)

bottom_left_local = np.array([-plane_width / 2, -plane_height / 2, plane_dist])

# scene

# final array and RT properties
image = np.zeros((rank_image_height, image_width, 3), dtype=np.float64)
bounce_limit = 2
num_ray_primary_obj_bounce = 10
num_ray_secondary_obj_bounce = 10
# num_ray_per_bounce = 2
num_frames = 1


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


def RC(ray):
    closest_dist = 10 ** 6
    closest_obj = Object()
    for obj in scene:
        hit = obj.collision(ray)
        if hit.didHit:
            if closest_dist > hit.dist:
                closest_dist = hit.dist
                closest_obj = obj
    return closest_obj.colour


def Trace(ray, ray_colour, remaining_bounce, hit_obj, primary_obj_ray):
    incoming_light = np.array([0.0, 0.0, 0.0])
    hit, new_hit_obj = CalculateRyCollision(ray, hit_obj)
    if hit.didHit:
        new_ray_colour = copy.copy(ray_colour)
        new_ray = Ray()
        if new_hit_obj.emission_strength > 0.001:
            emitted_light = new_hit_obj.emission_colour * new_hit_obj.emission_strength
            incoming_light += emitted_light * new_ray_colour
            return incoming_light  # * (num_ray_per_bounce**(remaining_bounce+1))
        if remaining_bounce == 0:
            return incoming_light

        remaining_bounce -= 1
        new_ray.origin = hit.hitPoint
        new_ray_colour *= new_hit_obj.colour

        if primary_obj_ray != 0:
            num_ray_per_bounce = primary_obj_ray
        else:
            num_ray_per_bounce = num_ray_secondary_obj_bounce

        for _ in range(num_ray_per_bounce):
            diffuse_dir = normalize(hit.normal + normalize(np.random.randn(3)))
            specular_dir = reflect_dir(ray_dir=ray.direction, normal=hit.normal)
            new_ray.direction = normalize(lerp(diffuse_dir, specular_dir, new_hit_obj.smoothness))
            incoming_light += Trace(new_ray, new_ray_colour, remaining_bounce, new_hit_obj, 0)
        return incoming_light / num_ray_per_bounce
    return incoming_light


def RT(ray):
    ray_colour = np.array([1.0, 1.0, 1.0])
    pixel_colour = Trace(ray, ray_colour, bounce_limit, None, num_ray_primary_obj_bounce)
    return pixel_colour


total_main_function_time = 0.0

averaged_frame_image = image.copy()
start_rank_time = time()
for frame in range(num_frames):
    start_frame_time = time()
    for y_rank in range(rank_image_height):
        y = y_rank + rank * rank_image_height
        start_row_time = time()
        for x in range(image_width):
            tx = x / (image_width - 1)
            ty = y / (image_height - 1)
            point_local = bottom_left_local + np.array([plane_width * tx, plane_height * ty, 0.0])
            point = camera.pos + camera.right * point_local[0] + camera.up * point_local[1] + camera.forward * point_local[2]
            ray = Ray(origin=camera.pos, direction=normalize(point - camera.pos))

            start_main_function_time = time()

            # Rasterization part for scene visualization
            # image[y_rank, x] = RC(ray)

            # Parallel Ray Traced for final image
            image[y_rank, x] = RT(ray)

            end_main_function_time = time()
            total_main_function_time += end_main_function_time - start_main_function_time
        end_row_time = time()
    averaged_frame_image += image
    end_frame_time = time()
    pre_collection = 0
    if pre_collection:
        pre_average_frame = image
        pre_average_frame = np.clip(pre_average_frame, 0.0, 1.0) * 255
        pre_average_frame = pre_average_frame.astype(np.uint8)
        pre_collected_image = comm.gather(pre_average_frame, root=0)
        if rank == 0:
            final_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            for i in range(size):
                final_image[i * rank_image_height:(i + 1) * rank_image_height] = pre_collected_image[i]
            Image.fromarray(final_image).save(f"./outputs/rpre_oldprt1_{frame}.png")
            playsound('note.mp3')
end_rank_time = time()
averaged_frame_image = averaged_frame_image / num_frames
averaged_frame_image = np.clip(averaged_frame_image, 0.0, 1.0) * 255
averaged_frame_image = averaged_frame_image.astype(np.uint8)

rank_time = end_rank_time - start_rank_time
print(f"rank: {rank}, time taken: {rank_time}")
print(f"rank: {rank}, time taken by the main function: {total_main_function_time}")

max_time = comm.reduce(rank_time, MPI.MAX, root=0)
collected_image = comm.gather(averaged_frame_image, root=0)
if rank == 0:
    final_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for i in range(size):
        final_image[i * rank_image_height:(i + 1) * rank_image_height] = collected_image[i]
    print(f"\nmax rank time: {max_time}")
    Image.fromarray(final_image).save(f"./outputs/rc-{max_time} {bounce_limit},{num_frames}.png")
    playsound('note.mp3')
