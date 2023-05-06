import numpy as np

from libs import *
import Scenes
import Sphere
import Plane
from PIL import Image
from time import time
from mpi4py import MPI

world_up = np.array([0.0, 1.0, 0.0])
directional_light_dir = -normalize(np.array([-1.0, -1.0, 1.0]))
directional_light_colour = np.array([1.0, 1.0, 1.0])
directional_light_intensity = 1.0

# image parameters
image_width = 480
image_height = 360

# camera properties
camera_pos = np.array([0.0, 0.0, 0.0])
camera_forward = normalize(np.array([0.0, 0.0, 1.0]))
camera_right = normalize(-np.cross(camera_forward, world_up))
camera_up = normalize(np.cross(camera_right, camera_forward))
camera_fov = 60

scene = Scenes.rgb_box()

# RT parameters
bounce_limit = 1

# camera ray direction
near_plane_dist = 10.0
near_plane_height = near_plane_dist * np.tan(np.rad2deg(camera_fov * 0.5)) * 2
near_plane_width = near_plane_height * image_width / image_height

bottom_left_corner_local = np.array([-near_plane_width/2, -near_plane_height/2, near_plane_dist])

image = np.zeros([image_height, image_width, 3], dtype=np.float64)


# Ray Casting
def RayCasting(ray):
    closest_dist = 10 ** 6
    closest_obj = Object()
    for obj in scene:
        hit = obj.collision(ray)
        if hit.didHit:
            if closest_dist > hit.dist:
                closest_dist = hit.dist
                closest_obj = obj
    return closest_obj.colour


# Ray Tracing
def CalculateRayCollision(ray, origin_obj=None):
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


def RayTracing(ray):
    # direct Illumination
    hit, hit_obj = CalculateRayCollision(ray)
    DI, diffuse, specular = np.array([0.0, 0.0, 0.0])
    if hit.didHit:
        shadow_ray = Ray(hit.hitPoint, directional_light_dir)
        shadow_hit, shadow_hit_obj = CalculateRayCollision(shadow_ray, hit_obj)
        reflection_ray = Ray(hit.hitPoint, reflect_dir(ray.direction, hit.normal))
        reflect_hit, reflect_hit_obj = CalculateRayCollision(reflection_ray, hit_obj)
        if not shadow_hit.didHit:
            diffuse = directional_light_colour * hit_obj.colour * np.dot(hit.normal, directional_light_dir) * directional_light_intensity
        if reflect_hit.didHit:
            specular = reflect_hit_obj.colour


    return diffuse


# Render Loop
start_time = time()
for y in range(image_height):
    for x in range(image_width):
        # pixel ray
        ray = Ray()
        ray.origin = camera_pos
        ray_direction_local = normalize(bottom_left_corner_local + np.array([near_plane_width*x/(image_width-1), near_plane_height*y/(image_height-1), 0.0]))
        ray.direction = normalize(camera_right*ray_direction_local[0] + camera_up*ray_direction_local[1] + camera_forward*ray_direction_local[2])

        # ray casting (for relatively faster scene visualization)
        # image[y, x] = RayCasting(ray)

        # ray tracing (for realistic shadows and reflections)
        image[y, x] = RayTracing(ray)

end_time = time()
image = np.clip(image, 0.0, 1.0)*255
# print(image)
image = image.astype(np.uint8)
print(f"\nfor loop time: {end_time - start_time}")
Image.fromarray(image).save(f"./outputs/rt-{end_time - start_time}.png")