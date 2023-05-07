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
num_ray_per_pixels = 1
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
    closest_dist = 10**6
    closest_obj = Object()
    for obj in scene:
        hit = obj.collision(ray)
        if hit.didHit:
            if closest_dist > hit.dist:
                closest_dist = hit.dist
                closest_obj = obj
    return closest_obj.colour


def Trace(ray, rngState):
    incoming_light = np.array([0.0, 0.0, 0.0])
    ray_colour = np.array([1.0, 1.0, 1.0])
    hit_obj = None
    for _ in range(bounce_limit + 1):
        hit, hit_obj = CalculateRyCollision(ray, hit_obj)
        if hit.didHit:
            ray.origin = hit.hitPoint
            diffuse_dir = normalize(hit.normal + randomDirection(rngState))
            specular_dir = reflect_dir(ray_dir=ray.direction, normal=hit.normal)
            ray.direction = normalize(lerp(diffuse_dir, specular_dir, hit_obj.smoothness))
            if hit_obj.emission_strength > 0.001:
                emitted_light = hit_obj.emission_colour * hit_obj.emission_strength
                incoming_light += emitted_light * ray_colour
                break
            ray_colour *= hit_obj.colour
        else:
            # incoming_light += directional_light_intensity * directional_light_colour * ray_colour * np.dot(ray.direction, directional_light_dir)
            break
    return incoming_light


def RT(ray, rngState):
    total_incoming_light = np.array([0.0, 0.0, 0.0])
    for _ in range(num_ray_per_pixels):
        total_incoming_light += Trace(ray, rngState)
    total_incoming_light = total_incoming_light / num_ray_per_pixels
    return total_incoming_light


averaged_frame_image = image.copy()
start_rank_time = time()
for frame in range(num_frames):
    for y_rank in range(rank_image_height):
        y = y_rank + rank * rank_image_height
        for x in range(image_width):
            rngState = np.array([y * image_width + x + frame * 719393], dtype=np.uint32)
            tx = x / (image_width - 1)
            ty = y / (image_height - 1)
            point_local = bottom_left_local + np.array([plane_width * tx, plane_height * ty, 0.0])
            point = camera.pos + camera.right * point_local[0] + camera.up * point_local[1] + camera.forward * point_local[2]
            ray = Ray(origin=camera.pos, direction=normalize(point - camera.pos))

            # Rasterization part for scene visualization
            # image[y_rank, x] = RC(ray)

            # Parallel Ray Traced for final image
            image[y_rank, x] = RT(ray, rngState)
    averaged_frame_image += image
end_rank_time = time()
averaged_frame_image = averaged_frame_image / num_frames
averaged_frame_image = np.clip(averaged_frame_image, 0.0, 1.0)*255
averaged_frame_image = averaged_frame_image.astype(np.uint8)

rank_time = end_rank_time - start_rank_time
print(f"rank: {rank}, time taken: {rank_time}")

collected_image = comm.gather(averaged_frame_image, root=0)
max_time = comm.reduce(rank_time, MPI.MAX, root=0)
if rank == 0:
    final_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for i in range(size):
        final_image[i*rank_image_height:(i+1)*rank_image_height] = collected_image[i]
    print(f"\nmax rank time: {max_time}")
    Image.fromarray(final_image).save(f"./outputs/oldprt-{max_time} {bounce_limit},{num_ray_per_pixels},{num_frames}.png")
    playsound('note.mp3')
