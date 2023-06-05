import copy
from libs import *
from PIL import Image
from time import time
import Scenes
from mpi4py import MPI
from playsound import playsound

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

world_up = np.array([0.0, 1.0, 0.0])

# Image properties
image_width = 480
image_height = 360
if (image_height % size) != 0:
    exit()
rank_image_height = int(image_height / size)

# Scene and camera setup
scene, camera = Scenes.rgb_box()

# Virtual image plane properties
plane_dist = 0.1
plane_height = plane_dist * np.tan(np.deg2rad(camera.fov * 0.5)) * 2
plane_width = plane_height * (image_width / image_height)

bottom_left_local = np.array([-plane_width / 2, -plane_height / 2, plane_dist])

# Initialize the image array
image = np.zeros((rank_image_height, image_width, 3), dtype=np.float64)

# Ray tracing parameters
bounce_limit = 2
num_ray_primary_obj_bounce = 2
num_ray_secondary_obj_bounce = 2


# Function to calculate the closest collision with objects in the scene
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


# Function for rasterization (not used in the final code)
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


# Function for tracing rays and calculating incoming light recursively
def Trace(ray, ray_colour, remaining_bounce, hit_obj, primary_obj_ray):
    incoming_light = np.array([0.0, 0.0, 0.0])

    # Find the closest intersection point and object
    hit, new_hit_obj = CalculateRyCollision(ray, hit_obj)

    if hit.didHit:
        new_ray_colour = copy.copy(ray_colour)
        new_ray = Ray()

        if new_hit_obj.emission_strength > 0.001:
            # Handle emission from the object
            emitted_light = new_hit_obj.emission_colour * new_hit_obj.emission_strength
            incoming_light += emitted_light * new_ray_colour
            return incoming_light

        if remaining_bounce == 0:
            # Reached the maximum bounce limit, stop recursion
            return incoming_light

        remaining_bounce -= 1
        new_ray.origin = hit.hitPoint
        new_ray_colour *= new_hit_obj.colour

        if primary_obj_ray != 0:
            num_ray_per_bounce = primary_obj_ray
        else:
            num_ray_per_bounce = num_ray_secondary_obj_bounce

        for _ in range(num_ray_per_bounce):
            # Generate random diffuse and specular directions
            diffuse_dir = normalize(hit.normal + normalize(np.random.randn(3)))
            specular_dir = reflect_dir(ray_dir=ray.direction, normal=hit.normal)
            new_ray.direction = normalize(lerp(diffuse_dir, specular_dir, new_hit_obj.smoothness))

            # Recursively trace the new ray and accumulate incoming light
            incoming_light += Trace(new_ray, new_ray_colour, remaining_bounce, new_hit_obj, 0)

        return incoming_light / num_ray_per_bounce

    return incoming_light


# Function for performing ray tracing for a pixel
def RT(ray):
    ray_colour = np.array([1.0, 1.0, 1.0])
    pixel_colour = Trace(ray, ray_colour, bounce_limit, None, num_ray_primary_obj_bounce)
    return pixel_colour


# Variables to track time
total_main_function_time = 0.0
start_rank_time = time()

# Loop through each row assigned to the rank
for y_rank in range(rank_image_height):
    y = y_rank + rank * rank_image_height
    start_row_time = time()
    # Loop through each pixel in the row
    for x in range(image_width):
        tx = x / (image_width - 1)
        ty = y / (image_height - 1)
        # Calculate the position of the pixel on the virtual image plane
        point_local = bottom_left_local + np.array([plane_width * tx, plane_height * ty, 0.0])
        # Calculate the position of the pixel in the world space
        point = camera.pos + camera.right * point_local[0] + camera.up * point_local[1] + camera.forward * point_local[2]
        # Create a ray from the camera position to the pixel position
        ray = Ray(origin=camera.pos, direction=normalize(point - camera.pos))

        start_main_function_time = time()

        # Rasterization part for scene visualization (not used in the final code)
        # image[y_rank, x] = RC(ray)

        # Parallel Ray Tracing for final image
        image[y_rank, x] = RT(ray)

        end_main_function_time = time()
        total_main_function_time += end_main_function_time - start_main_function_time
    end_row_time = time()
end_rank_time = time()

# Clip and convert the image to the appropriate data type
image = np.clip(image, 0.0, 1.0) * 255
image = image.astype(np.uint8)

# Calculate the maximum rank time across all processes
rank_time = end_rank_time - start_rank_time
max_time = comm.reduce(rank_time, MPI.MAX, root=0)

# Print timing information
print(f"rank: {rank}, time taken: {rank_time}")
print(f"rank: {rank}, time taken by the main function: {total_main_function_time}")

# Gather the image data from all processes to process 0
collected_image = comm.gather(image, root=0)

if rank == 0:
    # Combine the collected images from each rank
    final_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for i in range(size):
        final_image[i * rank_image_height:(i + 1) * rank_image_height] = collected_image[i]

    # Save the final image
    Image.fromarray(final_image).save(f"./outputs/rc-{max_time} {bounce_limit}.png")
    # Play a sound to indicate completion
    playsound('note.mp3')
    print(f"\nmax rank time: {max_time}")
