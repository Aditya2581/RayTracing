# RayTracing
Project for HPCE course

Currently old_renderer.py works, it has parallel code (serial version can be found in previous commits or other files, a seperate file for seriel code will be added later)

To run the code, download the entire repository install all the dependencies
give the command in the termial: mpiexec -n 8 python old_renderer.py

Important files apart from the renderer file are:
    libs.py
    Sphere.py
    Plane.py
    Scenes.py
