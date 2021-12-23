# QuadTree

Image Segmentation Animation using QuadTree concepts.

<p align="center">
    <img src="Results/bananas.jpg" alt="Bananas" />
    <img src="Results/bananas_quad.png" alt="Segmented Bananas" />
    <img src="Results/bananas.gif" alt="Bananas GIF" />
    <img src="Results/donuts.jpg" alt="Donuts" />
    <img src="Results/donuts_quad.png" alt="Segmented Donuts" />
    <img src="Results/donuts.gif" alt="Donuts GIF" />
    <img src="Results/forest.jpg" alt="Forest" />
    <img src="Results/forest_quad.png" alt="Segmented Forest" />
    <img src="Results/forest.gif" alt="Forest GIF" />
</p>

## Usage

```
usage: quad.py [-h] [-fps FPS] [-i ITERATIONS] [-ws WRITESTART] [-b] [-img] [-s STEP] input output

Quadtree Image Segmentation.

positional arguments:
  input                 Image to segment.
  output                Output filename.

optional arguments:
  -h, --help            show this help message and exit
  -fps FPS              Output FPS.
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations.
  -ws WRITESTART, --writestart WRITESTART
                        Number of frames to write in sequence initially.
  -b, --border          Add borders to subimages.
  -img, --image         Save final output image.
  -s STEP, --step STEP  Once `iterations > ws`, only save a frame every `(iterations - ws)^s` iterations.
```

## Dependencies

```
numpy
tqdm
imageio
imageio-ffmpeg

pip install numpy tqdm imageio
pip install imageio-ffmpeg --user
```