# QuadTree

Image Segmentation Animation using QuadTree concepts.

1. Split the Image into four quadrants.
2. Split the quadrant with the highest error into four quadrants.
3. Repeat N times.

## Images

<img src="Results/acacia_quad.png" alt="Segmented Image of an Acacia Tree" />
<img src="Results/rocks_quad.png" alt="Segmented Image of some Rocks" />
<img src="Results/palm_quad.png" alt="Segmented Image of some Palm Trees" />
<img src="Results/prairie_quad.png" alt="Segmented Image of a Prairie" />
<img src="Results/tree_quad.png" alt="Segmented Image of a tree on a cliff" />
<img src="Results/river_quad.png" alt="Segmented Image of a river" />
<img src="Results/tropical_quad.png" alt="Segmented Image of a Tropical Beach" />
<img src="Results/coastline_quad.png" alt="Segmented Image of a Coastline" />
<img src="Results/waterfalls_quad.png" alt="Segmented Image of some Waterfalls" />
<img src="Results/houses_quad.png" alt="Segmented Image of some Houses" />
<img src="Results/pipes_quad.png" alt="Segmented Image of some Pipes" />
<img src="Results/sand_quad.png" alt="Segmented Image of some sandstone stairs" />
<img src="Results/palace_quad.png" alt="Segmented Image of a Palace" />
<img src="Results/land_quad.png" alt="Segmented Image of a Landscape" />
<img src="Results/lightning_quad.png" alt="Segmented Image of Lightning" />
<img src="Results/night_quad.png" alt="Segmented Image of a road at night" />
<img src="Results/road_quad.png" alt="Segmented Image of a twisty road" />

## Video

<img src="Results/fish.gif" alt="Fish Video" />

## Usage

```
Quadtree Image Segmentation.

positional arguments:
  input                 Image to segment.
  output                Output filename.

optional arguments:
  -h, --help            show this help message and exit
  -q QUALITY, --quality QUALITY
                        Quality of the output video. (0-10), 0 worst, 10 best.
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations.
  -b, --border          Add borders to subimages.
  -au, --audio          Add audio from the input file to the output file.
  -mw MINWIDTH, --minwidth MINWIDTH
                        Minimum width of the smallest image quadrant.
  -mh MINHEIGHT, --minheight MINHEIGHT
                        Minimum height of the smallest image quadrant.
```

## Dependencies

```
numpy
tqdm
imageio
imageio-ffmpeg

pip install numpy tqdm imageio imageio-ffmpeg
```