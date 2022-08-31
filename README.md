# QuadTree

Image Segmentation Animation using QuadTree concepts.

1. Split the Image into four quadrants.
2. Split the quadrant with the highest error into four quadrants.
3. Repeat N times.

## Images

<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/acacia_quad.png" alt="Segmented Image of an Acacia Tree" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/rocks_quad.png" alt="Segmented Image of some Rocks" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/palm_quad.png" alt="Segmented Image of some Palm Trees" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/prairie_quad.png" alt="Segmented Image of a Prairie" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/tree_quad.png" alt="Segmented Image of a tree on a cliff" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/river_quad.png" alt="Segmented Image of a river" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/tropical_quad.png" alt="Segmented Image of a Tropical Beach" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/coastline_quad.png" alt="Segmented Image of a Coastline" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/moon_quad.png" alt="Segmented Image of Moonlight" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/waterfalls_quad.png" alt="Segmented Image of some Waterfalls" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/houses_quad.png" alt="Segmented Image of some Houses" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/pipes_quad.png" alt="Segmented Image of some Pipes" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/sand_quad.png" alt="Segmented Image of some sandstone stairs" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/palace_quad.png" alt="Segmented Image of a Palace" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/land_quad.png" alt="Segmented Image of a Landscape" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/lightning_quad.png" alt="Segmented Image of Lightning" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/night_quad.png" alt="Segmented Image of a road at night" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/road_quad.png" alt="Segmented Image of a twisty road" />

## Borders

<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/mountain.jpg" alt="Image of a Mountain Road" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/mountain_quad.png" alt="Segmented Image of a Mountain Road" />
<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/mountain_quad_noborder.png" alt="Segmented Image of a Mountain Road - No Borders" />

## Compression

With or without borders, the quadtree images achieve great compression, especially when using `png` encoding. Looking at the Mountain Images above, the original is a `1.51 MB jpg` file (`7.89 MB` when converted to `png`), while the Quadtree Image with borders is a `333 KB png` and the one without borders is a `160 KB png`.

## Video

<img src="https://github.com/AlexEidt/docs/blob/master/Quadtree/fish.gif" alt="Fish Video" />

## Usage

```
usage: quad.py [-h] [-q QUALITY] [-b] [-au] [-mw MINWIDTH] [-mh MINHEIGHT] input output iterations

Quadtree Image Segmentation.

positional arguments:
  input                 Image to segment.
  output                Output filename.
  iterations            Number of segmentation iterations.

optional arguments:
  -h, --help            show this help message and exit
  -q QUALITY, --quality QUALITY
                        Quality of the output video. (0-10), 0 worst, 10 best.
  -b, --border          Add borders to subimages.
  -a, --audio           Add audio from the input file to the output file.
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