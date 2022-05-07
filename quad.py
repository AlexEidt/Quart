#!/usr/bin/env python3

"""
Quadtree Image Segmentation
Alex Eidt

Split the image into four quadrants.
Find the quadrant with the highest error.
Split that quadrant into four quadrants.
Repeat N times.
"""

import heapq
import argparse
import imageio
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm


def border(image):
    """
    Add a black border around the given image quadrant.
    Add two black lines in the middle of the quadrant vertically and horizontally.
    """
    # Add black border
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0

    h, w = image.shape[:2]
    half_w = w // 2
    half_h = h // 2
    # Horizontal Line
    image[half_h : half_h + 1, :] = 0
    # Vertical Line
    image[:, half_w : half_w + 1] = 0


def error(image, avg):
    """
    Compute the error of a given quadrant.
    """
    if image.size == 0:
        return 0
    h, w = image.shape[:2]
    mean = avg[0] * 0.299 + avg[1] * 0.587 + avg[2] * 0.114
    return np.sum((image - mean) ** 2) / (h * w)


def quad(image, edited, iterations, quadrants=None, min_width=10, min_height=10, set_border=True):
    """
    Split the given image into four quadrants.
    Update the edited image by coloring in the newly split quadrants to the average rgb
    color of the original image.
    Find the quadrant with the maximum error, remove it from the "quadrants" list and return it.

    The resulting quadtree image is stored in "edited".
    """
    if iterations <= 0:
        return image, edited

    if quadrants is None:
        quadrants = []

    gray = (image * np.array([0.299, 0.587, 0.114])).sum(axis=2, dtype=np.uint8)

    h, w = image.shape[:2]
    # Create the integral image, edge padded by one to the top and left.
    I = np.pad(image.astype(np.uint32), ((1, 0), (1, 0), (0, 0)), mode='edge')
    np.cumsum(I, axis=0, out=I)
    np.cumsum(I, axis=1, out=I)

    # Top left quadrant x and y coordinates.
    x, y = 0, 0

    for _ in range(iterations):
        if h > min_height and w > min_width:
            hw, hh = w // 2, h // 2

            tlA, tlB, tlC, tlD = I[y, x], I[y, x+hw], I[y+hh, x], I[y+hh, x+hw]
            trA, trB, trC, trD = I[y, x+hw], I[y, x+w], I[y+hh, x+hw], I[y+hh, x+w]
            blA, blB, blC, blD = I[y+hh, x], I[y+hh, x+hw], I[y+h, x], I[y+h, x+hw]
            brA, brB, brC, brD = I[y+hh, x+hw], I[y+hh, x+w], I[y+h, x+hw], I[y+h, x+w]

            tl_avg = (tlD + tlA - tlB - tlC) / (hw * hh)
            tr_avg = (trD + trA - trB - trC) / ((w - hw) * hh)
            bl_avg = (blD + blA - blB - blC) / (hw * (h - hh))
            br_avg = (brD + brA - brB - brC) / ((w - hw) * (h - hh))

            edited[y:y+hh, x:x+hw] = tl_avg         # Top Left
            edited[y:y+hh, x+hw:x+w] = tr_avg       # Top Right
            edited[y+hh:y+h, x:x+hw] = bl_avg       # Bottom Left
            edited[y+hh:y+h, x+hw:x+w] = br_avg     # Bottom Right

            if set_border:
                border(edited[y:y+h, x:x+w])

            heapq.heappush(quadrants, (-error(gray[y:y+hh, x:x+hw], tl_avg), x, y, hw, hh))
            heapq.heappush(quadrants, (-error(gray[y:y+hh, x+hw:x+w], tr_avg), x + hw, y, w - hw, hh))
            heapq.heappush(quadrants, (-error(gray[y+hh:y+h, x:x+hw], bl_avg), x, y + hh, hw, h - hh))
            heapq.heappush(quadrants, (-error(gray[y+hh:y+h, x+hw:x+w], br_avg), x + hw, y + hh, w - hw, h - hh))

        if quadrants:
            _, x, y, w, h = heapq.heappop(quadrants)
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Quadtree Image Segmentation.")
    parser.add_argument("input", type=str, help="Image to segment.")
    parser.add_argument("output", type=str, help="Output filename.")
    parser.add_argument("-q", "--quality", type=int, default=5, help="Quality of the output video. (0-10), 0 worst, 10 best.")
    parser.add_argument("-i", "--iterations", type=int, default=12, help="Number of iterations.")
    parser.add_argument("-b", "--border", action="store_true", help="Add borders to subimages.")
    parser.add_argument("-au", "--audio", action="store_true", help="Add audio from the input file to the output file.")
    parser.add_argument("-mw", "--minwidth", type=int, default=10, help="Minimum width of the smallest image quadrant.")
    parser.add_argument("-mh", "--minheight", type=int, default=10, help="Minimum height of the smallest image quadrant.")

    return parser.parse_args()


def quadtree_video(args):
    """
    Convert every frame of a video to a quadtree image.
    """
    # Convert every frame of input video to quadtree image and store as output video.
    with imageio.read(args.input) as video:
        data = video.get_meta_data()

        kwargs = {"fps": data["fps"], "quality": min(max(args.quality, 0), 10)}
        if args.audio:
            kwargs["audio_path"] = args.input

        writer = imageio_ffmpeg.write_frames(args.output, data["source_size"], **kwargs)
        writer.send(None)

        quadrants = []
        buffer = np.empty(data['source_size'][::-1] + (3,), dtype=np.uint8)
        for frame in tqdm(video, total=int(data["fps"] * data["duration"] + 0.5)):
            np.copyto(buffer, frame)
            quad(
                frame,
                buffer,
                args.iterations,
                quadrants=quadrants,
                min_width=args.minwidth,
                min_height=args.minheight,
                set_border=args.border,
            )
            writer.send(buffer)
            quadrants.clear()

        writer.close()

    
def quadtree_image(args, image):
    copy = image.copy()
    quad(
        image,
        copy,
        args.iterations,
        min_width=args.minwidth,
        min_height=args.minheight,
        set_border=args.border,
    )
    imageio.imsave(args.output, copy)


def main():
    args = parse_args()

    # Try to load an image from the given input. If this fails, assume it's a video.
    try:
        image = imageio.imread(args.input)[..., :3]
    except Exception:
        quadtree_video(args)
    else:
        quadtree_image(args, image)


if __name__ == "__main__":
    main()
