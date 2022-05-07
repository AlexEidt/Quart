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

    gray = (image * np.array([0.299, 0.587, 0.114])).sum(axis=2)

    h, w = image.shape[:2]
    # Create the integral image, edge padded by one to the top and left.
    I = np.pad(image.astype(np.uint32), ((1, 0), (1, 0), (0, 0)), mode='edge')
    np.cumsum(I, axis=0, out=I)
    np.cumsum(I, axis=1, out=I)

    # Top left quadrant x and y coordinates.
    tlx, tly = 0, 0

    index = 0
    for _ in range(iterations):
        h, w = image.shape[:2]

        if h > min_height and w > min_width:
            hw, hh = w // 2, h // 2

            tlA, tlB, tlC, tlD = I[tly, tlx], I[tly, tlx+hw], I[tly+hh, tlx], I[tly+hh, tlx+hw]
            trA, trB, trC, trD = I[tly, tlx+hw], I[tly, tlx+w], I[tly+hh, tlx+hw], I[tly+hh, tlx+w]
            blA, blB, blC, blD = I[tly+hh, tlx], I[tly+hh, tlx+hw], I[tly+h, tlx], I[tly+h, tlx+hw]
            brA, brB, brC, brD = I[tly+hh, tlx+hw], I[tly+hh, tlx+w], I[tly+h, tlx+hw], I[tly+h, tlx+w]

            tl = image[:hh, :hw]
            tr = image[:hh, hw:]
            bl = image[hh:, :hw]
            br = image[hh:, hw:]

            tlg = gray[:hh, :hw]
            trg = gray[:hh, hw:]
            blg = gray[hh:, :hw]
            brg = gray[hh:, hw:]

            (tlh, tlw), (trh, trw), (blh, blw), (brh, brw) = tlg.shape, trg.shape, blg.shape, brg.shape

            tl_avg = (tlD + tlA - tlB - tlC) / (tlh * tlw)
            tr_avg = (trD + trA - trB - trC) / (trh * trw)
            bl_avg = (blD + blA - blB - blC) / (blh * blw)
            br_avg = (brD + brA - brB - brC) / (brh * brw)

            edited[:hh, :hw] = tl_avg
            edited[:hh, hw:] = tr_avg
            edited[hh:, :hw] = bl_avg
            edited[hh:, hw:] = br_avg

            if set_border:
                border(edited)

            # The "index" acts as a unique identifier for the quadrant. If the error of two quadrants is the same,
            # "heapq" will attempt to compare the next parameter, which would normally be the actual image quadrant as an
            # np array. This would result in a "truth value of an array is ambiguous" error. Instead, the next parameter
            # is the "index" value, which is unique for every quadrant, which stops this error from occurring.
            heapq.heappush(quadrants, (-error(tlg, tl_avg), index+0, (tl, edited[:hh, :hw], tlg, (tlx, tly))))
            heapq.heappush(quadrants, (-error(trg, tr_avg), index+1, (tr, edited[:hh, hw:], trg, (tlx + hw, tly))))
            heapq.heappush(quadrants, (-error(blg, bl_avg), index+2, (bl, edited[hh:, :hw], blg, (tlx, tly + hh))))
            heapq.heappush(quadrants, (-error(brg, br_avg), index+3, (br, edited[hh:, hw:], brg, (tlx + hw, tly + hh))))

            index += 4

        if quadrants:
            _, _, (image, edited, gray, (tlx, tly)) = heapq.heappop(quadrants)
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
