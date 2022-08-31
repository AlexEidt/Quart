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
from tqdm import trange, tqdm


def border(image):
    # Add a black border around the given image quadrant.
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0


def error(total, squared, n):
    # Compute the error of a given quadrant as the sum of squared error.
    return (squared - (total * total / n)) / n


def quad(image, edited, iterations, quadrants=None, min_width=10, min_height=10, set_border=True):
    """
    Performs the quadtree segmentation algorithm on the image.
    The resulting quadtree image is stored in "edited".
    """
    if quadrants is None:
        quadrants = []

    h, w = image.shape[:2]
    # Create the integral image, edge padded by one to the top and left.
    I = np.pad(image.astype(np.uint32), ((1, 0), (1, 0), (0, 0)), mode='edge')
    np.cumsum(I, axis=0, out=I)
    np.cumsum(I, axis=1, out=I)

    np.multiply(image, np.array([0.299, 0.587, 0.114]), out=image, casting='unsafe')
    np.sum(image, axis=2, dtype=np.uint8, out=image[..., 0])

    gray = np.pad(image[..., 0], ((1, 0), (1, 0)), mode='edge')

    # Create the integral image of all gray values, edge padded by one to the top and left.
    Ig = gray.astype(np.uint64)
    np.cumsum(Ig, axis=0, out=Ig)
    np.cumsum(Ig, axis=1, out=Ig)
    # Create the integral image of all gray values squared, edge padded by one to the top and left.
    Isq = gray.astype(np.uint64)
    np.multiply(Isq, Isq, out=Isq)
    np.cumsum(Isq, axis=0, out=Isq)
    np.cumsum(Isq, axis=1, out=Isq)

    del gray

    # Top left quadrant x and y coordinates.
    x, y = 0, 0

    for _ in trange(iterations):
        if h > min_height and w > min_width:
            hw, hh = w // 2, h // 2

            # Original Image Integral Image bounding box
            tl = I[y+hh, x+hw] + I[y, x] - I[y, x+hw] - I[y+hh, x]          # Top Left
            tr = I[y+hh, x+w] + I[y, x+hw] - I[y, x+w] - I[y+hh, x+hw]      # Top Right
            bl = I[y+h, x+hw] + I[y+hh, x] - I[y+hh, x+hw] - I[y+h, x]      # Bottom Left
            br = I[y+h, x+w] + I[y+hh, x+hw] - I[y+hh, x+w] - I[y+h, x+hw]  # Bottom Right
            # Squared Grayscale Image Integral Image bounding box
            tls = Isq[y+hh, x+hw] + Isq[y, x] - Isq[y, x+hw] - Isq[y+hh, x]
            trs = Isq[y+hh, x+w] + Isq[y, x+hw] - Isq[y, x+w] - Isq[y+hh, x+hw]
            bls = Isq[y+h, x+hw] + Isq[y+hh, x] - Isq[y+hh, x+hw] - Isq[y+h, x]
            brs = Isq[y+h, x+w] + Isq[y+hh, x+hw] - Isq[y+hh, x+w] - Isq[y+h, x+hw]
            # Grayscale Image Integral Image bounding box
            tlg = Ig[y+hh, x+hw] + Ig[y, x] - Ig[y, x+hw] - Ig[y+hh, x]
            trg = Ig[y+hh, x+w] + Ig[y, x+hw] - Ig[y, x+w] - Ig[y+hh, x+hw]
            blg = Ig[y+h, x+hw] + Ig[y+hh, x] - Ig[y+hh, x+hw] - Ig[y+h, x]
            brg = Ig[y+h, x+w] + Ig[y+hh, x+hw] - Ig[y+hh, x+w] - Ig[y+h, x+hw]

            tlw, tlh = hw, hh
            trw, trh = w - hw, hh
            blw, blh = hw, h - hh
            brw, brh = w - hw, h - hh

            edited[y:y+hh, x:x+hw] = tl / (tlw * tlh)
            edited[y:y+hh, x+hw:x+w] = tr / (trw * trh)
            edited[y+hh:y+h, x:x+hw] = bl / (blw * blh)
            edited[y+hh:y+h, x+hw:x+w] = br / (brw * brh)

            if set_border:
                border(edited[y:y+hh, x:x+hw])
                border(edited[y:y+hh, x+hw:x+w])
                border(edited[y+hh:y+h, x:x+hw])
                border(edited[y+hh:y+h, x+hw:x+w])

            heapq.heappush(quadrants, (-error(tlg, tls, tlw * tlh), x, y, tlw, tlh))
            heapq.heappush(quadrants, (-error(trg, trs, trw * trh), x + hw, y, trw, trh))
            heapq.heappush(quadrants, (-error(blg, bls, blw * blh), x, y + hh, blw, blh))
            heapq.heappush(quadrants, (-error(brg, brs, brw * brh), x + hw, y + hh, brw, brh))

        if quadrants:
            _, x, y, w, h = heapq.heappop(quadrants)
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Quadtree Image Segmentation.")
    parser.add_argument("input", type=str, help="Image to segment.")
    parser.add_argument("output", type=str, help="Output filename.")
    parser.add_argument("iterations", type=int, help="Number of segmentation iterations.")
    parser.add_argument("-q", "--quality", type=int, default=5, help="Quality of the output video. (0-10), 0 worst, 10 best.")
    parser.add_argument("-b", "--border", action="store_true", help="Add borders to subimages.")
    parser.add_argument("-a", "--audio", action="store_true", help="Add audio from the input file to the output file.")
    parser.add_argument("-mw", "--minwidth", type=int, default=10, help="Minimum width of the smallest image quadrant.")
    parser.add_argument("-mh", "--minheight", type=int, default=10, help="Minimum height of the smallest image quadrant.")

    return parser.parse_args()


def quadtree_video(args):
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
