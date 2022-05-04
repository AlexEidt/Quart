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


def rgb_mean(image):
    """
    Compute the mean rgb value of the given image quadrant.
    """
    if image.size == 0:
        return 0, 0, 0
    r = np.uint8(image[..., 0].mean())
    g = np.uint8(image[..., 1].mean())
    b = np.uint8(image[..., 2].mean())
    return r, g, b


def error(image):
    """
    Compute the error of a given quadrant.
    """
    if image.size == 0:
        return 0
    h, w = image.shape[:2]
    return np.sum((image - image.mean()) ** 2) / (h * w)


def quad(iterations, image, edited, min_width=10, min_height=10, set_border=True):
    """
    Split the given image into four quadrants.
    Update the edited image by coloring in the newly split quadrants to the average rgb
    color of the original image.
    Find the quadrant with the maximum error, remove it from the "quads" list and return it.
    """
    if iterations <= 0:
        return image, edited

    quads = []
    grayscaled = (image * np.array([0.299, 0.587, 0.114])).sum(axis=2)
    gray = grayscaled  # Intensity

    for _ in range(iterations):
        h, w = image.shape[:2]

        if h > min_height and w > min_width:
            half_w = w // 2
            half_h = h // 2

            tl = image[:half_h, :half_w]
            tr = image[:half_h, half_w:]
            bl = image[half_h:, :half_w]
            br = image[half_h:, half_w:]

            tlI = gray[:half_h, :half_w]
            trI = gray[:half_h, half_w:]
            blI = gray[half_h:, :half_w]
            brI = gray[half_h:, half_w:]

            edited[:half_h, :half_w] = rgb_mean(tl)
            edited[:half_h, half_w:] = rgb_mean(tr)
            edited[half_h:, :half_w] = rgb_mean(bl)
            edited[half_h:, half_w:] = rgb_mean(br)

            if set_border:
                border(edited)

            heapq.heappush(quads, (-error(tlI), (tl, edited[:half_h, :half_w], tlI)))
            heapq.heappush(quads, (-error(trI), (tr, edited[:half_h, half_w:], trI)))
            heapq.heappush(quads, (-error(blI), (bl, edited[half_h:, :half_w], blI)))
            heapq.heappush(quads, (-error(brI), (br, edited[half_h:, half_w:], brI)))

        if quads:
            _, (image, edited, gray) = heapq.heappop(quads)
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Quadtree Image Segmentation.")
    parser.add_argument("input", type=str, help="Image to segment.")
    parser.add_argument("output", type=str, help="Output filename.")
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=5,
        help="Quality of the output video. (0-10), 0 worst, 10 best.",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=12, help="Number of iterations."
    )
    parser.add_argument(
        "-b", "--border", action="store_true", help="Add borders to subimages."
    )
    parser.add_argument(
        "-au",
        "--audio",
        action="store_true",
        help="Add audio from the input file to the output file.",
    )
    parser.add_argument(
        "-mw",
        "--minwidth",
        type=int,
        default=10,
        help="Minimum width of the smallest image quadrant.",
    )
    parser.add_argument(
        "-mh",
        "--minheight",
        type=int,
        default=10,
        help="Minimum height of the smallest image quadrant.",
    )

    return parser.parse_args()


def create_video(args):
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

        for frame in tqdm(video, total=int(data["fps"] * data["duration"] + 0.5)):
            edited = frame.copy()
            quad(
                args.iterations,
                frame,
                edited,
                min_width=args.minwidth,
                min_height=args.minheight,
                set_border=args.border,
            )
            writer.send(edited)

        writer.close()


def main():
    args = parse_args()

    # Try to load an image from the given input. If this fails, assume it's a video.
    try:
        image = imageio.imread(args.input)[..., :3]
    except Exception:
        create_video(args)
    else:
        # Convert input image to quadtree image and save as output image.
        copy = image.copy()
        quad(
            args.iterations,
            image,
            copy,
            min_width=args.minwidth,
            min_height=args.minheight,
            set_border=args.border,
        )
        imageio.imsave(args.output, copy)


if __name__ == "__main__":
    main()
