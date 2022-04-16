"""
Quadtree Image Segmentation
Alex Eidt

Split the image into four quadrants.
Find the quadrant with the highest error.
Split that quadrant into four quadrants.
Repeat N times.
"""

import argparse
import imageio
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm

MIN_WIDTH = 10
MIN_HEIGHT = 10


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
    r = image[:, :, 0].mean()
    g = image[:, :, 1].mean()
    b = image[:, :, 2].mean()
    return r, g, b


def error(image):
    """
    Compute the error of a given quadrant.
    """
    if image.size == 0:
        return 0
    # Grayscale Image
    h, w = image.shape[:2]
    image = np.sum(image * np.array([0.299, 0.587, 0.114]), axis=2)
    return np.sum((image - image.mean()) ** 2) / (h * w)


def quad(iterations, image, edited, quads, set_border=True):
    """
    Split the given image into four quadrants.
    Update the edited image by coloring in the newly split quadrants to the average rgb
    color of the original image.
    Find the quadrant with the maximum error, remove it from the "quads" list and return it.
    """
    if iterations <= 0:
        return image, edited
    oh, ow = image.shape[:2]
    for _ in range(iterations):
        h, w = image.shape[:2]

        if h > MIN_HEIGHT and w > MIN_WIDTH:
            half_w = w // 2
            half_h = h // 2

            top_left = image[:half_h, :half_w]
            top_right = image[:half_h, half_w:]
            bottom_left = image[half_h:, :half_w]
            bottom_right = image[half_h:, half_w:]

            edited[:half_h, :half_w] = rgb_mean(top_left)
            edited[:half_h, half_w:] = rgb_mean(top_right)
            edited[half_h:, :half_w] = rgb_mean(bottom_left)
            edited[half_h:, half_w:] = rgb_mean(bottom_right)

            if set_border:
                border(edited)

            quads.append((error(top_left), top_left, edited[:half_h, :half_w]))
            quads.append((error(top_right), top_right, edited[:half_h, half_w:]))
            quads.append((error(bottom_left), bottom_left, edited[half_h:, :half_w]))
            quads.append((error(bottom_right), bottom_right, edited[half_h:, half_w:]))

        if quads:
            data = max(quads, key=lambda x: x[0])
            quads.remove(data)

            image = data[1]
            edited = data[2]
        else:
            break

    return data[1:]


def main():
    parser = argparse.ArgumentParser(description="Quadtree Image Segmentation.")
    parser.add_argument("input", type=str, help="Image to segment.")
    parser.add_argument("output", type=str, help="Output filename.")
    parser.add_argument("-fps", type=int, default=1, help="Output FPS.")
    parser.add_argument(
        "-q", "--quality", type=int, default=5, help="Quality of the output video."
    )
    parser.add_argument(
        "-a",
        "--animate",
        action="store_true",
        help="Save intermediary frames as video.",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=12, help="Number of iterations."
    )
    parser.add_argument(
        "-b", "--border", action="store_true", help="Add borders to subimages."
    )
    parser.add_argument(
        "-img", "--image", action="store_true", help="Save final output image."
    )
    parser.add_argument(
        "-s",
        "--step",
        type=float,
        default=2.0,
        help="Only save a frame every `s^(iteration)` iterations. For use with --animate only.",
    )
    parser.add_argument(
        "-f",
        "--frames",
        action="store_true",
        help="Save intermediary frames as images.",
    )
    parser.add_argument(
        "-au",
        "--audio",
        action="store_true",
        help="Add audio from the input file to the output file.",
    )
    args = parser.parse_args()

    # Try to load an image from the given input. If this fails, assume it's a video.
    try:
        image = imageio.imread(args.input)[:, :, :3]
    except Exception:
        # Convert every frame of input video to quadtree image and store as output video.
        with imageio.read(args.input) as video:
            data = video.get_meta_data()

            kwargs = {"fps": data["fps"], "quality": min(max(args.quality, 0), 10)}
            if args.audio:
                kwargs["audio_path"] = args.input

            writer = imageio_ffmpeg.write_frames(
                args.output, data["source_size"], **kwargs
            )
            writer.send(None)

            quads = []
            for frame in tqdm(video, total=int(data["fps"] * data["duration"] + 0.5)):
                copy = frame.copy()
                edited = copy
                quad(args.iterations, frame, edited, quads, set_border=args.border)
                writer.send(edited)
                quads.clear()

            writer.close()

    else:
        # Convert input image to quadtree image and save as output image.
        copy = image.copy()
        edited = copy

        quads = []

        if args.animate:
            with imageio.save(args.output, fps=args.fps) as writer:
                for i in tqdm(range(args.iterations)):
                    image, edited = quad(
                        int(args.step**i),
                        image,
                        edited,
                        quads,
                        set_border=args.border,
                    )

                    writer.append_data(copy)
                    if args.frames:
                        imageio.imsave(f'{args.output.rsplit(".", 1)[0]}_{i}.png', copy)

            if args.image:
                imageio.imsave(f'{args.output.rsplit(".", 1)[0]}_quad.png', copy)
        else:
            quad(args.iterations, image, edited, quads, set_border=args.border)
            imageio.imsave(args.output, copy)


if __name__ == "__main__":
    main()
