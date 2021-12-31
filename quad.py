"""
Quadtree Image Segmentation
Alex Eidt
"""

import argparse
import imageio
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
    image[half_h:half_h+1, :] = 0
    # Vertical Line
    image[:, half_w:half_w+1] = 0


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


def error(image, error_type='sse'):
    """
    Compute the error of a given quadrant.
    """
    if image.size == 0:
        return 0
    # Grayscale Image
    image = np.sum(image * np.array([0.299, 0.587, 0.114]), axis=2)
    if error_type == 'sse': # Sum of Squared Errors
        return np.sum((image - image.mean()) ** 2)
    elif error_type == 'minmax': # Max Difference
        return image.max() - image.min()
    else: # Max Difference of mean
        return np.max(np.abs(image - image.mean()))


def quad(iterations, image, edited, quads, set_border=True, error_type='sse'):
    """
    Split the given image into four quadrants.
    Update the edited image by coloring in the newly split quadrants to the average rgb
    color of the original image.
    Find the quadrant with the maximum error, remove it from the "quads" list and return it. 
    """
    if iterations < 0:
        return image, edited
    for _ in range(iterations):
        h, w = image.shape[:2]
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

        quads.append((error(top_left, error_type=error_type), top_left, edited[:half_h, :half_w]))
        quads.append((error(top_right, error_type=error_type), top_right, edited[:half_h, half_w:]))
        quads.append((error(bottom_left, error_type=error_type), bottom_left, edited[half_h:, :half_w]))
        quads.append((error(bottom_right, error_type=error_type), bottom_right, edited[half_h:, half_w:]))

        data = max(quads, key=lambda x: x[0])
        quads.remove(data)

        image = data[1]
        edited = data[2]

    return data[1:]


def main():
    parser = argparse.ArgumentParser(description='Quadtree Image Segmentation.')
    parser.add_argument('input', type=str, help='Image to segment.')
    parser.add_argument('output', type=str, help='Output filename.')
    parser.add_argument('-fps', type=int, default=1, help='Output FPS.')
    parser.add_argument('-q', '--quality', type=int, default=5, help='Quality of the output video.')
    parser.add_argument('-g', '--gif', action='store_true', help='Output as gif.')
    parser.add_argument('-i', '--iterations', type=int, default=12, help='Number of iterations.')
    parser.add_argument('-b', '--border', action='store_true', help='Add borders to subimages.')
    parser.add_argument('-img', '--image', action='store_true', help='Save final output image.')
    parser.add_argument('-s', '--step', type=int, default=2, help='Only save a frame every `(iteration)^s` iterations.')
    parser.add_argument('-e', '--error', type=str, default='sse', help='Error type: Sum of Squared Error (sse), Min-Max Difference (minmax) or Max Difference (max).')
    parser.add_argument('-f', '--frames', action='store_true', help='Save frames.')
    args = parser.parse_args()

    # Try to load an image from the given input. If this fails, assume it's a video.
    try:
        image = imageio.imread(args.input)
    except Exception:
        # Convert every frame of input video to quadtree image and store as output video.
        with imageio.read(args.input) as video:
            data = video.get_meta_data()
            with imageio.save(args.output, fps=data['fps'], quality=min(max(args.quality, 0), 10)) as writer:
                quads = []
                for frame in tqdm(video, total=int(data['duration'] * data['fps'])):
                    copy = frame.copy()
                    edited = copy
                    quad(args.iterations, frame, edited, quads, set_border=args.border, error_type=args.error)
                    writer.append_data(edited)
                    quads.clear()
    else:
        # Convert input image to quadtree image and save as output image.
        copy = image.copy()
        edited = copy

        quads = []

        if args.gif:
            with imageio.save(args.output, fps=args.fps) as writer:
                for i in tqdm(range(args.iterations)):
                    image, edited = quad(args.step ** i, image, edited, quads, set_border=args.border, error_type=args.error)

                    writer.append_data(copy)
                    if args.frames:
                        imageio.imsave(f'{args.output.rsplit(".", 1)[0]}_{i}.png', copy)

            if args.image:
                imageio.imsave(f'{args.output.rsplit(".", 1)[0]}_quad.png', copy)
        else:
            quad(args.iterations, image, edited, quads, set_border=args.border, error_type=args.error)
            imageio.imsave(args.output, copy)


if __name__ == '__main__':
    main()