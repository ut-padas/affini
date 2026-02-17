import argparse
from pathlib import Path

import imageio.v3 as iio


def create_gif_from_pngs(directory_path, output_name="animation.gif", duration=0.5):
    """Create GIF from PNG files in directory."""
    path = Path(directory_path)
    png_files = sorted(path.glob("*.png"))

    if not png_files:
        print("No PNG files found")
        return

    images = [iio.imread(str(png_file)) for png_file in png_files]
    fps = (
        1.0 / duration
    )  # Convert duration (seconds per frame) to fps (frames per second)
    iio.imwrite(str(path / output_name), images, extension=".gif", fps=fps)
    print(f"GIF saved as {path / output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a GIF from PNG files in a specified directory."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Path to the directory containing PNG files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="animation.gif",
        help="Name of the output GIF file.",
    )
    parser.add_argument(
        "-t",
        "--duration",
        type=float,
        default=0.5,
        help="Duration (in seconds) for each frame in the GIF.",
    )
    args = parser.parse_args()
    create_gif_from_pngs(args.directory, args.output, args.duration)
