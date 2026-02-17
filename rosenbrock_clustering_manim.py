import os

import matplotlib.pyplot as plt
import numpy as np
from manim import *
from matplotlib.colors import LogNorm
from tqdm import tqdm

from src.adaptive_clustering import adaptive_clustering
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

DOMAIN = [-2, 3]
N = 10_000
SEED = 0
TOL = 0.1

np.random.seed(SEED)
X_train = np.random.uniform(DOMAIN[0], DOMAIN[1], size=(N, 2))

result = adaptive_clustering(
    X=X_train,
    f_fn=rosenbrock,
    g_fn=rosenbrock_grad,
    TOL=TOL,
    ERROR_METRIC="Linf-local",
    NUM_INITIAL_CLUSTERS=10,
    STORE_TREE_DATA=True,
    USE_HESSIAN_INFORMED_SPLIT=True,
    INITIAL_CLUSTERING_METHOD="kmeans",
)
tree_data = result["tree_data"]


class RosenbrockAdaptiveClustering(Scene):
    def construct(self):
        # Set up coordinate system
        axes = Axes(
            x_range=[DOMAIN[0], DOMAIN[1], 1],
            y_range=[DOMAIN[0], DOMAIN[1], 1],
            x_length=8,
            y_length=6,
            axis_config={"color": WHITE},
        )
        axes_labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")

        # Create background contour for Rosenbrock function
        self.create_rosenbrock_contour(axes)

        # Add title (positioned higher to avoid covering axis labels)
        title = Text("Adaptive Clustering Animation", font_size=32)
        title.to_edge(UP, buff=0.1)

        # Add axes and title
        self.play(Create(axes), Write(axes_labels), Write(title))

        # Colors for different levels
        level_colors = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE]

        # Storage for current level objects
        current_level_objects = None
        current_label = None

        # Show all levels
        levels_to_show = list(tree_data.keys())  # Show all levels

        # Animate through each level
        for counter, level in tqdm(
            enumerate(levels_to_show),
            desc="Animating levels",
            total=len(levels_to_show),
        ):
            level_data = tree_data[level]
            centroids = level_data["centroids"]
            Qs = level_data["Q"]

            # Note: Previous levels persist as requested

            color = level_colors[counter % len(level_colors)]

            # Create level label
            level_label = Text(
                f"Level {level} ({len(centroids)} clusters)", font_size=24, color=color
            )
            level_label.to_corner(DL, buff=0.5)

            # Show level label
            if current_label is not None:
                self.play(Transform(current_label, level_label), run_time=0.2)
            else:
                self.play(Write(level_label), run_time=0.2)
                current_label = level_label

            # Create centroids as dots
            centroid_dots = VGroup()
            for i, centroid in enumerate(centroids):
                dot = Dot(
                    point=axes.coords_to_point(centroid[0], centroid[1]),
                    color=color,
                    radius=0.06,
                )
                centroid_dots.add(dot)

            # Create ellipses
            ellipse_group = VGroup()
            for i, (centroid, Q) in enumerate(zip(centroids, Qs)):
                ellipse_mob = self.create_trust_ellipse(axes, centroid, Q, color)
                ellipse_group.add(ellipse_mob)

            # Define current level objects
            current_level_objects = VGroup(centroid_dots, ellipse_group)

            # Animate appearance of centroids
            self.play(
                LaggedStart(
                    *[GrowFromCenter(dot) for dot in centroid_dots], lag_ratio=0.02
                ),
                run_time=0.3,
            )

            # Animate appearance of ellipses
            self.play(
                LaggedStart(
                    *[Create(ellipse) for ellipse in ellipse_group], lag_ratio=0.02
                ),
                run_time=0.3,
            )

            # Pause to show the level
            self.wait(0.1)

        # Final pause to show complete result
        self.wait(2)

        # Show summary
        summary = Text("Adaptive Clustering Complete!", font_size=32, color=YELLOW)
        summary.to_edge(DOWN)
        self.play(Write(summary))
        self.wait(2)

    def create_trust_ellipse(self, axes, mu, Q, color):
        """Create a trust region ellipse"""
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Q)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Calculate ellipse parameters
        axes_lengths = 2 / np.sqrt(eigvals + 1e-12)
        width, height = axes_lengths

        # Calculate rotation angle
        v0 = eigvecs[:, 0]
        angle_deg = np.degrees(np.arctan2(v0[1], v0[0]))
        angle_rad = np.radians(angle_deg)

        # Create ellipse with proper scaling
        ellipse = Ellipse(
            width=width,
            height=height,
            color=color,
            fill_opacity=0.1,
            stroke_opacity=0.9,
            stroke_width=2,
        )

        # Position and rotate
        ellipse.move_to(axes.coords_to_point(mu[0], mu[1]))
        ellipse.rotate(angle_rad)

        return ellipse

    def create_rosenbrock_contour(self, axes):
        """Create background contour for Rosenbrock function"""
        # Generate contour plot using matplotlib
        x_vals = np.linspace(DOMAIN[0], DOMAIN[1], 200)
        y_vals = np.linspace(DOMAIN[0], DOMAIN[1], 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        Z = rosenbrock(points).reshape(X.shape)

        # Plot to a figure
        fig = plt.figure(figsize=(10, 10))
        # Use full figure area
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # type: ignore
        ax.set_axis_off()
        fig.add_axes(ax)

        # Set limits to match DOMAIN exactly so the image matches the coordinates
        ax.set_xlim(DOMAIN[0], DOMAIN[1])
        ax.set_ylim(DOMAIN[0], DOMAIN[1])

        # Contourf
        ax.contourf(X, Y, Z, levels=50, cmap="viridis", norm=LogNorm())

        # Save to file
        img_path = os.path.join("media", "images", "rosenbrock_contour.png")

        plt.savefig(img_path, pad_inches=0, dpi=500)
        plt.close(fig)
        print(f"Saved contour image to: {img_path}")

        # Create ImageMobject
        contour_img = ImageMobject(img_path)

        # Resize image to match the axes in the scene
        contour_img.stretch_to_fit_width(axes.x_length)
        contour_img.stretch_to_fit_height(axes.y_length)

        contour_img.move_to(axes.get_center())
        contour_img.set_opacity(0.8)

        # Add background grid for better visualization (without duplicate axes)
        grid = NumberPlane(
            x_range=[DOMAIN[0], DOMAIN[1], 0.5],
            y_range=[DOMAIN[0], DOMAIN[1], 0.5],
            x_length=axes.x_length,
            y_length=axes.y_length,
            background_line_style={
                "stroke_color": GRAY,
                "stroke_width": 0.5,
                "stroke_opacity": 0.3,
            },
            axis_config={"stroke_width": 0},  # Hide duplicate axes
        )
        grid.move_to(axes.get_center())

        self.add(contour_img, grid)
