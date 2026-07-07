import random
from collections import deque

import numpy as np


class PerlinNoiseGenerator:
    """Generates a single octave of 2D Perlin noise.

    Perlin noise works by scattering random gradient vectors across a coarse
    grid, then for every pixel in the fine output image, blending the
    gradients of the four surrounding coarse grid corners together. The
    result is smooth, natural-looking noise (as opposed to pure random static).
    """

    def __init__(self, random_number_generator):
        # A numpy Generator instance, shared so the whole map uses one seed.
        self.random_number_generator = random_number_generator

    @staticmethod
    def _smoothstep(fraction):
        """Eases a 0..1 value so it accelerates and decelerates smoothly
        instead of blending linearly (this is what makes Perlin noise look
        organic instead of blocky)."""
        return 6 * fraction**5 - 15 * fraction**4 + 10 * fraction**3

    def generate(self, output_height, output_width, grid_rows, grid_cols):
        """Generates one octave of Perlin noise.

        Args:
            output_height: height in pixels of the noise image to produce.
            output_width: width in pixels of the noise image to produce.
            grid_rows: number of coarse gradient-grid rows (fewer = blobbier
                noise, more = finer detail).
            grid_cols: number of coarse gradient-grid columns.

        Returns:
            A (output_height, output_width) numpy array of noise values.
        """
        # Assign a random direction (angle) to every corner of the coarse
        # gradient grid. These are the "gradient vectors" Perlin noise blends.
        corner_angles = 2 * np.pi * self.random_number_generator.random(
            (grid_rows + 1, grid_cols + 1)
        )
        gradient_vectors = np.stack(
            (np.cos(corner_angles), np.sin(corner_angles)), axis=-1
        )

        # For every output pixel, figure out which coarse grid cell it falls
        # inside, and how far across that cell it is (as a 0..1 fraction).
        row_positions = np.linspace(0, grid_rows, output_height, endpoint=False)
        col_positions = np.linspace(0, grid_cols, output_width, endpoint=False)

        row_cell_index = row_positions.astype(int)
        col_cell_index = col_positions.astype(int)
        row_fraction_within_cell = row_positions - row_cell_index
        col_fraction_within_cell = col_positions - col_cell_index

        row_cell_index_grid, col_cell_index_grid = np.meshgrid(
            row_cell_index, col_cell_index, indexing="ij"
        )
        row_fraction_grid, col_fraction_grid = np.meshgrid(
            row_fraction_within_cell, col_fraction_within_cell, indexing="ij"
        )

        def dot_product_with_corner(row_offset, col_offset):
            """Dot product between a cell corner's gradient vector and the
            vector pointing from that corner to the pixel being sampled."""
            corner_row = row_cell_index_grid + row_offset
            corner_col = col_cell_index_grid + col_offset
            corner_gradient = gradient_vectors[corner_row, corner_col]
            vector_to_pixel = np.stack(
                (row_fraction_grid - row_offset, col_fraction_grid - col_offset),
                axis=-1,
            )
            return np.sum(corner_gradient * vector_to_pixel, axis=-1)

        top_left = dot_product_with_corner(0, 0)
        bottom_left = dot_product_with_corner(1, 0)
        top_right = dot_product_with_corner(0, 1)
        bottom_right = dot_product_with_corner(1, 1)

        # Blend the four corner values together, smoothly, based on how far
        # across the cell the pixel is.
        horizontal_ease = self._smoothstep(col_fraction_grid)
        vertical_ease = self._smoothstep(row_fraction_grid)

        top_blend = top_left * (1 - horizontal_ease) + top_right * horizontal_ease
        bottom_blend = (
            bottom_left * (1 - horizontal_ease) + bottom_right * horizontal_ease
        )
        return top_blend * (1 - vertical_ease) + bottom_blend * vertical_ease


class FractalNoiseGenerator:
    """Combines several octaves of Perlin noise at increasing detail levels
    (fractal/"fBm" noise). Low octaves create the broad island shapes, high
    octaves add coastline roughness and small-scale detail on top."""

    def __init__(self, random_number_generator):
        self.perlin_noise_generator = PerlinNoiseGenerator(random_number_generator)

    def generate(
        self,
        height,
        width,
        base_grid_rows=8,
        base_grid_cols=8,
        octave_count=5,
        persistence=0.5,
        lacunarity=2.0,
    ):
        """
        Args:
            height, width: size of the noise image to produce, in pixels.
            base_grid_rows, base_grid_cols: coarse gradient-grid size for the
                very first (lowest detail) octave.
            octave_count: how many detail layers to stack.
            persistence: how much quieter each successive octave is
                (0.5 = each octave contributes half as much as the last).
            lacunarity: how much finer each successive octave's grid is
                (2.0 = each octave doubles the gradient-grid resolution).
        """
        combined_noise = np.zeros((height, width))
        octave_amplitude = 1.0
        total_amplitude_used = 0.0

        for octave_index in range(octave_count):
            grid_rows = int(base_grid_rows * lacunarity**octave_index)
            grid_cols = int(base_grid_cols * lacunarity**octave_index)
            octave_noise = self.perlin_noise_generator.generate(
                height, width, grid_rows, grid_cols
            )
            combined_noise += octave_amplitude * octave_noise
            total_amplitude_used += octave_amplitude
            octave_amplitude *= persistence

        return combined_noise / total_amplitude_used


class ArchipelagoMap:
    """Represents one generated map: its terrain grid, the raw elevation
    values behind it, and a guaranteed-reachable ship start/goal pair.

    Terrain types (from lowest elevation to highest):
        "water"    - open sea, the only terrain a ship can sail through.
        "beach"    - shallow water / sandbar, a hazard the ship cannot cross.
        "land"     - regular dry land.
        "mountain" - high elevation land.
    """

    # A ship may only travel across these terrain types.
    NAVIGABLE_TERRAIN_TYPES = {"water"}

    def __init__(
        self,
        size=40,
        sea_level=0.25,
        seed=None,
        minimum_open_water_fraction=0.35,
        island_scatter=8,
        edge_avoidance_weight=0.15,
    ):
        """
        Args:
            size: the map is a square grid of this many tiles per side.
            sea_level: elevation threshold; tiles above this are land.
                Higher values mean less land overall, which (combined with
                island_scatter) is what breaks one big landmass into several
                smaller separate islands.
            seed: an integer to reproduce the exact same map later, or None
                to generate a brand-new random map every time.
            minimum_open_water_fraction: if the largest connected body of
                open water ends up smaller than this fraction of the whole
                map, the map is regenerated (guards against maps that are
                mostly landlocked ponds with barely any sailable area).
            island_scatter: coarseness of the base noise layer. Low values
                (e.g. 4) produce one large landmass; higher values (e.g. 8-12)
                produce several smaller, separate islands spread across the
                map.
            edge_avoidance_weight: how strongly islands are discouraged from
                touching the map border, from 0.0 (no effect, islands can
                spawn anywhere including the edges) to 1.0 (strong pull
                toward a single central landmass). Keep this low for a
                scattered archipelago look.
        """
        self.size = size
        self.sea_level = sea_level
        self.minimum_open_water_fraction = minimum_open_water_fraction
        self.island_scatter = island_scatter
        self.edge_avoidance_weight = edge_avoidance_weight

        # A seed of None means "pick a fresh, unpredictable seed every run".
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        self.seed = seed

        self.terrain_grid = None
        self.elevation_map = None
        self.ship_start_position = None
        self.ship_goal_position = None

        self._generate()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self):
        """Builds the elevation map, classifies it into terrain, and picks a
        reachable start/goal pair. Regenerates with a new random seed if the
        result doesn't have enough open water to be navigable."""
        random_number_generator = np.random.default_rng(self.seed)

        detail_noise = FractalNoiseGenerator(random_number_generator).generate(
            self.size,
            self.size,
            base_grid_rows=self.island_scatter,
            base_grid_cols=self.island_scatter,
        )
        detail_noise = self._normalize_to_range(detail_noise, -1, 1)

        # A mask that is high in the center of the map and fades to zero at
        # the edges. Weighted lightly (edge_avoidance_weight) it just keeps
        # islands from spawning right on the map border; weighted heavily it
        # would pull everything into one central landmass instead of a
        # scattered archipelago.
        island_mask = self._build_radial_island_mask()

        self.elevation_map = (
            detail_noise * (1 - self.edge_avoidance_weight)
            + (island_mask * 2 - 1) * self.edge_avoidance_weight
        )
        self.terrain_grid = self._classify_terrain(self.elevation_map)

        largest_water_region = self._find_largest_connected_water_region()
        map_area = self.size * self.size
        if (
            largest_water_region is None
            or len(largest_water_region) < self.minimum_open_water_fraction * map_area
        ):
            # This map didn't produce enough sailable water (e.g. everything
            # got carved into disconnected ponds) - try again with a new seed.
            self.seed = random.SystemRandom().randint(0, 2**31 - 1)
            self._generate()
            return

        self.ship_start_position, self.ship_goal_position = (
            self._choose_start_and_goal(largest_water_region, random_number_generator)
        )

    @staticmethod
    def _normalize_to_range(values, new_min, new_max):
        """Rescales an array so its values span exactly [new_min, new_max]."""
        old_min, old_max = values.min(), values.max()
        unit_scaled = (values - old_min) / (old_max - old_min)
        return unit_scaled * (new_max - new_min) + new_min

    def _build_radial_island_mask(self, falloff_strength=2.0):
        """Builds a (size, size) array that is 1.0 at the center of the map
        and fades toward 0.0 at the edges, based on distance from center."""
        row_coords, col_coords = np.mgrid[0 : self.size, 0 : self.size]
        center_row, center_col = self.size / 2, self.size / 2

        normalized_distance_from_center = np.sqrt(
            ((row_coords - center_row) / (self.size / 2)) ** 2
            + ((col_coords - center_col) / (self.size / 2)) ** 2
        )
        normalized_distance_from_center = np.clip(
            normalized_distance_from_center, 0, 1
        )
        return 1 - normalized_distance_from_center**falloff_strength

    def _classify_terrain(self, elevation_map):
        """Converts raw elevation floats into named terrain types."""
        terrain_grid = np.full(elevation_map.shape, "water", dtype=object)

        is_above_sea_level = elevation_map > self.sea_level
        is_beach = (~is_above_sea_level) & (
            elevation_map > self.sea_level - 0.15
        )
        is_mountain = is_above_sea_level & (elevation_map > self.sea_level + 0.35)

        terrain_grid[is_above_sea_level] = "land"
        terrain_grid[is_beach] = "beach"
        terrain_grid[is_mountain] = "mountain"
        return terrain_grid

    # ------------------------------------------------------------------
    # Water connectivity / pathing
    # ------------------------------------------------------------------

    def _find_largest_connected_water_region(self):
        """Flood-fills every group of connected navigable tiles and returns
        the largest such group, as a set of (row, col) tuples. Returns None
        if there are no navigable tiles at all."""
        all_regions = self._find_all_connected_water_regions()
        if not all_regions:
            return None
        return max(all_regions, key=len)

    def _find_all_connected_water_regions(self):
        """Flood-fills the whole map and returns every separate connected
        group of navigable tiles as a list of sets of (row, col) tuples."""
        visited = np.zeros((self.size, self.size), dtype=bool)
        regions = []

        for start_row in range(self.size):
            for start_col in range(self.size):
                already_visited = visited[start_row, start_col]
                is_navigable = self.terrain_grid[start_row, start_col] in (
                    self.NAVIGABLE_TERRAIN_TYPES
                )
                if already_visited or not is_navigable:
                    continue

                region_tiles = set()
                tiles_to_visit = deque([(start_row, start_col)])
                visited[start_row, start_col] = True

                while tiles_to_visit:
                    current_row, current_col = tiles_to_visit.popleft()
                    region_tiles.add((current_row, current_col))
                    for neighbor_row, neighbor_col in self._get_neighbors(
                        current_row, current_col
                    ):
                        neighbor_is_navigable = self.terrain_grid[
                            neighbor_row, neighbor_col
                        ] in self.NAVIGABLE_TERRAIN_TYPES
                        if not visited[neighbor_row, neighbor_col] and neighbor_is_navigable:
                            visited[neighbor_row, neighbor_col] = True
                            tiles_to_visit.append((neighbor_row, neighbor_col))

                regions.append(region_tiles)

        return regions

    def _get_neighbors(self, row, col):
        """Yields the in-bounds up/down/left/right neighbors of a tile."""
        for row_delta, col_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_row, neighbor_col = row + row_delta, col + col_delta
            if 0 <= neighbor_row < self.size and 0 <= neighbor_col < self.size:
                yield neighbor_row, neighbor_col

    def path_exists(self, start_position, goal_position):
        """Returns True if a ship could sail from start_position to
        goal_position using only navigable tiles."""
        tiles_to_visit = deque([start_position])
        visited_tiles = {start_position}

        while tiles_to_visit:
            current_row, current_col = tiles_to_visit.popleft()
            if (current_row, current_col) == goal_position:
                return True
            for neighbor_row, neighbor_col in self._get_neighbors(
                current_row, current_col
            ):
                neighbor_position = (neighbor_row, neighbor_col)
                neighbor_is_navigable = self.terrain_grid[
                    neighbor_row, neighbor_col
                ] in self.NAVIGABLE_TERRAIN_TYPES
                if neighbor_position not in visited_tiles and neighbor_is_navigable:
                    visited_tiles.add(neighbor_position)
                    tiles_to_visit.append(neighbor_position)

        return False

    def _choose_start_and_goal(self, navigable_region, random_number_generator, sample_attempts=40):
        """Picks two tiles from within a single connected navigable region,
        favoring pairs that are far apart (in straight-line distance) so the
        navigation task is non-trivial. Because both tiles come from the same
        connected region, a valid path between them is guaranteed to exist.

        Uses the map's own seeded random_number_generator (rather than the
        global `random` module) so that two ArchipelagoMap instances built
        with the same seed produce the exact same start/goal pair."""
        candidate_tiles = list(navigable_region)

        best_pair = None
        best_squared_distance = -1
        for _ in range(sample_attempts):
            sampled_indices = random_number_generator.choice(
                len(candidate_tiles), size=2, replace=False
            )
            tile_a = candidate_tiles[sampled_indices[0]]
            tile_b = candidate_tiles[sampled_indices[1]]
            squared_distance = (tile_a[0] - tile_b[0]) ** 2 + (tile_a[1] - tile_b[1]) ** 2
            if squared_distance > best_squared_distance:
                best_squared_distance = squared_distance
                best_pair = (tile_a, tile_b)

        return best_pair

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def render(self, use_emoji=True):
        """Prints the map to the console, marking the ship's start and goal
        positions."""
        if use_emoji:
            # Plain colored-square emoji only: these are fixed single-width
            # glyphs, so columns stay aligned in a monospace terminal (unlike
            # emoji such as a wave or a ship, which render at inconsistent
            # widths and break grid alignment).
            terrain_symbols = {
                "water": "🟦",
                "beach": "🟨",
                "land": "🟩",
                "mountain": "🟫",
            }
            ship_symbol, goal_symbol = "🟥", "🟪"
        else:
            terrain_symbols = {"water": "~", "beach": ".", "land": "#", "mountain": "^"}
            ship_symbol, goal_symbol = "S", "G"

        for row in range(self.size):
            row_characters = []
            for col in range(self.size):
                position = (row, col)
                if position == self.ship_start_position:
                    row_characters.append(ship_symbol)
                elif position == self.ship_goal_position:
                    row_characters.append(goal_symbol)
                else:
                    row_characters.append(terrain_symbols[self.terrain_grid[row, col]])
            print("".join(row_characters))


if __name__ == "__main__":
    archipelago_map = ArchipelagoMap()
    print(
        f"seed={archipelago_map.seed}  "
        f"start={archipelago_map.ship_start_position}  "
        f"goal={archipelago_map.ship_goal_position}"
    )
    archipelago_map.render()
