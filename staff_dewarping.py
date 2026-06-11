import cv2
import numpy as np
import PIL.Image

from gtrs.debug import Debug
from gtrs.model import TabStaff
from gtrs.simple_logging import eprint
from gtrs.type_definitions import NDArray


class DelaunayTriangulation:
    def __init__(self, points: NDArray) -> None:
        self.points = np.array(points, dtype=np.float32)
        self.simplices: NDArray | None = None
        self._triangulate()

    def _triangulate(self) -> None:
        points = self.points
        n = len(points)
        if n < 3:
            raise ValueError("Need at least 3 points for triangulation")

        rect = cv2.boundingRect(points.reshape(-1, 1, 2))
        rect = (rect[0] - 10, rect[1] - 10, rect[2] + 20, rect[3] + 20)
        subdiv = cv2.Subdiv2D(rect)

        for point in points:
            subdiv.insert((float(point[0]), float(point[1])))

        triangle_list = subdiv.getTriangleList()
        triangles = []

        for t in triangle_list:
            pt1 = np.array([t[0], t[1]], dtype=np.float32)
            pt2 = np.array([t[2], t[3]], dtype=np.float32)
            pt3 = np.array([t[4], t[5]], dtype=np.float32)

            idx1 = self._find_point_index(pt1, points)
            idx2 = self._find_point_index(pt2, points)
            idx3 = self._find_point_index(pt3, points)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangles.append([idx1, idx2, idx3])

        self.simplices = np.array(triangles, dtype=np.int32)

    def _find_point_index(
        self, point: NDArray, points: NDArray, tolerance: float = 1e-3
    ) -> int | None:
        distances = np.linalg.norm(points - point, axis=1)
        min_idx = int(np.argmin(distances))
        if distances[min_idx] < tolerance:
            return min_idx
        return None

    def find_simplex(self, points: NDArray) -> NDArray:
        if self.simplices is None:
            return np.full(len(points), -1, dtype=np.int32)

        points = np.atleast_2d(points)
        result = np.full(len(points), -1, dtype=np.int32)

        for i, point in enumerate(points):
            for j, simplex in enumerate(self.simplices):
                triangle = self.points[simplex]
                if self._point_in_triangle(point, triangle):
                    result[i] = j
                    break

        return result

    def _point_in_triangle(self, point: NDArray, triangle: NDArray) -> bool:
        def sign(p1: NDArray, p2: NDArray, p3: NDArray) -> float:
            return float((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))

        d1 = sign(point, triangle[0], triangle[1])
        d2 = sign(point, triangle[1], triangle[2])
        d3 = sign(point, triangle[2], triangle[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)


class PiecewiseAffineTransform:
    def __init__(
        self,
        source_points: NDArray,
        target_points: NDArray,
        triangulation: DelaunayTriangulation,
    ) -> None:
        self.source_points = source_points
        self.target_points = target_points
        self.triangulation = triangulation
        self.transforms: list[NDArray] = []
        self._compute_transforms()

    def _compute_transforms(self) -> None:
        if self.triangulation.simplices is None:
            return

        for simplex in self.triangulation.simplices:
            src_tri = self.source_points[simplex].astype(np.float32)
            dst_tri = self.target_points[simplex].astype(np.float32)
            M = cv2.getAffineTransform(src_tri, dst_tri)
            self.transforms.append(M)

    def transform_image(self, image: NDArray) -> NDArray:
        if len(self.transforms) == 0:
            return image

        h, w = image.shape[:2]
        result = np.full_like(image, 255)

        for i, simplex in enumerate(self.triangulation.simplices):
            dst_tri = self.target_points[simplex].astype(np.float32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), 1)

            src_tri = self.source_points[simplex].astype(np.float32)
            M_inv = cv2.getAffineTransform(dst_tri, src_tri)

            warped = cv2.warpAffine(
                image, M_inv, (w, h), borderMode=cv2.BORDER_REPLICATE
            )

            result[mask == 1] = warped[mask == 1]

        return result


class TabStaffDewarping:
    def __init__(self, staff: TabStaff) -> None:
        self.staff = staff

    def dewarp(self, image: NDArray) -> NDArray:
        try:
            source_points = self._get_staff_control_points()
            target_points = self._get_straight_control_points(source_points)

            if len(source_points) < 3:
                eprint("WARN_DEWARP_FAILED: Not enough control points")
                return image

            triangulation = DelaunayTriangulation(source_points)
            transform = PiecewiseAffineTransform(
                source_points, target_points, triangulation
            )
            return transform.transform_image(image)
        except Exception as e:
            eprint(f"WARN_DEWARP_FAILED: {e}")
            return image

    def _get_staff_control_points(self) -> NDArray:
        points = []
        for grid_point in self.staff.grid:
            for y in grid_point.y:
                points.append([grid_point.x, y])
        return np.array(points, dtype=np.float32)

    def _get_straight_control_points(self, source: NDArray) -> NDArray:
        target = source.copy()
        if len(self.staff.grid) > 0 and len(self.staff.grid[0].y) > 0:
            y_coords = self.staff.grid[0].y
            spacing = np.diff(y_coords)
            if len(spacing) > 0:
                mean_spacing = float(np.mean(spacing))
                for i in range(len(self.staff.grid)):
                    base_y = self.staff.grid[i].y[0]
                    for j in range(len(self.staff.grid[i].y)):
                        idx = i * len(self.staff.grid[i].y) + j
                        if idx < len(target):
                            target[idx][1] = base_y + j * mean_spacing
        return target


def dewarp_staff(image: NDArray, staff: TabStaff, debug: Debug) -> NDArray:
    dewarper = TabStaffDewarping(staff)
    result = dewarper.dewarp(image)
    debug.write_image("dewarped", result)
    return result