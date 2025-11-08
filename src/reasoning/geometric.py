"""
Geometric and Topological Reasoning
"""

import numpy as np
from scipy.spatial import distance, ConvexHull, Delaunay, Voronoi
from scipy.spatial.transform import Rotation
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GeometricObject(str, Enum):
    POINT = "point"
    LINE = "line"
    PLANE = "plane"
    CIRCLE = "circle"
    SPHERE = "sphere"
    POLYGON = "polygon"
    POLYHEDRON = "polyhedron"
    MANIFOLD = "manifold"


@dataclass
class GeometricEntity:
    """Represents a geometric entity"""
    type: GeometricObject
    dimension: int
    coordinates: np.ndarray
    properties: Dict[str, Any] = None
    transformations: List[np.ndarray] = None


class GeometricReasoner:
    """
    Geometric reasoning and computation engine
    """

    def __init__(self):
        self.logger = logger.bind(module="geometric_reasoner")
        self.entities = {}

    def create_point(self, coordinates: List[float], name: Optional[str] = None) -> GeometricEntity:
        """Create a point in n-dimensional space"""
        point = GeometricEntity(
            type=GeometricObject.POINT,
            dimension=len(coordinates),
            coordinates=np.array(coordinates),
            properties={"name": name}
        )

        if name:
            self.entities[name] = point

        return point

    def create_line(self, point: np.ndarray, direction: np.ndarray) -> GeometricEntity:
        """Create a line from point and direction vector"""
        return GeometricEntity(
            type=GeometricObject.LINE,
            dimension=len(point),
            coordinates=np.array([point, direction]),
            properties={"parametric": True}
        )

    def create_plane(self, point: np.ndarray, normal: np.ndarray) -> GeometricEntity:
        """Create a plane from point and normal vector"""
        return GeometricEntity(
            type=GeometricObject.PLANE,
            dimension=3,
            coordinates=np.array([point, normal]),
            properties={"equation": self._plane_equation(point, normal)}
        )

    def distance_between(self, entity1: GeometricEntity, entity2: GeometricEntity) -> float:
        """Calculate distance between two geometric entities"""
        if entity1.type == GeometricObject.POINT and entity2.type == GeometricObject.POINT:
            return float(np.linalg.norm(entity1.coordinates - entity2.coordinates))

        elif entity1.type == GeometricObject.POINT and entity2.type == GeometricObject.LINE:
            return self._point_to_line_distance(entity1.coordinates, entity2.coordinates)

        elif entity1.type == GeometricObject.POINT and entity2.type == GeometricObject.PLANE:
            return self._point_to_plane_distance(entity1.coordinates, entity2.coordinates)

        else:
            raise NotImplementedError(f"Distance between {entity1.type} and {entity2.type} not implemented")

    def _point_to_line_distance(self, point: np.ndarray, line: np.ndarray) -> float:
        """Calculate distance from point to line"""
        line_point, line_direction = line[0], line[1]
        line_direction = line_direction / np.linalg.norm(line_direction)

        v = point - line_point
        projection = np.dot(v, line_direction) * line_direction
        perpendicular = v - projection

        return float(np.linalg.norm(perpendicular))

    def _point_to_plane_distance(self, point: np.ndarray, plane: np.ndarray) -> float:
        """Calculate distance from point to plane"""
        plane_point, plane_normal = plane[0], plane[1]
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        v = point - plane_point
        distance = abs(np.dot(v, plane_normal))

        return float(distance)

    def _plane_equation(self, point: np.ndarray, normal: np.ndarray) -> str:
        """Get plane equation ax + by + cz + d = 0"""
        a, b, c = normal
        d = -np.dot(normal, point)
        return f"{a}x + {b}y + {c}z + {d} = 0"

    def intersection(
        self,
        entity1: GeometricEntity,
        entity2: GeometricEntity
    ) -> Optional[GeometricEntity]:
        """Find intersection of two geometric entities"""
        if entity1.type == GeometricObject.LINE and entity2.type == GeometricObject.LINE:
            return self._line_line_intersection(entity1.coordinates, entity2.coordinates)

        elif entity1.type == GeometricObject.LINE and entity2.type == GeometricObject.PLANE:
            return self._line_plane_intersection(entity1.coordinates, entity2.coordinates)

        elif entity1.type == GeometricObject.PLANE and entity2.type == GeometricObject.PLANE:
            return self._plane_plane_intersection(entity1.coordinates, entity2.coordinates)

        else:
            raise NotImplementedError(f"Intersection of {entity1.type} and {entity2.type} not implemented")

    def _line_line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[GeometricEntity]:
        """Find intersection point of two lines"""
        p1, d1 = line1[0], line1[1]
        p2, d2 = line2[0], line2[1]

        # Check if lines are parallel
        cross = np.cross(d1, d2)
        if np.allclose(cross, 0):
            return None  # Parallel or coincident

        # For 2D case
        if len(p1) == 2:
            # Solve parametric equations
            A = np.array([d1, -d2]).T
            b = p2 - p1

            try:
                t = np.linalg.solve(A, b)
                intersection_point = p1 + t[0] * d1
                return self.create_point(intersection_point)
            except:
                return None

        # For 3D case - lines might be skew
        return None

    def _line_plane_intersection(self, line: np.ndarray, plane: np.ndarray) -> Optional[GeometricEntity]:
        """Find intersection point of line and plane"""
        line_point, line_direction = line[0], line[1]
        plane_point, plane_normal = plane[0], plane[1]

        # Check if line is parallel to plane
        denominator = np.dot(plane_normal, line_direction)
        if abs(denominator) < 1e-10:
            return None  # Line is parallel to plane

        # Find parameter t for intersection
        t = np.dot(plane_normal, plane_point - line_point) / denominator
        intersection_point = line_point + t * line_direction

        return self.create_point(intersection_point)

    def _plane_plane_intersection(self, plane1: np.ndarray, plane2: np.ndarray) -> Optional[GeometricEntity]:
        """Find intersection line of two planes"""
        point1, normal1 = plane1[0], plane1[1]
        point2, normal2 = plane2[0], plane2[1]

        # Line direction is perpendicular to both normals
        line_direction = np.cross(normal1, normal2)

        # Check if planes are parallel
        if np.allclose(line_direction, 0):
            return None  # Planes are parallel

        # Find a point on the intersection line
        # Solve the system of equations
        # We need to find a point that satisfies both plane equations
        # Use the method of finding the closest point to origin on the line

        # Normalize direction
        line_direction = line_direction / np.linalg.norm(line_direction)

        # Find point on line closest to origin
        A = np.array([normal1, normal2])
        b = np.array([np.dot(normal1, point1), np.dot(normal2, point2)])

        # Add constraint to fix one coordinate
        A = np.vstack([A, [1, 0, 0]])
        b = np.append(b, 0)

        try:
            line_point = np.linalg.lstsq(A, b, rcond=None)[0]
            return self.create_line(line_point, line_direction)
        except:
            return None

    def angle_between(self, entity1: GeometricEntity, entity2: GeometricEntity) -> float:
        """Calculate angle between two entities (in radians)"""
        if entity1.type == GeometricObject.LINE and entity2.type == GeometricObject.LINE:
            v1 = entity1.coordinates[1]
            v2 = entity2.coordinates[1]
        elif entity1.type == GeometricObject.PLANE and entity2.type == GeometricObject.PLANE:
            v1 = entity1.coordinates[1]
            v2 = entity2.coordinates[1]
        else:
            raise NotImplementedError(f"Angle between {entity1.type} and {entity2.type} not implemented")

        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Calculate angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return float(angle)

    def transform(
        self,
        entity: GeometricEntity,
        transformation: str,
        parameters: Dict[str, Any]
    ) -> GeometricEntity:
        """
        Apply geometric transformation to entity

        Args:
            entity: Entity to transform
            transformation: Type of transformation (translate, rotate, scale, reflect)
            parameters: Transformation parameters

        Returns:
            Transformed entity
        """
        if transformation == "translate":
            translation = np.array(parameters["vector"])
            if entity.type == GeometricObject.POINT:
                new_coords = entity.coordinates + translation
            else:
                new_coords = entity.coordinates.copy()
                new_coords[0] += translation  # Translate reference point

        elif transformation == "rotate":
            angle = parameters["angle"]
            axis = parameters.get("axis", [0, 0, 1])  # Default to z-axis
            center = parameters.get("center", np.zeros(entity.dimension))

            # Create rotation matrix
            if entity.dimension == 2:
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
            else:
                rotation = Rotation.from_rotvec(angle * np.array(axis))
                rotation_matrix = rotation.as_matrix()

            # Apply rotation
            if entity.type == GeometricObject.POINT:
                centered = entity.coordinates - center
                rotated = rotation_matrix @ centered
                new_coords = rotated + center
            else:
                new_coords = entity.coordinates.copy()
                # Rotate both point and direction/normal
                new_coords[0] = rotation_matrix @ (new_coords[0] - center) + center
                new_coords[1] = rotation_matrix @ new_coords[1]

        elif transformation == "scale":
            scale_factor = parameters["factor"]
            center = parameters.get("center", np.zeros(entity.dimension))

            if entity.type == GeometricObject.POINT:
                new_coords = center + scale_factor * (entity.coordinates - center)
            else:
                new_coords = entity.coordinates.copy()
                new_coords[0] = center + scale_factor * (new_coords[0] - center)
                if entity.type == GeometricObject.LINE:
                    new_coords[1] = new_coords[1] * scale_factor

        elif transformation == "reflect":
            if "plane" in parameters:
                plane_point = parameters["plane"]["point"]
                plane_normal = parameters["plane"]["normal"]
                plane_normal = plane_normal / np.linalg.norm(plane_normal)

                if entity.type == GeometricObject.POINT:
                    # Reflect point across plane
                    v = entity.coordinates - plane_point
                    distance = np.dot(v, plane_normal)
                    new_coords = entity.coordinates - 2 * distance * plane_normal
                else:
                    new_coords = entity.coordinates.copy()
                    # Reflect reference point
                    v = new_coords[0] - plane_point
                    distance = np.dot(v, plane_normal)
                    new_coords[0] = new_coords[0] - 2 * distance * plane_normal
                    # Reflect direction/normal
                    new_coords[1] = new_coords[1] - 2 * np.dot(new_coords[1], plane_normal) * plane_normal
            else:
                raise ValueError("Reflection requires a plane")

        else:
            raise ValueError(f"Unknown transformation: {transformation}")

        # Create new entity with transformation recorded
        new_entity = GeometricEntity(
            type=entity.type,
            dimension=entity.dimension,
            coordinates=new_coords,
            properties=entity.properties.copy() if entity.properties else {},
            transformations=entity.transformations.copy() if entity.transformations else []
        )

        if new_entity.transformations is None:
            new_entity.transformations = []
        new_entity.transformations.append({
            "type": transformation,
            "parameters": parameters
        })

        return new_entity

    def convex_hull(self, points: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute convex hull of a set of points

        Args:
            points: List of points

        Returns:
            Convex hull information
        """
        points_array = np.array(points)

        if len(points) < 3:
            return {
                "vertices": points,
                "volume": 0,
                "area": 0,
                "simplices": []
            }

        try:
            hull = ConvexHull(points_array)

            return {
                "vertices": points_array[hull.vertices],
                "volume": hull.volume,
                "area": hull.area,
                "simplices": hull.simplices,
                "num_vertices": len(hull.vertices),
                "num_faces": len(hull.simplices)
            }
        except Exception as e:
            self.logger.error(f"Convex hull computation failed: {e}")
            return {
                "error": str(e),
                "vertices": [],
                "volume": 0,
                "area": 0,
                "simplices": []
            }

    def triangulation(self, points: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute Delaunay triangulation of points

        Args:
            points: List of points

        Returns:
            Triangulation information
        """
        points_array = np.array(points)

        try:
            tri = Delaunay(points_array)

            return {
                "points": points_array,
                "simplices": tri.simplices,
                "neighbors": tri.neighbors,
                "num_simplices": len(tri.simplices),
                "convex_hull": tri.convex_hull
            }
        except Exception as e:
            self.logger.error(f"Triangulation failed: {e}")
            return {"error": str(e)}

    def voronoi_diagram(self, points: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute Voronoi diagram of points

        Args:
            points: List of points

        Returns:
            Voronoi diagram information
        """
        points_array = np.array(points)

        try:
            vor = Voronoi(points_array)

            return {
                "points": vor.points,
                "vertices": vor.vertices,
                "ridge_points": vor.ridge_points,
                "ridge_vertices": vor.ridge_vertices,
                "regions": vor.regions,
                "point_region": vor.point_region,
                "num_regions": len(vor.regions)
            }
        except Exception as e:
            self.logger.error(f"Voronoi diagram failed: {e}")
            return {"error": str(e)}


class TopologicalAnalyzer:
    """
    Topological analysis of mathematical structures
    """

    def __init__(self):
        self.logger = logger.bind(module="topological_analyzer")

    def euler_characteristic(self, vertices: int, edges: int, faces: int) -> int:
        """
        Compute Euler characteristic V - E + F

        Args:
            vertices: Number of vertices
            edges: Number of edges
            faces: Number of faces

        Returns:
            Euler characteristic
        """
        return vertices - edges + faces

    def genus(self, euler_char: int, boundary_components: int = 0) -> int:
        """
        Compute genus of a surface

        Args:
            euler_char: Euler characteristic
            boundary_components: Number of boundary components

        Returns:
            Genus
        """
        # For closed orientable surface: χ = 2 - 2g
        # For surface with b boundaries: χ = 2 - 2g - b
        return (2 - euler_char - boundary_components) // 2

    def homology_groups(self, complex: Dict[str, List]) -> Dict[str, Any]:
        """
        Compute homology groups of a simplicial complex

        Args:
            complex: Simplicial complex representation

        Returns:
            Homology group information
        """
        # This is a simplified computation
        # Real homology computation would use more sophisticated algorithms

        vertices = complex.get("vertices", [])
        edges = complex.get("edges", [])
        faces = complex.get("faces", [])

        # Betti numbers (simplified)
        b0 = self._compute_connected_components(vertices, edges)
        b1 = len(edges) - len(vertices) + b0  # First homology (cycles)
        b2 = len(faces) if faces else 0  # Second homology (voids)

        return {
            "betti_numbers": [b0, b1, b2],
            "euler_characteristic": b0 - b1 + b2,
            "connected_components": b0,
            "holes": b1,
            "voids": b2
        }

    def _compute_connected_components(self, vertices: List, edges: List) -> int:
        """Compute number of connected components"""
        if not vertices:
            return 0

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(len(vertices)))

        for edge in edges:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])

        return nx.number_connected_components(G)

    def fundamental_group(self, space_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute fundamental group of a topological space

        Args:
            space_type: Type of space (circle, torus, sphere, etc.)
            parameters: Space parameters

        Returns:
            Fundamental group information
        """
        fundamental_groups = {
            "circle": {
                "group": "Z",
                "generators": 1,
                "relations": [],
                "abelian": True,
                "description": "Infinite cyclic group"
            },
            "sphere": {
                "group": "trivial" if parameters.get("dimension", 2) >= 2 else "Z",
                "generators": 0 if parameters.get("dimension", 2) >= 2 else 1,
                "relations": [],
                "abelian": True,
                "description": "Trivial group" if parameters.get("dimension", 2) >= 2 else "Infinite cyclic"
            },
            "torus": {
                "group": "Z × Z",
                "generators": 2,
                "relations": ["[a, b] = 1"],
                "abelian": True,
                "description": "Product of two infinite cyclic groups"
            },
            "projective_plane": {
                "group": "Z/2Z",
                "generators": 1,
                "relations": ["a^2 = 1"],
                "abelian": True,
                "description": "Cyclic group of order 2"
            },
            "klein_bottle": {
                "group": "<a, b | aba^{-1} = b^{-1}>",
                "generators": 2,
                "relations": ["aba^{-1} = b^{-1}"],
                "abelian": False,
                "description": "Non-abelian group"
            }
        }

        if space_type in fundamental_groups:
            return fundamental_groups[space_type]
        else:
            return {
                "group": "unknown",
                "error": f"Fundamental group for {space_type} not implemented"
            }

    def knot_invariants(self, knot_code: str) -> Dict[str, Any]:
        """
        Compute knot invariants

        Args:
            knot_code: Knot notation (e.g., "3_1" for trefoil)

        Returns:
            Knot invariants
        """
        # Simplified knot invariants
        known_knots = {
            "0_1": {  # Unknot
                "name": "Unknot",
                "crossing_number": 0,
                "bridge_number": 1,
                "unknotting_number": 0,
                "genus": 0,
                "alexander_polynomial": "1",
                "jones_polynomial": "1"
            },
            "3_1": {  # Trefoil
                "name": "Trefoil knot",
                "crossing_number": 3,
                "bridge_number": 2,
                "unknotting_number": 1,
                "genus": 1,
                "alexander_polynomial": "t - 1 + t^{-1}",
                "jones_polynomial": "t + t^3 - t^4"
            },
            "4_1": {  # Figure-eight knot
                "name": "Figure-eight knot",
                "crossing_number": 4,
                "bridge_number": 2,
                "unknotting_number": 1,
                "genus": 1,
                "alexander_polynomial": "-t + 3 - t^{-1}",
                "jones_polynomial": "t^{-2} - t^{-1} + 1 - t + t^2"
            },
            "5_1": {  # Cinquefoil
                "name": "Cinquefoil knot",
                "crossing_number": 5,
                "bridge_number": 2,
                "unknotting_number": 2,
                "genus": 2,
                "alexander_polynomial": "t^2 - t + 1 - t^{-1} + t^{-2}",
                "jones_polynomial": "t^2 + t^4 - t^5 + t^6 - t^7"
            }
        }

        if knot_code in known_knots:
            return known_knots[knot_code]
        else:
            return {
                "name": f"Knot {knot_code}",
                "error": "Invariants not in database",
                "crossing_number": int(knot_code.split("_")[0]) if "_" in knot_code else None
            }

    def manifold_classification(self, dimension: int, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a manifold based on its properties

        Args:
            dimension: Manifold dimension
            properties: Manifold properties (orientable, compact, etc.)

        Returns:
            Classification information
        """
        classification = {
            "dimension": dimension,
            "properties": properties,
            "possible_manifolds": []
        }

        if dimension == 2:
            # Classification of 2-manifolds (surfaces)
            if properties.get("compact", True):
                if properties.get("orientable", True):
                    # Orientable compact surfaces: sphere, torus, n-torus
                    genus = properties.get("genus", 0)
                    if genus == 0:
                        classification["manifold"] = "2-sphere (S²)"
                    elif genus == 1:
                        classification["manifold"] = "Torus (T²)"
                    else:
                        classification["manifold"] = f"{genus}-torus"

                    classification["euler_characteristic"] = 2 - 2 * genus
                else:
                    # Non-orientable compact surfaces
                    genus = properties.get("genus", 1)
                    if genus == 1:
                        classification["manifold"] = "Projective plane (RP²)"
                    elif genus == 2:
                        classification["manifold"] = "Klein bottle"
                    else:
                        classification["manifold"] = f"Connected sum of {genus} projective planes"

                    classification["euler_characteristic"] = 2 - genus
            else:
                # Non-compact surfaces
                classification["manifold"] = "Non-compact surface"
                classification["possible_manifolds"] = ["Plane (R²)", "Cylinder", "Möbius strip"]

        elif dimension == 3:
            # 3-manifolds are much more complex
            if properties.get("compact", True) and properties.get("orientable", True):
                classification["possible_manifolds"] = [
                    "3-sphere (S³)",
                    "3-torus (T³)",
                    "Lens space",
                    "Seifert fiber space",
                    "Hyperbolic 3-manifold"
                ]

                if properties.get("simply_connected", False):
                    classification["manifold"] = "3-sphere (S³) by Poincaré conjecture"
            else:
                classification["possible_manifolds"] = [
                    "Euclidean 3-space (R³)",
                    "Product manifolds"
                ]

        else:
            classification["note"] = f"Classification of {dimension}-manifolds is complex"

        return classification

    def persistent_homology(self, point_cloud: List[np.ndarray], max_dimension: int = 2) -> Dict[str, Any]:
        """
        Compute persistent homology of a point cloud (simplified)

        Args:
            point_cloud: List of points
            max_dimension: Maximum homology dimension to compute

        Returns:
            Persistent homology information
        """
        points = np.array(point_cloud)
        n_points = len(points)

        # Compute pairwise distances
        distances = distance.cdist(points, points)

        # Find persistence intervals (simplified)
        persistence = {
            "dimension_0": [],  # Connected components
            "dimension_1": [],  # Loops
            "dimension_2": []   # Voids
        }

        # Birth times for 0-dimensional features (each point starts as component)
        for i in range(n_points):
            persistence["dimension_0"].append({"birth": 0, "death": float('inf')})

        # Find when components merge (simplified)
        epsilon_values = np.unique(distances[distances > 0])[:20]  # First 20 values

        for eps in epsilon_values:
            # Build graph at this epsilon
            G = nx.Graph()
            G.add_nodes_from(range(n_points))

            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if distances[i, j] <= eps:
                        G.add_edge(i, j)

            # Count components
            n_components = nx.number_connected_components(G)

            # Update death times for merged components
            if n_components < n_points:
                for i in range(n_points - n_components):
                    if persistence["dimension_0"][i]["death"] == float('inf'):
                        persistence["dimension_0"][i]["death"] = eps

        # Compute persistence diagram features
        diagram_0 = [(p["birth"], p["death"]) for p in persistence["dimension_0"] if p["death"] < float('inf')]

        return {
            "persistence_intervals": persistence,
            "diagram_0": diagram_0,
            "max_persistence_0": max([d - b for b, d in diagram_0]) if diagram_0 else 0,
            "num_features_0": len([p for p in persistence["dimension_0"] if p["death"] == float('inf')]),
            "bottleneck_distance": None  # Would require comparison with another diagram
        }


# Example usage
if __name__ == "__main__":
    # Test geometric reasoner
    reasoner = GeometricReasoner()

    # Create geometric entities
    p1 = reasoner.create_point([0, 0, 0], "origin")
    p2 = reasoner.create_point([1, 1, 1], "point_1")

    # Calculate distance
    dist = reasoner.distance_between(p1, p2)
    print(f"Distance between points: {dist:.3f}")

    # Create line and plane
    line = reasoner.create_line(np.array([0, 0, 0]), np.array([1, 1, 1]))
    plane = reasoner.create_plane(np.array([0, 0, 1]), np.array([0, 0, 1]))

    # Find intersection
    intersection = reasoner.intersection(line, plane)
    if intersection:
        print(f"Intersection point: {intersection.coordinates}")

    # Transform a point
    rotated = reasoner.transform(p2, "rotate", {"angle": np.pi/4, "axis": [0, 0, 1]})
    print(f"Rotated point: {rotated.coordinates}")

    # Test topological analyzer
    analyzer = TopologicalAnalyzer()

    # Euler characteristic of a cube
    euler = analyzer.euler_characteristic(vertices=8, edges=12, faces=6)
    print(f"Euler characteristic of cube: {euler}")

    # Fundamental group of torus
    fg = analyzer.fundamental_group("torus", {})
    print(f"Fundamental group of torus: {fg['group']}")

    # Knot invariants
    knot = analyzer.knot_invariants("3_1")
    print(f"Trefoil knot genus: {knot['genus']}")

    # Manifold classification
    manifold = analyzer.manifold_classification(2, {"compact": True, "orientable": True, "genus": 1})
    print(f"2-manifold classification: {manifold['manifold']}")

    # Persistent homology
    point_cloud = np.random.randn(20, 3)
    persistence = analyzer.persistent_homology(point_cloud.tolist())
    print(f"Number of persistent features: {persistence['num_features_0']}")