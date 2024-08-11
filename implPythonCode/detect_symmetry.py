import numpy as np

def detect_symmetry(points):
    centroid = np.mean(points, axis=0)
    reflected_points = 2 * centroid - points
    differences = np.linalg.norm(points - reflected_points, axis=1)
    symmetry = np.all(differences < 1e-6)  
    return symmetry
