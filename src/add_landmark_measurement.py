import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Extract the estimated pose of X(4) and the optimized position of L(2)
    pose_4 = initial_estimate.atPose2(X(4))
    l2_point = result.atPoint2(L(2))
    
    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    # using GTSAM's built-in methods to simulate the exact measurement
    rotation = pose_4.bearing(l2_point).degrees()
    distance = pose_4.range(l2_point)
    
    graph.add(gtsam.BearingRangeFactor2D(
        X(4), 
        L(2), 
        gtsam.Rot2.fromDegrees(rotation), 
        distance, 
        MEASUREMENT_NOISE
    ))
    
    return graph