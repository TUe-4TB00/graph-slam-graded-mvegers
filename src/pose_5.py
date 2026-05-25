import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    # (Print omitted inside the function to avoid spamming the console during the loops)
    
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum_of_marginals = float('inf')

    # Iterate over all candidate poses and both landmarks
    for pose_key, pose_5 in pose_options.items():
        for landmark_idx in [1, 2]:
            # Clone graph and estimate so we don't mutate the originals during our search
            temp_graph = graph.clone()
            temp_estimate = gtsam.Values(initial_estimate)

            # Build and optimize for this specific combination
            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)
            result = optimize(temp_graph, temp_estimate)
            
            temp_graph = add_landmark_measurement(temp_graph, result, pose_5, landmark_idx)
            result = optimize(temp_graph, temp_estimate)

            # TODO: Calculate marginal covariances for the relevant variables
            marginals = gtsam.Marginals(temp_graph, result)
            
            # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            # We calculate the sum of both landmarks to see the overall reduction in landmark uncertainty
            sum_of_marginals = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()

            # Keep track of the lowest covariance found
            if sum_of_marginals < min_sum_of_marginals:
                min_sum_of_marginals = sum_of_marginals
                best_pose = pose_key
                best_landmark = landmark_idx

    return best_pose, best_landmark, min_sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum_of_errors = float('inf')

    # Ideal/Ground truth positions from the first part of the assignment:
    # X(1): (0, 0), X(2): (2, 0), X(3): (4, 0)
    ideal_poses = {
        1: gtsam.Point2(0.0, 0.0),
        2: gtsam.Point2(2.0, 0.0),
        3: gtsam.Point2(4.0, 0.0)
    }

    for pose_key, pose_5 in pose_options.items():
        for landmark_idx in [1, 2]:
            temp_graph = graph.clone()
            temp_estimate = gtsam.Values(initial_estimate)

            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)
            result = optimize(temp_graph, temp_estimate)
            
            temp_graph = add_landmark_measurement(temp_graph, result, pose_5, landmark_idx)
            result = optimize(temp_graph, temp_estimate)

            # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []
            
            # Calculate the Euclidean distance between the estimated pose and the known ground truth
            for i in [1, 2, 3]:
                est_pose = result.atPose2(X(i))
                ideal_point = ideal_poses[i]
                
                # Positional error (distance)
                error = np.sqrt((est_pose.x() - ideal_point[0])**2 + (est_pose.y() - ideal_point[1])**2)
                list_of_errors.append(error)

            # TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < min_sum_of_errors:
                min_sum_of_errors = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark_idx

    return best_pose, best_landmark, min_sum_of_errors