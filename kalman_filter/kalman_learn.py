#!/usr/bin/python

from pykalman import KalmanFilter

import numpy as np
kf = KalmanFilter(transition_matrices = [[1, 0], [0, 1]], observation_matrices = [[1, 0], [0, 1]])
measurements = np.asarray([[1,0], [0,0], [0,1],[1,2]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

print(filtered_state_means)




# transition_matrices = [[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
# observation_matrices = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]]
# initcovariance=1.0e-3*np.eye(6)
# transistionCov=1.0e-4*np.eye(6)
# observationCov=1.0e-1*np.eye(3)
# kf = KalmanFilter(transition_matrices=transition_matrices, 
#                 observation_matrices=observation_matrices, 
#                 initial_state_mean=first_measure,
#                 initial_state_covariance=initcovariance,
#                 transition_covariance=transistionCov,
#                 observation_covariance=observationCov)

# (filtered_state_means, filtered_state_covariances) = kf.filter(diffed_data[:,1:4])

# covariance = k.initial_measurement(first_measure, second_measure, time_elapsed)[1]

# means = diffed_data[0,1:4]
# means, covariance = k.update(next_measurement,means,covariance)