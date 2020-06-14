#!/usr/bin/python

from pykalman import KalmanFilter

import numpy as np

'''
References:
 http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies
 https://www.intechopen.com/books/introduction-and-implementations-of-the-kalman-filter/introduction-to-kalman-filter-and-its-applications 
 https://pykalman.github.io/#pykalman.KalmanFilter
 https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Tracking.html 

'''
class Kalman:
    initCov = 1.0e-3*np.eye(6)
    def __init__(self):
        transition_matrices = [[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        observation_matrices = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        self.kf = KalmanFilter(transition_matrices=transition_matrices, observation_matrices=observation_matrices)

    def update(self,next_measurement,means,cov):
        # self.kf.em(next_measurement, n_iter=5)
        next_means, next_cov = self.kf.filter_update(means,cov,next_measurement)
        return next_means, next_cov

    # @staticmethod
    # def initial_measurement(first_measure, second_measure,time_elapsed):
    #     initcovariance=1.0e-3*np.eye(6)
    #     initmeans = (first_measure - second_measure)/time_elapsed
    #     return initmeans, initcovariance


