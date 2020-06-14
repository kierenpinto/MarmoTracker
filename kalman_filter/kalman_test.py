from kalman import Kalman
import numpy as np
from pykalman import KalmanFilter
test_data = np.tile(np.arange(0,10,1),[4,1]).T # t, x,y,z
diffed_data = np.zeros((9,7))
diffed_data[:,0:4] = test_data[1:,:]
diffed_data[:,4] = test_data[1:10,1]-test_data[0:9,1] #dx,dy,dz
diffed_data[:,5] = test_data[1:10,2]-test_data[0:9,2]
diffed_data[:,6] = test_data[1:10,3]-test_data[0:9,3]
# print(diffed_data)

first_measure = diffed_data[0,1:]
second_measure = diffed_data[1,1:]
time_elapsed = diffed_data[1,0] - diffed_data[0,0]
k = Kalman()


covariance=k.initCov
means = diffed_data[0,1:7]
print(means.shape,covariance.shape)

for i in range(1,diffed_data.shape[0]):
    next_measurement = diffed_data[i,1:]
    means, covariance = k.update(next_measurement,means,covariance)
    # print(means,covariance)
    