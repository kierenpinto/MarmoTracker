#!/usr/bin/python
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
'''
Documentation can be found:
https://matplotlib.org/3.1.3/tutorials/toolkits/mplot3d.html#quiver 
'''
import matplotlib.pyplot as plt
import numpy as np
import time
from tracking import Sequence
from matplotlib.animation import FuncAnimation
from multiprocessing import Process, Queue
import queue

class MarmosetMovementPlot:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlabel('X (horizontal m)')
        self.ax.set_ylabel('Y (vertical m)')
        self.ax.set_zlabel('Z (depth m)')
        self.ax.set_xlim([-0.5,0.5])
        self.ax.set_ylim([-0.2,0.2])
        self.ax.set_zlim([0,1])

    def plotSegment(self,startPoint,endPoint):
        plt.ioff()
        line_distance = np.sqrt((startPoint[0]-endPoint[0])**2 + (startPoint[1] -endPoint[1])**2 + (startPoint[2] - endPoint[2])**2 )
        self.ax.quiver(startPoint[0],startPoint[1],startPoint[2], endPoint[0],endPoint[1],endPoint[2],length=line_distance, normalize=True)
        self.fig.canvas.draw()
        plt.ion()

    def exit(self):
        plt.close(self.fig)

class MarmosetScatterPlot:
    def __init__(self):
        # self.fig = plt.figure()
        # self.ax = self.fig.gca()
        # self.data = np.array([[0,0]])
        print("intialised Plot")
        self.q = Queue()
        self.p = Process(target=self.processUpdate,args=(self.q,))
        self.p.start()

    def processUpdate(self,q):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlim([-0.5,0.5])
        ax.set_ylim([-0.2,0.2])
        ax.set_xlabel('X (horizontal m)')
        ax.set_ylabel('Y (vertical m)')
        # data = np.array([[0,0]])
        def plotSegment(i):
            while (q.qsize()>1):
                data = q.get()
            try:
                data = q.get(block=False)
                ax.cla
                ax.scatter(data[:,0],-data[:,1],color='b')
            except queue.Empty:
                pass
            
        ani = FuncAnimation(fig,plotSegment,interval=1000)
        plt.show()

    def start(self):
        pass
        # def plotSegment(i):
        #     print("plot segment")
        #     if (type(self.data)==np.ndarray):
        #         self.ax.scatter(self.data[:,0],self.data[:,1])
        
        # self.ani = FuncAnimation(self.fig,plotSegment,interval=1000)
        # print("1")
        # self.fig.show()
        # plt.show()
        # print("2")

    def exit(self):
        plt.close(self.fig)



class AnimationPlot:
    def __init__(self):
        self.instance = MarmosetScatterPlot()
        self.sequence = None
    def update(self,sequence: Sequence):
        if sequence is not None:
            self.sequence = np.stack(np.array(sequence.arr)[:,2])
            # print(self.sequence[:,2])
            self.instance.q.put(self.sequence)

    def start(self):
        print("start amimation")
        self.instance.data = self.sequence
        self.instance.start()

    def exit(self):
        self.instance.p.terminate()


class PlotSequence:
    def __init__(self):
        self.instance = MarmosetMovementPlot()
        self.previousPoint = ()
    def update(self,sequence: Sequence):
        if sequence is not None:
            current_datapoint = sequence.arr[-1]
            if len(self.previousPoint) == 0:
                # Don't plot
                self.previousPoint = current_datapoint
            else:
                # print(self.previousPoint[2
                # 
                # ],current_datapoint[2])
                x,y,z = self.previousPoint[2]
                x2,y2,z2 = current_datapoint[2]
                print(x,y,z,x2,y2,z2)
                self.instance.plotSegment((x,y,z),(x2,y2,z2))
                self.previousPoint = current_datapoint


# class PlotInstance:
#     plot = MarmosetMovementPlot()
#     def update(self,sequence)