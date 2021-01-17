import matplotlib.pyplot as plt
import matrices as mt
import numpy as np
import seaborn as sns

class Visualize:
    def draw(self):
        plt.plot(self.data_y,range(len(self.data_y)),label="line1")
        plt.plot(range(len(self.data_y)),self.data_y,label="line2")
        plt.ylabel(self.title)
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
    def add_plot(self,new_data,new_label):
        plt.plot(self.data_y,range(len(self.data_y)),label=self.title)
        plt.plot(new_data,range(len(new_data)),label=new_label)
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
    def heatmap(self,matrix):
        sns.heatmap(matrix)
        plt.show()