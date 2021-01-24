import matplotlib.pyplot as plt
import matrices as mt
import seaborn as sns
import random
from matplotlib.animation import FuncAnimation

class Visualize:
    count = 0
    numberOfPlots = 2
    rows = 1
    columns = 1
    def __init__(self,titles):
        if(isinstance(titles,list)):
            self.data_y = [[] for i in range(len(titles))]
        else:
            self.data_y = []
        self.title = titles
        # plt.style.use('fivethirtyeight')
        self.number = Visualize.count + 1
        Visualize.count += 1
    def draw(self,new_data):
        if(isinstance(new_data,list)):
            self.data_x = [i for i in range(len(self.data_y[0])+1)]
            for i in range(len(new_data)):
                self.data_y[i].append(new_data[i])
                plt.subplot(Visualize.rows, Visualize.columns, self.number)
                plt.plot(self.data_x,self.data_y[i],label = self.title[i])
                plt.xlabel("Epochs")
            plt.legend(loc=4)
        else:
            self.data_y.append(new_data)
            self.data_x = [i for i in range(len(self.data_y))]
            plt.subplot(Visualize.rows, Visualize.columns, self.number)
            plt.plot(self.data_x,self.data_y)
            plt.ylabel(self.title)
            plt.xlabel("Epochs")

    @staticmethod
    def heatmap(matrix):
        sns.heatmap(matrix)
        plt.show()

    @staticmethod
    def setNumberOfPlots(number):
        Visualize.count=0
        while(Visualize.rows*Visualize.columns < number):
            if((Visualize.columns+Visualize.rows)%2 != 0):
                Visualize.columns += 1
            else:
                Visualize.rows += 1