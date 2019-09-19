import math
import operator
from handledata import handledata
from knn import knn

class Fitness():
    def __init__(self):
        self.akurasiF = []
        self.fitt = 0

    def fitness(self, x, DataTes, Prediksi):
        self.testset = DataTes
        self.prediksi = Prediksi
        Partikel = x
        newknn = knn(len(Partikel))
        for i in range(len(Partikel)):
            #print("Partikel : "+repr(Partikel))
            #newknn = knn(Partikel[i])
            self.fitt = newknn.getAccuracy(self.testset, self.prediksi)
            self.akurasiF.append(self.fitt)
        #print("List Fitness : " + repr(self.akurasiF))
        return self.akurasiF
