import math
import operator
from handledata import handledata
from knn import knn
from FitnessFunction import Fitness
import random

class Particle():
    def __init__(self, nPartikel):
        self.nPartikel = nPartikel
        self.Posisi = []
        self.Kecepatan = []
        self.Posisi_best_i = []
        self.akurasi_i = -1
        self.akurasi_best = -1
        global dimensi

    def partikel(self):
        p = 3
        for x in range (self.nPartikel):
            self.Posisi.append(p)
            p+=2
        #print("Partikel : " + repr(self.Posisi))
        #print("Kecepatan Initial Partikel: " + repr(self.Kecepatan))
        return self.Posisi

    def kecepatan(self):
        for y in range (self.nPartikel):
            self.Kecepatan.append(v)
            v = 0
        return self.Kecepatan

    def evaluasi(self, fitness):
        self.akurasi_i = fitness

        if self.akurasi_i > self.akurasi_best or self.akurasi_best == -1:
            self.Posisi_best_i = self.Posisi.copy()
            self.akurasi_best = self.akurasi_i
        return self.akurasi_i

    def updateKecepatan(self, gbest):
        w = 0.5
        c1 = 1
        c2 = 1

        for i in range(0, dimensi):
            r1 = random.random()
            r2 = random.random()

            kecepatan_kognitif = c1*r1*(self.Posisi_best_i[i] - self.Posisi[i])
            kecepatan_sosial = c2*r2*(gbest[i] - self.Posisi[i])
            self.Kecepatan[i] = w*self.Kecepatan[i]+kecepatan_kognitif+kecepatan_sosial

    def updatePosisi(self):
        for i in range(0, dimensi):
            self.Posisi[i] = self.Posisi[i] + self.Kecepatan[i]