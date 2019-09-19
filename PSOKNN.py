import math
import operator
from handledata import handledata
from knn import knn
from ParticlePSO import Particle
from FitnessFunction import Fitness

class PSOkNN():
    def __init__(self, Jpartikel, IterMax):
        global dimensi
        dimensi = len(Particle(Jpartikel).partikel())
        self.JPartikel = Jpartikel
        self.IterX = IterMax
        self.akurasi_best = -1
        self.gbest = []

    def classify(self):
        swarm = []
        fitness = Fitness().fitness(Particle(self.JPartikel).partikel())

        #for i in range(0,self.JPartikel):
        swarm.append(Particle(self.JPartikel).partikel())
        #print("Partikel: "+repr(swarm))
        i = 0
        while i < self.IterX:
            for j in range(dimensi):
                swarm[j].Particle(self.JPartikel).evaluasi(fitness)

                if swarm[j].akurasi_i < self.akurasi_best or self.akurasi_best == -1:
                    self.gbest = list(swarm[j].Posisi)
                    self.akurasi_best = float(swarm[j].akurasi_i)
            for j in range(0, dimensi):
                swarm[j].updateKecepatan(self.gbest)
                swarm[j].updatePosisi()
            i+=1

        print("Gbest : " + repr(self.gbest))
        print("Akurasi: " + repr(self.akurasi_best))


