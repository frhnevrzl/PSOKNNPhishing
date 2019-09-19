from handledata import handledata
from knn import knn
from ParticlePSO import Particle
from FitnessFunction import Fitness
#from guitest import Mainform
from PSOKNN import PSOkNN


if __name__ == '__main__':

    #form = Mainform()
    #form.Run()

    print("Operasi KNN :")
    paramK = 3
    newknn = knn(paramK)

    print("\n")

    print("Operasi PSO KNN:")
    jp = 7
    part = Particle(jp)
    nPart = part.partikel()
    print("Partikel:" + repr(nPart))
    print("Jumlah Partikel: " + repr(len(nPart)))

    ftns = Fitness()
    ftns.fitness(nPart)

    iterasi = 10

    #PSOKNN = PSOkNN(jp,iterasi)
    #PSOKNN.classify()

