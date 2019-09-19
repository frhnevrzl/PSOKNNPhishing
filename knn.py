import math
import operator
import time
from handledata import handledata

class knn():
    def __init__(self, nTetangga):
        self.k = nTetangga

    def JarakEuclidian(self, instance1, instance2, length):
        jarak = 0
        for x in range(length):
            # rumus jarak euclidian
            jarak += pow(instance1[x] - instance2[x], 2)
            return math.sqrt(jarak)

    def getNeighbors(self, trainingset, testinstance, k):
        jarak = []
        length = len(testinstance) - 1
        for x in range(len(trainingset)):
            jrk = self.JarakEuclidian(testinstance, trainingset[x], length)
            jarak.append((trainingset[x], jrk))
            jarak.sort(key=operator.itemgetter(1))
        tetangga = []
        for x in range(k):
            tetangga.append(jarak[x][0])
        return tetangga

    # Voting hasil prediksi berdasar nilai n tetangga terdekat
    def getResponse(self, Tetangga):
        VotingKelas = {}
        for x in range(len(Tetangga)):
            respon = Tetangga[x][-1]
            if respon in VotingKelas:
                VotingKelas[respon] += 1
            else:
                VotingKelas[respon] = 1

        urutkanVoting = sorted(VotingKelas.items(), key=operator.itemgetter(1), reverse=True)
        return urutkanVoting[0][0]

    def getAccuracy(self,testset, predictions):
        correct = 0
        for x in range(len(testset)):
            if testset[x][-1] is predictions[-1]:
                correct += 1
        return (correct / float(len(testset))) * 100.0

    def getTime(self):
        time_start = time.clock()
        time_elapsed = (time.clock() - time_start)
        return time_elapsed