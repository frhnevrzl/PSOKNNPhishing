import csv
import random
from pathlib import Path

class handledata:
    def __init__(self, filename, split, trainingSet=[], testSet=[]):
        # konversi tipe file ke .csv
        with open(str(filename)) as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            # Membagi dataset menjadi Data Training dan Testing dengan Fungsi Random
            dataset = list(lines)
            for x in range(len(dataset) - 1):
                for y in range(9):
                    dataset[x][y] = float(dataset[x][y])
                    # Spliting data
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])