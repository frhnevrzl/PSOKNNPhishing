import PySimpleGUI as sg
from handledata import handledata
from pathlib import Path
from knn import knn
from ParticlePSO import Particle
from FitnessFunction import Fitness

class Mainform:
    def __init__(self):
        sg.ChangeLookAndFeel('GreenMono')
        column1 = [
            [sg.Frame(title='Option', layout=[
                [sg.Radio("k-Nearest Neighbour", disabled=True, key="rbKNN", group_id="Option", enable_events=True,
                          default=True)],
                [sg.Radio("PSO k-Nearest Neighbour", disabled=True, key="rbPSOkNN", group_id="Option",
                          enable_events=True)]
            ])],
            [sg.Frame(title='Parameter_k', layout=[
                [sg.Text(text="Parameter k", size=(12, 1)),
                 sg.Input(key="_Parameter_k_", size=(4, 1), tooltip="Masukan Nilai Integer", disabled=True)]
            ])],
            [sg.Frame(title='Parameter PSO', layout=[
                [sg.Text(text="Jumlah Partikel", size=(12, 1)),
                 sg.Input(key="_Jumlah_Partikel_", size=(4, 1), tooltip="Masukan Nilai Integer", disabled=True)],
                [sg.Text(text="Jumlah Iterasi", size=(12, 1)),
                 sg.Input(key="_Jumlah_iterasi_", size=(4, 1), tooltip="Masukan Nilai Integer", disabled=True)]
            ])],
            [sg.Frame(title='Bobot Inertia', layout=[
                [sg.Radio("Constant", disabled=True, key="rbwC", group_id="wOption", enable_events=True,
                          default=True)],
                [sg.Radio("Linear Decreasing", disabled=True, key="rbwL", group_id="wOption", enable_events=True)],
                [sg.Radio("Random", disabled=True, key="rbwR", group_id="wOption", enable_events=True)]
            ])]
        ]

        column2 = [
            [sg.Button('Mulai', size=(58, 1), disabled=True)],
            [sg.Text('-' * 117)],
            [sg.Frame("Hasil Klasifikasi", layout=[
                [sg.Text(text="Data Training : ", size=(48, 1), key="_Training_")],
                [sg.Text(text="Data Testing : ", size=(48, 1), key="_Testing_")],
                [sg.Text(text="Nilai Akurasi : ", size=(48, 1), key="_akurasi_")],
                [sg.Text(text="Waktu Komputasi : ", size=(48, 1), key="_waktu_komputasi_")]
            ])],
            [sg.Multiline(key="_output_", disabled=True)]
        ]
        layout = [
            [sg.FileBrowse('Muat', size=(6, 1), target='_Path_',
                           initial_folder=r"E:\File Kuliah\TA\ProgramTA\PSOKNNPhishing",
                           file_types=[("Txt files", ".txt")]), sg.Column([
                [sg.Input(key="_Path_", disabled=True, size=(95, 1), enable_events=True)]
            ])],
            [sg.Text('-' * 187)],
            [sg.Column(column1), sg.VerticalSeparator(), sg.Column(column2)],
        ]
        self.window = sg.Window('Klasifikasi Web Phishing', layout, default_element_size=(40, 1), grab_anywhere=False)

    def Run(self):

        trainingSet = []
        testSet = []

        while True:
            event, values = self.window.Read()
            file_path = None
            split = 0.7
            akurasi = 0.0
            waktu_komputasi = 0.0

            if event == "_Path_":
                self.window.FindElement('rbKNN').Update(disabled=False)
                self.window.FindElement('rbPSOkNN').Update(disabled=False)
                self.window.FindElement('Mulai').Update(disabled=False)

                try:
                    file_path = Path(values['_Path_'])
                    dataset = handledata(file_path,split,trainingSet,testSet)
                except ValueError:
                    sg.Popup("\tFormat data salah !!\t", title="error")
                else:
                    self.window.FindElement('_Training_').Update('Data Training: ' + repr(len(trainingSet)))
                    self.window.FindElement('_Testing_').Update('Data Testing: ' + repr(len(testSet)))

            if event == "rbKNN":
                self.window.FindElement('_Parameter_k_').Update(disabled=False)
                self.window.FindElement('_Jumlah_Partikel_').Update(disabled = True)
                self.window.FindElement('_Jumlah_iterasi_').Update(disabled=True)
                self.window.FindElement('rbwC').Update(disabled=True)
                self.window.FindElement('rbwL').Update(disabled=True)
                self.window.FindElement('rbwR').Update(disabled=True)

            if event == "rbPSOkNN":
                self.window.FindElement('_Parameter_k_').Update(disabled=True)
                self.window.FindElement('_Jumlah_Partikel_').Update(disabled = False)
                self.window.FindElement('_Jumlah_iterasi_').Update(disabled=False)
                self.window.FindElement('rbwC').Update(disabled=False)
                self.window.FindElement('rbwL').Update(disabled=False)
                self.window.FindElement('rbwR').Update(disabled=False)

            param_k = None
            output = ""

            if event == "Mulai":
                try:
                    param_k = int(values['_Parameter_k_'])
                    if values['rbKNN'] == True:
                        predictions = []
                        newkNN = knn(param_k)
                        for x in range(len(testSet)):
                            neighbours = newkNN.getNeighbors(trainingSet, testSet[x], param_k)
                            result = newkNN.getResponse(neighbours)
                            predictions = result
                            output = output + '> Hasil Klasifikasi Program =' + repr(result) + ', Hasil Klasifikasi data Asli =' + repr(testSet[x][-1]) + '\n'
                        self.window.FindElement('_output_').Update(output)
                        accuracy = newkNN.getAccuracy(testSet, predictions)
                        waktu_komputasi = newkNN.getTime()
                        self.window.FindElement('_akurasi_').Update("Nilai Akurasi : " + repr(accuracy) + "%")
                        self.window.FindElement('_waktu_komputasi_').Update(("Waktu Komputasi : "+ repr(waktu_komputasi) + " detik"))

                except ValueError:
                    sg.Popup("Nilai Parameter k Harus Integer",title = "Error")

                    J_Partikel = int(values['_Jumlah_Partikel_'])
                    if values['rbPSOkNN'] == True:
                        predictions = []
                        fitness_i = []

                        part = Particle(J_Partikel)
                        partikel = part.partikel()
                        self.window.FindElement('_output_').Update(("Partikel PSO : " + repr(partikel)))

                        for i in partikel:
                            newkNN = knn(i)
                            print(repr(i))
                            for x in range (len(testSet)):
                                neighbours = newkNN.getNeighbors(trainingSet,testSet[x],i)
                                result = newkNN.getResponse(neighbours)
                                print(repr(result))
                                predictions = result
                        for j in range (len(partikel)):
                            accuracy = newkNN.getAccuracy(testSet, predictions)
                            print(repr(accuracy))
                            #print(repr(partikel))
                            fitness_i.append(accuracy)
                        self.window.FindElement('_output_').Update("List Fitness: " + repr(fitness_i))


            if event is None or event == 'Exit':
                self.window.Close()
                break
            print(event, values)

if __name__ == "__main__":
    form = Mainform()
    form.Run()
