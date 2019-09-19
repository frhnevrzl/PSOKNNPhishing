import PySimpleGUI as sg
import pandas as pd
import numpy as np

# from pathlib import Path
# from Evaluasi import Evaluasi
# from sklearn.model_selection import KFold
# from Praproses import Praproses
# from Classifier import kNearestNeighbor, modifiedKNearestNeighbor # Classifier
# from SeleksiGenetika import AlgoritmaGenetika # Fitur seleksi

class MainForm:

    def __init__(self):
        sg.ChangeLookAndFeel('Reddit')

        column1 = [
            [sg.Frame(title="Option", layout=[
                [sg.Radio("k-Nearest Neighbor", disabled=True, key="_rbkNN_", group_id="Option", enable_events=True,
                          default=True)],
                [sg.Radio("Modified k-Nearest Neighbor", disabled=True, key="_rbMkNN_", group_id="Option",
                          enable_events=True)],
                [sg.Radio("GA + k-Nearest Neighbor", disabled=True, key="_rbGAkNN_", group_id="Option",
                          enable_events=True)],
                [sg.Radio("GA + Modified k-Nearest Neighbor", disabled=True, key="_rbGAMkNN_", group_id="Option",
                          enable_events=True)]
            ])],
            [sg.Frame(title="Parameter k", layout=[
                [sg.Text(text="Parameter k", size=(12, 1)),
                 sg.Input(key="_parameter_k_", size=(16, 1), tooltip="Masukan nilai Integer", disabled=True)]
            ])],
            [sg.Frame(title="Parameter Algoritma Genetika", layout=[
                [sg.Text(text="Generasi", size=(12, 1)),
                 sg.Input(key="_parameter_gen_", size=(16, 1), tooltip="Masukan nilai Integer", disabled=True)],
                [sg.Text(text="Popsize", size=(12, 1)),
                 sg.Input(key="_parameter_popsize_", size=(16, 1), tooltip="Masukan nilai Integer", disabled=True)],
                [sg.Text(text="Crossover Rate", size=(12, 1)),
                 sg.Input(key="_parameter_cr_", size=(16, 1), tooltip="Masukan nilai float antara 0-1", disabled=True)],
                [sg.Text(text="Mutation Rate", size=(12, 1)),
                 sg.Input(key="_parameter_mr_", size=(16, 1), tooltip="Masukan nilai float antara 0-1", disabled=True)]
            ])]
        ]

        column2 = [
            [sg.Button('Mulai', size=(6, 1), disabled=True), sg.ProgressBar(max_value=1, orientation='h', size=(32, 20), key='progress_bar')],
            [sg.Text('-' * 105)],
            [sg.Frame("Hasil Klasifikasi", layout=[
                [sg.Text(text="Nilai Akurasi : ", size=(48, 1), key="_akurasi_")],
                [sg.Text(text="Waktu Komputasi : ", size=(48, 1), key="_waktu_komputasi_")],
                [sg.Text(text="Memori : ", size=(48, 1), key="_memori_")],
                [sg.Text(text="Fitur : ", size=(48, 1), key="_fitur_")]
            ])],
            [sg.Multiline(key="_output_", disabled=True)]
        ]

        layout = [
            [sg.FileBrowse('Muat', size=(6, 1), target='_Path_',
                           initial_folder=r"C:\Users\Momo\PycharmProjects\MkNNAndGeneticAlgorithm\venv\Datasets",
                           file_types=[("csv Files", ".csv")]), sg.Column([
                [sg.Input(key="_Path_", disabled=True, size=(80, 1), enable_events=True)],
                [sg.Text(text="Jumlah Fitur :         ", key="_JumlahFitur_")]
            ])],
            [sg.Text('-' * 187)],
            [sg.Column(column1), sg.VerticalSeparator(), sg.Column(column2)],
        ]

        self.window = sg.Window('kNN, MkNN dan Algen', layout, default_element_size=(40, 1), grab_anywhere=False)


    def __praproses(self, dataset):
        pra = Praproses()
        # praproses data masukan
        # dataset = dataset.sample(n=100, random_state=42)
        x_train = dataset.iloc[:, :(dataset.shape[1] - 1)].replace('?', np.NaN)
        y_label = dataset.iloc[:, (dataset.shape[1] - 1)].values
        x_train = x_train.apply(pd.to_numeric)

        x_train = pra.mean_imputation(x_train)
        x_train = pra.min_max_transform(x_train)

        return x_train, y_label

    def __k_fold_cross_validation(self, clf, x_data, y_label):
        eval = Evaluasi()
        kf = KFold(n_splits=10, random_state=42, shuffle=False)
        evaluasi = []
        i = 1
        for train_index, test_index in kf.split(x_data):
            x_train, x_test, y_train, y_test = x_data[train_index], x_data[test_index] \
                , y_label[train_index], y_label[test_index]
            # mem_start = eval.get_process_memory()
            time_start = eval.get_process_time()
            clf.load_datasets(x_train, y_train, x_test)
            y_predict = clf.predict()
            mem_end = eval.get_process_memory()
            time_end = eval.get_process_time()
            evaluasi.append((i, eval.get_auc_score(y_test, y_predict), (time_end - time_start), mem_end))
            # print(f"Fold {i} y_predict {y_predict}\n y_test {y_test}")
            i += 1
        return evaluasi

    def run(self):

        X = None
        y = None
        while True:
            event, values = self.window.Read()

            file_path = None
            akurasi = 0.0
            waktu_komputasi = 0.0
            memori = 0

            if event == "_Path_":
                self.window.FindElement('_rbkNN_').Update(disabled=False)
                self.window.FindElement('_rbMkNN_').Update(disabled=False)
                self.window.FindElement('_rbGAkNN_').Update(disabled=False)
                self.window.FindElement('_rbGAMkNN_').Update(disabled=False)
                self.window.FindElement('Mulai').Update(disabled=False)
                self.window.FindElement('_parameter_k_').Update(disabled=False)

                try:
                    file_path = Path(values['_Path_'])
                    dataset = pd.read_csv(file_path, sep=';', header=None)
                    X, y = self.__praproses(dataset)
                    # X = self.X_full
                except ValueError:
                    sg.Popup("\tFormat data salah !!\t", title="error")
                else:
                    self.window.FindElement("_JumlahFitur_").Update(f"Jumlah Fitur : {dataset.shape[1]-1}")

            if event == "_rbkNN_"  or event == "_rbMkNN_":
                self.window.FindElement('_parameter_gen_').Update(disabled=True)
                self.window.FindElement('_parameter_popsize_').Update(disabled=True)
                self.window.FindElement('_parameter_cr_').Update(disabled=True)
                self.window.FindElement('_parameter_mr_').Update(disabled=True)
            elif event == "_rbGAkNN_" or event == "_rbGAMkNN_":
                self.window.FindElement('_parameter_gen_').Update(disabled=False)
                self.window.FindElement('_parameter_popsize_').Update(disabled=False)
                self.window.FindElement('_parameter_cr_').Update(disabled=False)
                self.window.FindElement('_parameter_mr_').Update(disabled=False)

            parameter_k = None
            parameter_generasi = None
            parameter_popsize = None
            parameter_cr = None
            parameter_mr = None
            output = ""
            j_fitur_reduksi = X.shape[1] if X is not None else ""
            hasil_klasifikasi = []

            if event == 'Mulai':
                try:
                    parameter_k = int(values['_parameter_k_'])

                    if values["_rbGAkNN_"] == True or values["_rbGAMkNN_"] == True:
                        parameter_generasi = int(values['_parameter_gen_'])
                        parameter_popsize = int(values['_parameter_popsize_'])
                        parameter_cr = float(values['_parameter_cr_'])
                        parameter_mr = float(values['_parameter_mr_'])
                        if parameter_cr < 0 or parameter_cr > 1 or parameter_mr  < 0 or parameter_mr > 1: raise RuntimeError
                except ValueError:
                    sg.Popup("\tMasukan format data yang benar!!!\n Integer untuk parameter k, generasi dan popsize\n float untuk parameter cr dan mr", title="Error")
                except RuntimeError:
                    sg.Popup("Nilai parameter cr dan mr harus diantara angka 0-1", title="error")
                else:
                    self.window.FindElement("progress_bar").UpdateBar(0, 5)
                    if values["_rbkNN_"] == True:
                        # start_time = self.eval.get_process_time()
                        kNN = kNearestNeighbor(k_Neighbors=parameter_k)
                        evaluasi = self.k_fold_cross_validation(kNN, X, y)
                        for eva in evaluasi:
                            akurasi = akurasi + eva[1]
                            waktu_komputasi = waktu_komputasi + eva[2]
                            memori = memori + eva[3]
                            output = output + f"Fold : {eva[0]}, Akurasi : {eva[1]:.4f} Waktu : {eva[2]:.4f} detik , Memori : {eva[3]} byte\n"
                            hasil_klasifikasi.append([f"{eva[1]:4f}", f"{eva[2]:4f}", f"{eva[3]:4f}"])
                        # sortEval = sorted(evaluasi, key=lambda x: x[1], reverse=True)
                        # _, akurasi, waktu_komputasi, memori = sortEval[0]
                        akurasi = akurasi / 10
                        waktu_komputasi = waktu_komputasi / 10
                        memori = memori / 10
                    elif values["_rbMkNN_"] == True:
                        mKNN = modifiedKNearestNeighbor(k_Neighbors=parameter_k)
                        evaluasi = self.k_fold_cross_validation(mKNN, X, y)
                        for eva in evaluasi:
                            akurasi = akurasi + eva[1]
                            waktu_komputasi = waktu_komputasi + eva[2]
                            memori = memori + eva[3]
                            output = output + f"Fold : {eva[0]}, Akurasi : {eva[1]:.4f} Waktu : {eva[2]:.4f} detik , Memori : {eva[3]} byte\n"
                            hasil_klasifikasi.append([f"{eva[1]:4f}", f"{eva[2]:4f}", f"{eva[3]:4f}"])
                        akurasi = akurasi / 10
                        waktu_komputasi = waktu_komputasi / 10
                        memori = memori / 10
                    elif values["_rbGAkNN_"] == True:
                        kNN = kNearestNeighbor(k_Neighbors=parameter_k)
                        ga = AlgoritmaGenetika(classifier=kNN, n_generation=parameter_generasi, pop_size=parameter_popsize,
                                               crossover_rate=parameter_cr, mutation_rate=parameter_mr, n_elitsm=2)

                        ga.load_datasets(X, y)
                        X_reduksi = ga.run()
                        j_fitur_reduksi = X_reduksi.shape[1]
                        evaluasi = self.k_fold_cross_validation(kNN, X_reduksi, y)
                        for eva in evaluasi:
                            akurasi = akurasi + eva[1]
                            waktu_komputasi = waktu_komputasi + eva[2]
                            memori = memori + eva[3]
                            output = output + f"Fold : {eva[0]}, Akurasi : {eva[1]:.4f}, Waktu : {eva[2]:.4f} detik , Memori : {eva[3]} byte\n"
                        akurasi = akurasi / 10
                        waktu_komputasi = waktu_komputasi / 10
                        memori = memori / 10
                        file_name = Path(values["_Path_"]).name.split('.')[0]
                        path = f"C:/Users/Momo/PycharmProjects/MkNNAndGeneticAlgorithm/venv/Datasets/HasilReduksi/kNN-{file_name}-{akurasi:.4f}.csv"
                        dataset_reduksi = np.c_[X_reduksi, y]
                        pd.DataFrame(dataset_reduksi).to_csv(path, header=None, index=None, sep=';')

                    elif values["_rbGAMkNN_"] == True:
                        MkNN = modifiedKNearestNeighbor(k_Neighbors=parameter_k)
                        ga = AlgoritmaGenetika(classifier=MkNN, n_generation=parameter_generasi, pop_size=parameter_popsize,
                                               crossover_rate=parameter_cr, mutation_rate=parameter_mr, n_elitsm=2)

                        ga.load_datasets(X, y)
                        X_reduksi= ga.run()
                        j_fitur_reduksi = X_reduksi.shape[1]
                        evaluasi = self.k_fold_cross_validation(MkNN, X_reduksi, y)
                        for eva in evaluasi:
                            akurasi = akurasi + eva[1]
                            waktu_komputasi = waktu_komputasi + eva[2]
                            memori = memori + eva[3]
                            output = output + f"Fold : {eva[0]}, Akurasi : {eva[1]:.4f}, Waktu : {eva[2]:.4f} detik , Memori : {eva[3]} byte\n"
                        akurasi = akurasi / 10
                        waktu_komputasi = waktu_komputasi / 10
                        memori = memori / 10
                        file_name = Path(values["_Path_"]).name.split('.')[0]
                        path = f"C:/Users/Momo/PycharmProjects/MkNNAndGeneticAlgorithm/venv/Datasets/HasilReduksi/MkNN-{file_name}-{akurasi:.4f}.csv"
                        dataset_reduksi = np.c_[X_reduksi, y]
                        pd.DataFrame(dataset_reduksi).to_csv(path, header=None, index=None, sep=';')

                    self.window.FindElement("_akurasi_").Update(f"Akurasi : {akurasi:.4f}")
                    self.window.FindElement("_waktu_komputasi_").Update(f"Waktu Komputasi : {waktu_komputasi:.4f} detik")
                    self.window.FindElement("_memori_").Update(f"Memori : {memori} byte")
                    self.window.FindElement("_fitur_").Update(f"Fitur : {j_fitur_reduksi}")
                    self.window.FindElement("_output_").Update(output)
                    pd.DataFrame(hasil_klasifikasi).to_csv("Hasil_Klasifikasi.csv", sep=';', header=None, index=None)

            if event is None or event == 'Exit':
                self.window.Close()
                break
            print(event, values)


if __name__ == "__main__":
    form = MainForm()
    form.run()