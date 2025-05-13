from matplotlib import pyplot as plt
from processing import Processing

class visualize:
    def __init__(self):
        self._data = {}
        self._raw_data = {}
        self.p = Processing()

    def get_data(self, subject, run):
        self._raw_data = self.p.load_subject_run(subject, run)
        self._data = self._raw_data.copy().filter(7., 30., fir_design='firwin')
    
    def visualize(self):
        subject = input("subject between 1 and 10 : ")
        try :
            subject = int(subject)
        except:
            return False
        subject = f"S{subject:03}"
        valid_subject = [f"S{i:03}" for i in range(1, 11)]
        if subject not in valid_subject:
            return False
        run = input("Run 4, 6, 7, 8, 9 or 10 : ")
        valid_runs = [4, 6, 7, 8, 9, 10]
        try :
            run = int(run)
        except:
            return False
        if run not in valid_runs:
            return False
        self.get_data(subject, run)
        self._raw_data.plot(scalings='auto', title='EEG Raw', show=True)
        psd = self._raw_data.compute_psd(fmin=0, fmax=50)
        psd.plot()
        self._data.plot(scalings='auto', title='EEG filtered', show=True)
        psdf = self._data.compute_psd(fmin=0, fmax=50)
        psdf.plot()


v = visualize()
v.visualize()
plt.show()