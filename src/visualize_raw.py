"""
Script V.1.1: Visualiser les données EEG brutes et filtrées

Usage:
    python src/visualize_raw.py
    
Permet de:
1. Charger données brutes d'un sujet/run
2. Visualiser le signal brut
3. Visualiser le spectre avant filtrage
4. Appliquer filtrage (1-15 Hz bandpass + 60 Hz notch)
5. Visualiser signal filtré et spectre après
"""

from matplotlib import pyplot as plt
import mne
import numpy as np
from processing import load_subject_run

class RawDataVisualizer:
    def __init__(self):
        self.raw_data = None
        self.filtered_data = None
    
    def load_data(self, subject_id, run_id):
        """Charge les données brutes d'un sujet et d'un run"""
        print(f"\n[*] Chargement S{subject_id:03d}, run {run_id}...")
        subject_str = f"S{subject_id:03d}"
        self.raw_data = load_subject_run(subject_str, run_id)
        print(f"    Shape: {self.raw_data.get_data().shape}")
        print(f"    Channels: {self.raw_data.ch_names}")
        print(f"    Sampling rate: {self.raw_data.info['sfreq']} Hz")
        return True
    
    def visualize_raw(self):
        """Affiche le signal brut"""
        if self.raw_data is None:
            print("Erreur: données non chargées")
            return False
        
        print("\n[*] Visualisation du signal brut...")
        self.raw_data.plot(
            scalings='auto',
            title='EEG Raw Data - Before Preprocessing',
            show=True,
            n_channels=7
        )
        return True
    
    def visualize_raw_psd(self):
        """Affiche le spectre de puissance before filtering"""
        if self.raw_data is None:
            return False
        
        print("[*] Calcul du spectre de puissance (raw)...")
        psd = self.raw_data.compute_psd(fmin=0, fmax=50)
        fig = psd.plot(picks='eeg', exclude='bads', average=False)
        fig.set_figwidth(12)
        fig.suptitle('PSD - Raw Signal')
        plt.xlim(0, 50)
        plt.show()
        return True
    
    def apply_filtering(self, l_freq=1.0, h_freq=15.0, notch_freq=60):
        """Applique le filtrage: bandpass 1-15 Hz + notch 60 Hz"""
        if self.raw_data is None:
            return False
        
        print(f"\n[*] Filtrage: notch {notch_freq} Hz, bandpass {l_freq}-{h_freq} Hz...")
        self.filtered_data = self.raw_data.copy()
        
        # Notch filter (60 Hz)
        self.filtered_data = self.filtered_data.notch_filter(notch_freq, method="iir")
        
        # Bandpass filter
        self.filtered_data = self.filtered_data.filter(
            l_freq, h_freq,
            fir_design='firwin',
            skip_by_annotation="edge"
        )
        
        print("    ✓ Filtrage appliqué")
        return True
    
    def visualize_filtered(self):
        """Affiche le signal filtré"""
        if self.filtered_data is None:
            print("Erreur: données filtrées non disponibles")
            return False
        
        print("[*] Visualisation du signal filtré...")
        self.filtered_data.plot(
            scalings='auto',
            title='EEG Filtered (1-15 Hz bandpass + 60 Hz notch)',
            show=True,
            n_channels=7
        )
        return True
    
    def visualize_filtered_psd(self):
        """Affiche le spectre de puissance after filtering"""
        if self.filtered_data is None:
            return False
        
        print("[*] Calcul du spectre de puissance (filtré)...")
        psd = self.filtered_data.compute_psd(fmin=0, fmax=50)
        fig = psd.plot(picks='eeg', exclude='bads', average=False)
        fig.set_figwidth(12)
        fig.suptitle('PSD - Filtered Signal (1-15 Hz)')
        plt.xlim(0, 50)
        plt.show()
        return True
    
    def compare_raw_vs_filtered(self):
        """Crée une comparaison visuelle before/after"""
        if self.raw_data is None or self.filtered_data is None:
            print("Erreur: données incomplètes")
            return False
        
        print("[*] Comparaison raw vs filtré...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Pick un canal pour la demo
        ch_idx = 0
        ch_name = self.raw_data.ch_names[ch_idx]
        
        # Signal brut
        raw_signal = self.raw_data.get_data(picks=ch_idx)[0]
        time_raw = np.arange(raw_signal.shape[0]) / self.raw_data.info['sfreq']
        axes[0, 0].plot(time_raw[:2000], raw_signal[:2000])
        axes[0, 0].set_title(f'Raw Signal - {ch_name}')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude (uV)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Signal filtré
        filt_signal = self.filtered_data.get_data(picks=ch_idx)[0]
        time_filt = np.arange(filt_signal.shape[0]) / self.filtered_data.info['sfreq']
        axes[0, 1].plot(time_filt[:2000], filt_signal[:2000], color='green')
        axes[0, 1].set_title(f'Filtered Signal (1-15 Hz) - {ch_name}')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude (uV)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PSD raw
        psd_raw = self.raw_data.compute_psd(fmin=0, fmax=50)
        psds_raw, freqs = psd_raw.get_data(return_freqs=True)
        axes[1, 0].plot(freqs, 10 * np.log10(psds_raw[ch_idx]))
        axes[1, 0].axvline(1, color='red', linestyle='--', alpha=0.5, label='1 Hz')
        axes[1, 0].axvline(15, color='red', linestyle='--', alpha=0.5, label='15 Hz')
        axes[1, 0].axvline(60, color='orange', linestyle='--', alpha=0.5, label='60 Hz (notch)')
        axes[1, 0].set_title(f'PSD - Raw - {ch_name}')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power (dB)')
        axes[1, 0].set_xlim(0, 50)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # PSD filtered
        psd_filt = self.filtered_data.compute_psd(fmin=0, fmax=50)
        psds_filt, freqs = psd_filt.get_data(return_freqs=True)
        axes[1, 1].plot(freqs, 10 * np.log10(psds_filt[ch_idx]), color='green')
        axes[1, 1].axvline(1, color='red', linestyle='--', alpha=0.5, label='1 Hz')
        axes[1, 1].axvline(15, color='red', linestyle='--', alpha=0.5, label='15 Hz')
        axes[1, 1].set_title(f'PSD - Filtered - {ch_name}')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Power (dB)')
        axes[1, 1].set_xlim(0, 50)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eeg_preprocessing_comparison.png', dpi=150)
        print("    ✓ Graphique sauvegardé: eeg_preprocessing_comparison.png")
        plt.show()
        
        return True
    
    def run_full_visualization(self):
        """Lance la visualisation complète"""
        print("\n" + "="*70)
        print("EEG DATA VISUALIZATION - V.1.1 Preprocessing & Exploration")
        print("="*70)
        
        # Demander subject et run
        while True:
            try:
                subj_input = input("\n[?] Subject ID (1-109): ").strip()
                subject_id = int(subj_input)
                if not (1 <= subject_id <= 109):
                    print("    ❌ Subject doit être entre 1 et 109")
                    continue
                break
            except ValueError:
                print("    ❌ Entrée invalide")
        
        # Runs disponibles: 3,4,6,7,8,11,12,13,14
        while True:
            try:
                run_input = input("[?] Run ID (3,4,6,7,8,11,12,13,14): ").strip()
                run_id = int(run_input)
                if run_id not in [3, 4, 6, 7, 8, 11, 12, 13, 14]:
                    print("    ❌ Run invalide")
                    continue
                break
            except ValueError:
                print("    ❌ Entrée invalide")
        
        # Charger et visualiser
        if self.load_data(subject_id, run_id):
            print("\n[1/5] Visualisation du signal brut...")
            self.visualize_raw()
            
            print("\n[2/5] PSD du signal brut...")
            self.visualize_raw_psd()
            
            print("\n[3/5] Application du filtrage...")
            self.apply_filtering(l_freq=1.0, h_freq=15.0)
            
            print("\n[4/5] Visualisation du signal filtré...")
            self.visualize_filtered()
            
            print("\n[5/5] PSD du signal filtré...")
            self.visualize_filtered_psd()
            
            print("\n[6/5] BONUS: Comparaison raw vs filtré...")
            self.compare_raw_vs_filtered()
            
            print("\n" + "="*70)
            print("✓ Visualisation terminée!")
            print("="*70)


if __name__ == "__main__":
    viz = RawDataVisualizer()
    viz.run_full_visualization()
