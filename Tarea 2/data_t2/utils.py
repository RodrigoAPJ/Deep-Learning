import numpy as np
import librosa
import logging, copy
DEFAULT_RNG = np.random.default_rng()
logger = logging.getLogger(__name__)

def wav_a_mel_spectrogram(wav,sampling_rate=16000,mel_window_length=25,mel_window_step=10,mel_n_channels = 40):
    #recibe un wav y retorna el espectrograma
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

def embed_utterance(
    wav: np.ndarray
):
    #retorna el espectrograma
    mel = wav_a_mel_spectrogram(wav)
    return mel


class Audio(object): #clase audio, tiene algunos metodos utiles
    def __init__(self, data=None, sampleRate=None):
        self.data = data
        self.sampleRate = sampleRate
        if (data is not None) and (sampleRate is not None):
            self.duration = len(self.data) / self.sampleRate
        else:
            self.duration = None
        self._metadata = {}


    def read(self, datafile, sampleRate=16000): #lee audio
        self.data, self.sampleRate = librosa.load(
            path=datafile, sr=sampleRate, res_type="kaiser_fast"
        )
        self.data = np.nan_to_num(self.data)
        self.duration = len(self.data) / self.sampleRate
        self._metadata["datafile"] = datafile
        return self

    def clip(self, start, end): #extrae ventana del audio entre start y end segundos
        startIdx = int(start * self.sampleRate)
        endIdx = int(end * self.sampleRate)
        resAudio = Audio(self.data[startIdx:endIdx], self.sampleRate)
        resAudio._metadata = copy.deepcopy(self._metadata)
        if "history" in resAudio._metadata.keys():
            resAudio._metadata["history"].append(
                {"action": "clip", "values": [start, end]}
            )
        else:
            resAudio._metadata["history"] = [{"action": "clip", "values": [start, end]}]
        return resAudio

    def addNoise(self, stdFactor): # pueden agregar ruido
        if stdFactor == 0:
            return copy.deepcopy(self)
        stdAudio = np.std(self.data)
        stdNoise = stdFactor * stdAudio
        noiseData = np.random.randn(len(self.data)) * stdNoise
        resAudio = Audio(np.nan_to_num(self.data + noiseData), self.sampleRate)
        resAudio._metadata = copy.deepcopy(self._metadata)
        if "history" in resAudio._metadata.keys():
            resAudio._metadata["history"].append(
                {"action": "addNoise", "value": stdFactor}
            )
        else:
            resAudio._metadata["history"] = [{"action": "addNoise", "value": stdFactor}]
        return resAudio