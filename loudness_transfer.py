import torch
from torchaudio.io import play_audio

from dataset_tool import compute_loudness
from IPython.display import Audio
import pickle
import librosa as li
from noisebandnet.model import NoiseBandNet
import torch.nn.functional as F
import sounddevice as sd

def load_audio(path, fs, max_len, norm=True):
    x = li.load(path, sr=fs, mono=True)[0]
    if max_len > 0:
        if len(x)>max_len:
            x = x[:max_len]
    if norm:
        x = li.util.normalize(x)
    return x

device = 'cuda'

TRAIN_PATH = 'trained_models/drill/2025_02_04_19_15_53'
MODEL_PATH = f'{TRAIN_PATH}/model_4999.ckpt'
CONFIG_PATH = f'{TRAIN_PATH}/config.pickle'

#path to the target loudness sound
TARGET_LOUDNESS_AUDIO_PATH = 'dataset_audio/beat/beat.wav'

with (open(CONFIG_PATH, "rb")) as f:
    config = pickle.load(f)
FS = config.sampling_rate

x_audio = load_audio(path=TARGET_LOUDNESS_AUDIO_PATH, fs=FS, max_len=2**19)

x_audio = torch.from_numpy(x_audio).unsqueeze(0)


#sd.play(x_audio[0],FS)
#sd.wait()
#play_audio(x_audio,FS)
#Audio(x_audio[0], rate=FS)

synth = NoiseBandNet(hidden_size=config.hidden_size, n_band=config.n_band, synth_window=config.synth_window, n_control_params=config.n_control_params).to(device).float()

synth.load_state_dict(torch.load(MODEL_PATH))

loudness, _, _ = compute_loudness(audio_data=x_audio, sampling_rate=FS)
#user-defined scaling
loudness += 0.25
loudness = loudness.unsqueeze(0).float()
loudness = F.interpolate(input=loudness, scale_factor=1/config.synth_window, mode='linear').permute(0,2,1).float()
control_params = [loudness.to(device)]

print(control_params[0].shape)

with torch.no_grad():
    y_audio = synth(control_params)

print("Původní tvar y_audio:", y_audio.shape)

sd.play(y_audio[0].squeeze().detach().cpu().numpy(),FS)
sd.wait()