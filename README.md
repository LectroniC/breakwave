# WaveGuard Defense

Code for our USENIX 21 paper [WaveGuard: Understanding and Mitigating Audio Adversarial Examples
](https://www.usenix.org/system/files/sec21fall-hussain.pdf).

Audio Examples from paper [Audio Examples](https://waveguard.herokuapp.com/)

## Requirements

``pip install -r requirements.txt``

Also install Deepspeech following the same instructions as in [https://github.com/carlini/audio_adversarial_examples](https://github.com/carlini/audio_adversarial_examples) to evaluate the defense. 

## Running the defense

Running the defense on a directory of wav files (sampled at 16KHz): 

```
python Defender/defender_multiple.py --in_dir <PATH TO DIR WITH WAV FILES> --out_base <PATH TO OUTPUT DIR> --defender_type DEFENDER_TYPE --defender_hp DEFENDER_HYPERPARAMETER;
```

Defender type can be ``lpc, mel_heuristic, filter_power, quant, downsample_upsample``. defender_hp corresponds to number of lpc coeffecients, mel bins, quantization bits, downsampling rare for ``lpc, mel_heuristic, quant, downsample_upsample`` respectively.


Running the defense on a directory of wav files (sampled at 16KHz): 

```
python Defender/defender_multiple.py --in_dir <PATH TO DIR WITH WAV FILES> --out_base <PATH TO OUTPUT DIR> --defender_type DEFENDER_TYPE --defender_hp DEFENDER_HYPERPARAMETER;
```

Defender type can be ``lpc, mel_heuristic, filter_power, quant, downsample_upsample``. defender_hp corresponds to number of lpc coeffecients, mel bins, quantization bits, downsampling rare for ``lpc, mel_heuristic, quant, downsample_upsample`` respectively.


## Evaluating the AUC

```python evaluate_detector.py --in_orig <DIR CONTAINING ORIGINAL UNDEFENDED AUDIO> --in_orig_def <DIR CONTAINING ORIGINAL DEFENDED AUDIO> --in_adv <DIR CONTAINING ADVERSARIAL UNDEFENDED AUDIO> --in_orig <DIR CONTAINING ADVERSARIAL DEFENDED AUDIO>```

## Citing our work

```
@inproceedings{hussain2021waveguard,
  title={WaveGuard: Understanding and Mitigating Audio Adversarial Examples},
  author={Hussain, Shehzeen and Neekhara, Paarth and Dubnov, Shlomo and McAuley, Julian and Koushanfar, Farinaz},
  booktitle={USENIX Security 21},
  year={2021}
}
```
