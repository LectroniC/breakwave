# New README

The results are in the results.zip except for the transferability experiment.

Use python `3.6` (For example, `conda create --name my_env python=3.6`)

~~For windows you need to install vs standalone compiler. Details in [here](https://pypi.org/project/lws/).~~
You can't directly deploy this on windows.

You need sox for pysndfx. And sox doesn't support windows therefore can't run natively on windows. You can't run this on windows.

On Debian-based systm, you need `sudo apt-get install build-essential`. 

And install git lfs. (https://askubuntu.com/questions/799341/how-to-install-git-lfs-on-ubuntu-16-04)

And install `anaconda`.

And install SoX.

And `pip install numpy==1.16.2`

Then if you choose to install all the dependencies for WaveGuard and DeepSpeech

You need:

`pip install numba==0.48`

`pip install pysndfx==0.3.6`

Or you can use the `environment.yml` to set up the environment with `conda env update --file environment.yml --prune`.

Please also follow the setup in [WaveGuard](https://github.com/shehzeen/waveguard_defense) and the [CW attack](https://github.com/carlini/audio_adversarial_examples).

To run the code:
`python Defender/defender_multiple.py --in_dir input_audios --out_base output_audios --defender_type mel_heuristic --defender_hp 80`

example_audios downloaded from the [Audio Examples](https://waveguard.herokuapp.com/) page.

Choose `defender_hp=80` since this is the choice from the paper.

# Acknowledgement
Some code are borrowed from the following repo:

- [WaveGuard](https://github.com/shehzeen/waveguard_defense)
- [Audio CW attack](https://github.com/carlini/)

# Contact

[Licheng Luo](ll6@illinois.edu) 
