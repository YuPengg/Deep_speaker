#!/bin/bash

sox infer_audio.wav noise-infer_audio.wav trim 0 2.00
sox noise-infer_audio.wav -n noiseprof noise.prof
sox infer_audio.wav infer_audio_clean.wav noisered noise.prof 0.1

exit

