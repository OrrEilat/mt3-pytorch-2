from inference import InferenceHandler

handler = InferenceHandler("./pretrained")
handler.inference(
    "/home/chou150/depot/datasets/maestro/maestro_with_mistakes/score/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav/mix.wav",
    outpath="/home/chou150/code/mt3-pytorch-2/output.mid",
)
