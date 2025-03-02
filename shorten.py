import wave
import sys

print("here")

def trim_wav(input_path, output_path):
    with wave.open(input_path, 'rb') as in_wav:
        params = in_wav.getparams()
        total_frames = in_wav.getnframes()
        # Calculate 1/20 of the frames
        keep_frames = total_frames // 4
        frames = in_wav.readframes(keep_frames)
    with wave.open(output_path, 'wb') as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python trim.py input.wav [output.wav]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    # Default output file name if not provided
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.wav', '_trimmed.wav')
    
    trim_wav(input_path, output_path)
    print(f"Created trimmed wav file: {output_path}")
