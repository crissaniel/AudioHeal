# AudioHeal
AudioHeal is a Python program that reconstructs missing or corrupted audio segments in WAV files using numerical interpolation methods.
## Features 
- Supports clean and corrupted .wav audio files
- Simulation mode for testing and repair mode for real corruption
- Linear, Lagrange, and Newton interpolation methods
- User-defined polynomial order with automatic context adjustment
- Waveform visualization of original, damaged, and healed signals
  
## Usage
python audioheal.py input.wav --gap_start <start> --gap_end <end> [--simulate] [--method lagrange|newton|linear] [--plot]
