from scipy.io import wavfile
import pyroomacoustics as pra
import numpy as np

def stitch(audio_files, coordinates, room_dimensions, output_path):
    """
    Create spatial audio by simulating sound propagation in a virtual room.

    Parameters:
    - audio_files (np.array): Array of audio data.
    - coordinates (np.array): Array of 3D coordinates corresponding to the positions of audio data.
    - room_dimensions (list): List containing the dimensions of the virtual room in meters [length, width, height].
    - output_path (str): File path where the spatial audio will be saved.

    Returns:
    None

    Note:
    - The function assumes all audio files have the same sampling rate.
    - The listener's position (receiver) is set at the center of the room's ground level.
    - The simulated spatial audio is saved to the specified output path.
    - If an error occurs during simulation, an error message is printed.

    """
    # Room parameters
    fs, _ = wavfile.read(audio_files[0])  # Assuming all audio files have the same sampling rate

    # Create an empty room
    room = pra.ShoeBox(room_dimensions)

    # Place sources in the room
    for i, coord in enumerate(coordinates):
        audio = audio_files[i]

        my_source = pra.SoundSource(coord, signal=audio)
        room.add_source(my_source)

    # Set receiver position (listener's position)
    receiver_pos = [room_dimensions[0] / 2, room_dimensions[1] / 2, 0]
    room.add_microphone(receiver_pos)

    # Compute RIRs
    room.compute_rir()

    # Simulate sound propagation in the room
    room.simulate()

    # Retrieve the simulated spatial audio signal received by the microphone
    output_signal = room.mic_array.signals[0]

    # Save the spatial audio to a file
    wavfile.write(output_path, fs, output_signal)
