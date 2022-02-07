"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import pyroomacoustics as pra
try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err
methods = ["ism", "hybrid"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[1],
        help="Simulation method to use",
    )
    args = parser.parse_args()

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.3  # seconds
    room_dim = [10, 7.5, 3.5]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio = wavfile.read("/home/jzhou3083/work/pyroomacoustics/examples/samples/guitar_16k.wav")

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    print(max_order)

    # Create the room
    # if args.method == "ism":
    #     room = pra.ShoeBox(
    #         room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order, air_absorption=True, ray_tracing = False
    #     )
    # elif args.method == "hybrid":
    #     room = pra.ShoeBox(
    #         room_dim,
    #         fs=fs,
    #         materials=pra.Material(e_absorption),
    #         max_order=-1,
    #         ray_tracing=True,
    #         air_absorption=True,
    #     )

    path_to_musis_stl_file = "/home/jzhou3083/work/pyroomacoustics/examples/data/CR2.stl"
    # with numpy-stl
    the_mesh = mesh.Mesh.from_file(path_to_musis_stl_file)
    ntriang, nvec, npts = the_mesh.vectors.shape

    size_reduc_factor = 1.0  # to get a realistic room size (not 3km)
    material = pra.Material(energy_absorption="mat_CR2_concrete", scattering="mat_CR2_concrete")
    # create one wall per triangle
    walls = []
    for w in range(ntriang):
        walls.append(
            pra.wall_factory(
                the_mesh.vectors[w].T / size_reduc_factor,
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )

    room = (
        pra.Room(walls, fs=8000, max_order=1, ray_tracing=True, air_absorption=True,)
        .add_source([-2.0, 2.0, 1.8])
    )

    # place the source in the room
    room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)

    # define the locations of the microphones
    mic_locs = np.c_[
        [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    room.mic_array.to_wav(
        "/home/jzhou3083/work/pyroomacoustics/examples/samples/guitar_16k_reverb_{only_ray_t}.wav",
        norm=True,
        bitdepth=np.int16,
    )

    # measure the reverberation time
    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
    print("The measured RT60 is {}".format(rt60[1, 0]))

    # Create a plot
    plt.figure()

    # plot one of the RIR. both can also be plotted using room.plot_rir()
    rir_1_0 = room.rir[1][0]
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
    plt.title("The RIR from source 0 to mic 1")
    plt.xlabel("Time [s]")

    # plot signal at microphone 1
    plt.subplot(2, 1, 2)
    plt.plot(room.mic_array.signals[1, :])
    plt.title("Microphone 1 signal")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()
