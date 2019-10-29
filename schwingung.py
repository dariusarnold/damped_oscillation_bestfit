#!/usr/bin/env python3

"""
Script shows full seismogram and allows to graphically select part of it.
A damped oscillation A * exp(-d*t) * sin(phi + omega*t) is fitted by least
squares to the selected part. The parameters of the oscillation are then shown.

Pass seismogram (sg2 file) as first parameter.
"""

import sys

MIN_PYTHON_VERSION = (3, 6)
if sys.version_info < MIN_PYTHON_VERSION:
    sys.exit("This script needs python >=3.6")

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from pathlib import Path
import collections
from typing import Tuple, Optional
from matplotlib.offsetbox import AnchoredText
from matplotlib.artist import Artist, get
import matplotlib.backend_bases


def damped_oscillator(t: np.ndarray, A: float, delta: float, phi: float,
                        omega: float) -> np.ndarray:
    """
    Get amplitude of damped oscillator.
    :param A: amplitude
    :param t: 1D array of timesteps
    :param delta: Dampening constant
    :param phi: phase of sine
    :return:
    """
    # make copy here to not modify original array by shifting
    t = np.array(t - t[0])
    return A * np.exp(-delta * t) * np.sin(phi + t * omega)


def get_index_of_closest_value(array: np.ndarray, value: float):
    return (np.abs(array - value)).argmin()


def get_base_plot():
    fig, ax = plt.subplots()
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Amplitude")
    return fig, ax


class DataSelectionPlot:
    def __init__(self, t: np.ndarray, x: np.ndarray, linecolor="k"):
        self.fig, self.ax = get_base_plot()
        self.t = t
        self.data = collections.deque(maxlen=2)
        self.linecolor = linecolor
        self.ax.plot(t, x)
        self.ax.set_title(
            "Select two points by right clicking, last two clicked points are selected.\n"
            "Close plot after selecting points.")
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        plt.show()

    @staticmethod
    def _get_x_position(line: Artist) -> float:
        """
        Helper function so the deque only has to save the line, which already knows its x position
        :param line:
        :return: X position of line in data coordinates
        """
        return get(line, "xdata")[0]

    def _onclick(self, event) -> None:
        if event.button == matplotlib.backend_bases.MouseButton.RIGHT:
            self.add_new_point(event.xdata)

    def add_new_point(self, x_coordinate: float) -> None:
        """
        Add new line at x coordinate, remove oldest line if there are 2 lines already.
        :param x_coordinate: x position of new line in data coordinates.
        """
        if len(self.data) == 2:
            self.data.pop().remove()
        line = self.ax.axvline(x_coordinate, color=self.linecolor, linestyle="-.")
        self.data.appendleft(line)
        plt.draw()

    def get_points(self) -> Tuple[int, int]:
        """
        Return indices of the two selected points in the t/x arrays.
        :return: tuple of left index, right index.
        """
        index0 = get_index_of_closest_value(t, self._get_x_position(self.data[0]))
        index1 = get_index_of_closest_value(t, self._get_x_position(self.data[1]))
        if index1 < index0:
            index0, index1 = index1, index0
        return index0, index1


def select_data_from_recording(t: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    # use deque of size 2 as a ringbuffer to store x position of last two clicks
    # in data coordinates
    plot = DataSelectionPlot(t, x)
    return plot.get_points()


def plot_seismo_and_best_fit_curve(t: np.ndarray, x: np.ndarray,
                                   params: Tuple[float, float, float, float]):
    amplitude, dampening, phase, frequency = params
    fig, ax = get_base_plot()
    ax.plot(t, damped_oscillator(t, *params), label="Best fit curve")
    ax.plot(t, x, label="Seismogram")
    ax.legend()
    ax.set_title(fr"damped oscillation $x(t) = A * exp(-\delta t) * sin(\phi + \omega * t)$")
    param_string = "\n".join((fr"$A = ${amplitude:.2e}",
                              fr"$\delta = ${dampening:.2f}",
                              fr"$\phi = ${phase:.2f}",
                              fr"$\omega = ${frequency:.2f}"))
    text = AnchoredText(param_string, loc="upper left")
    ax.add_artist(text)
    plt.show()


DampedOscillationParams = collections.namedtuple("dampedOscillationParams",
                                                   "amplitude dampening phase frequency")


def fit_curve_to_data(t: np.ndarray, x: np.ndarray) -> DampedOscillationParams:
    # create some initial guesses for oscillation parameters
    amplitude = np.max(x)
    initial_guess = np.array((amplitude, 0.2, 0, 1.))
    params, _ = sp.curve_fit(damped_oscillator, t, x, p0=initial_guess, maxfev=300 * len(t))
    return DampedOscillationParams(*params)


def load_data(path: Path) -> np.ndarray:
    try:
        return np.genfromtxt(path, skip_header=1, skip_footer=1, unpack=True)
    except Exception as e:
        sys.exit(f"Couldn't parse file {path}. Error: {e}")


def handle_arguments() -> Path:
    if len(sys.argv) < 2:
        sys.exit("Specify path to sg2 file as command line argument.")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"Path {path} does not exist.")
    if not path.is_file():
        sys.exit(f"Given path {path} is not a file.")
    return path


if __name__ == '__main__':
    path = handle_arguments()
    t, x = load_data(path)
    start_index, stop_index = select_data_from_recording(t, x)
    t = t[start_index:stop_index]
    x = x[start_index:stop_index]
    params = fit_curve_to_data(t, x)
    plot_seismo_and_best_fit_curve(t, x, params)
