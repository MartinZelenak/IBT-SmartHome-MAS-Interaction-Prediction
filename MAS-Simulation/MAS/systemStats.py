"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: The SystemStats class for collecting and plotting the multi-agent system statistics.
Date: 2025-05-14
"""


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from typing import Dict, List
import logging

logger = logging.getLogger('MAS.system')

def SystemStatsSingleton(cls):
    instance = None

    def wrapper(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return wrapper

@SystemStatsSingleton
class SystemStats:
    def __init__(self):
        self.losses: Dict[str, List[float]] = {}
        self.predictions: Dict[str, List[float]] = {}
        self.actuals: Dict[str, List[float]] = {}
        self.users_locations: Dict[str, List[int]] = {}

        self.fig: Figure|None = None

    def clear(self):
        self.losses.clear()
        self.predictions.clear()
        self.actuals.clear()
        self.users_locations.clear()

    def live_plot(self):
        plt.ion()
        self.fig = plt.figure()
        self.update_figure()
    
    def _synchronize_data_lengths(self):
        """
        Ensures all data series have the same length by padding shorter ones with their last values.
        This keeps all subplots synchronized in time/steps.
        """
        # Find the maximum length among all data series
        max_length = 0
        for data_dict in [self.users_locations, self.actuals, self.predictions, self.losses]:
            for _, data in data_dict.items():
                max_length = max(max_length, len(data))
        
        # Extend shorter series by repeating their last values
        for data_dict in [self.users_locations, self.actuals, self.predictions, self.losses]:
            for key, data in data_dict.items():
                if len(data) < max_length and len(data) > 0:
                    last_value = data[-1]
                    data.extend([last_value] * (max_length - len(data)))
        
    def update_figure(self):
        if self.fig is None:
            return
            
        # Synchronize data lengths before plotting
        self._synchronize_data_lengths()
        
        n_subplots = len(self.losses) + len(self.predictions) + len(self.actuals) + len(self.users_locations)
        cur_subplot = 1

        self.fig.clear()  # Clear the figure before updating

        self.fig.set_size_inches(8, 6 * n_subplots)  # Set the figure size based on the number of subplots
        subplot = None
        for user, data in self.users_locations.items():
            subplot = self.fig.add_subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
            subplot.set_title(f'{user} location')
            cur_subplot += 1
            subplot.plot(data)

        for device, data in self.actuals.items():
            subplot = self.fig.add_subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
            subplot.set_title(f'{device} actual')
            cur_subplot += 1
            subplot.plot(data)

        for device, data in self.predictions.items():
            subplot = self.fig.add_subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
            subplot.set_title(f'{device} prediction')
            cur_subplot += 1
            subplot.plot(data)

        for device, data in self.losses.items():
            subplot = self.fig.add_subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
            subplot.set_title(f'{device} loss')
            cur_subplot += 1
            subplot.plot(data)

        self.fig.tight_layout()  # Adjust the spacing between subplots

        self.fig.canvas.draw()  # Redraw the figure

    def live_plot_process(self, timeout: float = 0.1):
        plt.pause(timeout)

    def save_plots(self, png_path: str):
        import os
        dirname = os.path.dirname(png_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        if self.fig is None:
            self.fig = plt.figure()
        self.update_figure()
        logger.info(f'Saving stats plot to {png_path}')
        plt.savefig(png_path)
