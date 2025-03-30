import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from typing import Dict, List

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

    # def live_plot_blocking(self):
    #     self.fig = plt.figure()
    #     def update(frame):
    #         n_subplots = len(self.losses) + len(self.predictions) + len(self.actuals) + len(self.users_locations)
    #         cur_subplot = 1

    #         for user, data in self.users_locations.items():
    #             subplot = plt.subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
    #             subplot.set_title(f'{user} location')
    #             cur_subplot += 1
    #             subplot.plot(data)

    #         for device, data in self.actuals.items():
    #             subplot = plt.subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
    #             subplot.set_title(f'{device} actual')
    #             cur_subplot += 1
    #             subplot.plot(data)

    #         for device, data in self.predictions.items():
    #             subplot = plt.subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
    #             subplot.set_title(f'{device} prediction')
    #             cur_subplot += 1
    #             subplot.plot(data)

    #         for device, data in self.losses.items():
    #             subplot = plt.subplot(n_subplots, 1, cur_subplot, sharex=subplot if cur_subplot > 1 else None)
    #             subplot.set_title(f'{device} loss')
    #             cur_subplot += 1
    #             subplot.plot(data)

    #     self.fig_ani = animation.FuncAnimation(self.fig, update, interval=1000)
    #     plt.show()

    def live_plot(self):
        plt.ion()
        self.fig = plt.figure()
        self.update_figure()
        
    def update_figure(self):
        if self.fig is None:
            return
        
        n_subplots = len(self.losses) + len(self.predictions) + len(self.actuals) + len(self.users_locations)
        cur_subplot = 1

        self.fig.clear()  # Clear the figure before updating

        self.fig.set_size_inches(8, 6 * n_subplots)  # Set the figure size based on the number of subplots

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

    def live_plot_process(self, timeout: int = 0.1):
        plt.pause(timeout)

    def save_plots(self, png_path: str):
        import os
        dirname = os.path.dirname(png_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        if self.fig is None:
            self.fig = plt.figure()
        self.update_figure()
        print(f'Saving plot to {png_path}')
        plt.savefig(png_path)

    def save_data(self, path: str):
        # TODO: Save the system stats data
        pass
        # import os
        # dirname = os.path.dirname(path)
        # if dirname and not os.path.exists(dirname):
        #     os.makedirs(dirname)
        # with open(path, 'w') as f:
        #     f.write(f'users_locations: {self.users_locations}\n')
        #     f.write(f'losses: {self.losses}\n')
        #     f.write(f'predictions: {self.predictions}\n')
        #     f.write(f'actuals: {self.actuals}\n')


if __name__ == '__main__':
    import time
    stats = SystemStats()
    stats.users_locations['user1'] = [1, 2, 3, 4, 5]
    stats.losses['device1'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats.predictions['device1'] = [1, 2, 3, 4, 5]
    stats.actuals['device1'] = [1, 2, 3, 4, 5]

    stats.live_plot()
    plot = False
    while True:
        stats.users_locations['user1'].append(2)
        stats.actuals['device1'].append(stats.actuals['device1'][-1] + 1)
        stats.predictions['device1'].append(stats.predictions['device1'][-1] + 1)
        stats.losses['device1'].append(stats.losses['device1'][-1])
        
        if plot:
            stats.update_figure()
        plot = not plot
        stats.live_plot_process(0.5)
        # time.sleep(0.5)