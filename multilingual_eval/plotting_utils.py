import numpy as np
import matplotlib.pyplot as plt

COLOR_PALETTE = ["#AAC1CD", "#C1B195", "#7F9E96", "#947D6A", "#7D6C7A"]


def get_statistics(result):
    mean = np.mean(result, axis=1)
    low = np.percentile(result, 2.5, axis=1)
    high = np.percentile(result, 97.5, axis=1)
    return mean, low, high


def plot_with_statistics(x, mean, low, high, color, legend, linestyle="-", alpha=0.3):
    plt.fill_between(x, low, high, edgecolor=color, facecolor=color, alpha=alpha)
    plt.plot(x, mean, color=color, linestyle=linestyle, label=legend)
