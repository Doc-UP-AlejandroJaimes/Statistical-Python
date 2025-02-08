import numpy as np
from scipy.stats import mode

class FrequencyTable:
    def __init__(self, data: np.array):
        self.data = data
        self.min_value = np.min(data)
        self.max_value = np.max(data)
        self.range = self.max_value - self.min_value
        self.k = self.sturgers_distribution(len(data))
        self.amplitude = np.ceil(self.range / self.k)
        self.intervals = self.get_intervals()
        self.abs_frequency, self.edges = self.calculate_absolute_frequency()
        self.mindpoints = self.calculate_midpoints()
        self.xf = self.mindpoints * self.abs_frequency
        self.rel_frequency = self.abs_frequency / len(data)
        self.cum_frequency = np.cumsum(self.abs_frequency)
        self.mean = np.mean(data)
        self.median = np.median(data)
        self.mode = mode(data)[0]
        self.variance = np.var(data)
        self.std_dev = np.sqrt(self.variance)

    def sturgers_distribution(self, total_data: int) -> int:
        k = 1 + np.log2(total_data)
        return np.ceil(k) if int(k) % 2 == 0 else np.floor(k)

    def get_intervals(self) -> np.array:
        intervals = np.arange(self.min_value, self.max_value, self.amplitude)
        return np.append(intervals, self.max_value)

    def calculate_absolute_frequency(self):
        return np.histogram(self.data, bins=self.intervals)

    def calculate_midpoints(self) -> np.array:
        return np.array([(self.edges[i] + self.edges[i + 1]) / 2 for i in range(len(self.edges) - 1)])

    def print_table(self):
        print(f"{'Interval':<15}{'Midpoint':<10}{'f':<10}{'xf':<10}{'Rel Freq':<10}{'Cum Freq':<10}")
        print("=" * 65)
        for i in range(len(self.edges) - 1):
            print(f"[{self.edges[i]:.2f}, {self.edges[i+1]:.2f})".ljust(15) +
                  f"{self.mindpoints[i]:<10.2f}{self.abs_frequency[i]:<10}" +
                  f"{self.xf[i]:<10.2f}{self.rel_frequency[i]:<10.3f}{self.cum_frequency[i]:<10}")

    def summary(self):
        print(f"\n **Summary Statistics**")
        print(f"Mean: {self.mean:.2f}, Median: {self.median:.2f}, Mode: {self.mode}")
        print(f"Variance: {self.variance:.2f}, Std Dev: {self.std_dev:.2f}")
        print(f"Total Intervals: {int(self.k)}, Amplitude: {self.amplitude:.2f}")

# 📌 Example Usage:
data = np.random.randint(50, 100, 50)  # Random dataset
table = FrequencyTable(data)
table.print_table()
table.summary()
