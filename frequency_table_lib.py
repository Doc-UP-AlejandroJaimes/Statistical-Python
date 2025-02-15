"""
StatsDescriptive: A Python class for performing descriptive statistical analysis.

Author: Alejandro Jaimes (https://github.com/Doc-UP-AlejandroJaimes)
Date Created: 14-02-2025

Features:
- Computes measures of central tendency: mean, median, and mode.
- Computes measures of dispersion: variance, standard deviation, and coefficient of variation.
- Computes shape measures: skewness and kurtosis.
- Constructs a frequency table with intervals, midpoints, frequencies, and cumulative frequencies.
- Uses Sturges' rule to determine the number of classes for grouped data.

Usage Example:
    import numpy as np
    from stats_descriptive import StatsDescriptive

    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    stats = StatsDescriptive(data)
    print(stats.mean)
    print(stats.std_dev)
"""

import numpy as np
from typing import Tuple
from scipy.stats import skew, kurtosis

class StatsDescriptive:
    """
    A class for computing descriptive statistics and frequency tables for a given dataset.
    """
    
    def __init__(self, data: np.array):
        """
        Initializes the StatsDescriptive object and computes various statistical measures.
        
        Parameters:
        - data (np.array): A NumPy array containing the dataset.
        """
        # Store dataset
        self.data = data
        
        # Measures of Central Tendency
        self.mean = np.mean(data)
        self.median = np.median(data)
        self.modes, self.mode_frqcy = self.mode()
        
        # Measures of Dispersion
        self.variance = np.var(data)
        self.std_dev = np.sqrt(self.variance)
        self.coef = (self.std_dev / self.mean) * 100  # Coefficient of Variation
        
        # Measures of Shape
        self.skewness = skew(data)
        self.kurtosis = kurtosis(data)
        
        # Frequency Table for Grouped Data
        self.min_value = np.min(data)
        self.max_value = np.max(data)
        self.range = self.max_value - self.min_value
        self.k = self.sturges_distribution()
        self.amplitude = np.ceil(self.range / self.k)
        self.intervals = self.get_intervals_numpy()
        self.abs_frequency, self.edges = self.abs_frqcy()
        self.midpoints = self.mindpoint(self.edges)
        self.midpoint_fprod = self.midpoints * self.abs_frequency
        self.variance_sq = (self.midpoints - self.mean) ** 2
        self.cum_variance_sq = self.variance_sq * self.abs_frequency
        self.rel_frequency = self.abs_frequency / self.data.size
        
        # Print the frequency table
        self.print_frqcy_table(self.edges, self.midpoints, self.midpoint_fprod,
                               self.abs_frequency, self.variance_sq, self.cum_variance_sq,
                               self.rel_frequency, np.cumsum(self.abs_frequency))
    
    def sturges_distribution(self) -> int:
        """Determines the number of intervals using Sturges' Rule."""
        k = 1 + np.log2(self.data.size)
        return np.ceil(k) if int(k) % 2 == 0 else np.floor(k)
    
    def mode(self) -> Tuple[np.array, np.array]:
        values, counts = np.unique(self.data, return_counts=True)
        return values[counts == np.max(counts)], counts[counts == np.max(counts)]
    
    def get_intervals_numpy(self) -> np.array:
        """Generates interval edges for frequency distribution."""
        intervals = np.arange(self.min_value, self.max_value, self.amplitude)
        return np.append(intervals, self.max_value)
    
    def abs_frqcy(self) -> Tuple[np.array, np.array]:
        """Calculates absolute frequency distribution."""
        ocurrencies, edges = np.histogram(self.data, bins=self.intervals)
        return ocurrencies, edges
    
    def mindpoint(self, edges) -> np.array:
        """Computes class midpoints for the frequency table."""
        return np.array([np.mean([edges[i], edges[i + 1]]) for i in range(edges.size - 1)])
    
    def summary_MCT(self):
        """Measures of central tendency."""
        print(f"Mean: {self.mean:.2f}, Median: {self.median:.2f}, Mode: {self.modes}")
    
    def summary_MD(self):
        """Measures of Dispersion."""
        print(f"Range: {self.range:.2f}, Variance: {self.variance:.2f}\nStandard Derivation: {self.std_dev}, Coefficient of Variation: {self.coef:.2f}")
        
    def summary_MS(self):
        """Measures of Shape."""
        print(f"Skewness: {self.skewness:.2f}, Kurtosis: {self.kurtosis:.2f}")
    
    def print_frqcy_table(self, edges, midpoints, xf, 
                          abs_frequency, var_sq, cum_var_sq,
                          rel_frequency, cum_frequency):
        """Prints the frequency table with various statistical measures."""
        print("{:<30} {:<18} {:<13} {:<13} {:<13} {:<18} {:<14} {:<10}".format(
            "INTERVAL", "x", "f", "xf", "(x-X̄)^2", "(x-X̄)^2 * f", "fr", "F"))
        print("-" * 130)
        
        for i in range(edges.size - 1):
            bins = f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
            print("{:<30} {:<18.2f} {:<13} {:<13.2f} {:<13.2f} {:<15.2f} {:<15.3f} {:<10}".format(
                bins, midpoints[i], abs_frequency[i], xf[i],
                var_sq[i], cum_var_sq[i], rel_frequency[i], cum_frequency[i]
            ))
    
