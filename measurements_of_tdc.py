import numpy as np

def mean(arr:np.array) -> np.float64:
    return np.round(arr.sum() / arr.size,3)

def median(arr:np.array) -> np.float64:
    # 1. sort array
    arr.sort()
    # 2. length array
    length = arr.size
    isOdd = length % 2 != 0
    # 3. return med
    mid_len = int(length / 2)
    med = np.mean([arr[mid_len - 1],arr[mid_len]]) if not isOdd else arr[length // 2]
    return med

def mode(arr:np.array) -> np.array:
    values, counts = np.unique(arr, return_counts=True)
    return values[counts == np.max(counts)]

if __name__ == '__main__':
    arr = np.array([42, 47, 53, 47, 50, 45, 46, 48, 41, 49, 45, 40, 50])
    st_mean   = mean(arr)
    st_median = median(arr)
    st_mode   = mode(arr)
    print(f'Mean: {st_mean}\nMedian: {st_median}\nModes: {st_mode}')