def shift_array_left(array):
    for i in range(1, len(array)):
        array[i - 1] = array[i]
    return array