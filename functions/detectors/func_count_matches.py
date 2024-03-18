def func_count_matches(rrInt, rrIntDiff, windowLength, r):
    differences = 0
    for i in range(windowLength - 1):
        for j in range(i + 1, windowLength):
            if abs(rrIntDiff[i]) < 0.40 and abs(rrIntDiff[i]) > 0.008:
            # if abs(rrIntDiff[i]) < 0.80 and abs(rrIntDiff[i]) > 0.004:
                # if rrInt[i] != 0 and rrInt[j] != 0:
                if abs(rrInt[i] - rrInt[j]) > r:
                    differences += 1
                # end
            # else:
            #     differences = 0

    diffInd = differences / (windowLength * (windowLength - 1) / 2)
    # diffInd = (9/256) * differences
    return diffInd