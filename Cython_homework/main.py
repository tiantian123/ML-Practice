# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
import tm


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean','count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result

def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict, count_dict = dict(), dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]]-1)
    return result

def main():
    x_label, y_label = 10, 2
    x = np.random.randint(x_label, size=(10000, 1))
    y = np.random.randint(y_label, size=(10000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    
    t1_start = time.time()
    result_1 = target_mean_v1(data, 'y', 'x')
    t1_end = time.time()
    v1_time = t1_end - t1_start
    print("target_mean_v1 using time: %.6f s" % v1_time)

    t2_start = time.time()
    result_2 = target_mean_v2(data, 'y', 'x')
    t2_end = time.time()
    print(f"target_mean_v2 using time: {round(t2_end - t2_start, 6)} s, compared with v1, speed up: {round(v1_time/(t2_end-t2_start), 6)}")
    diff = np.linalg.norm(result_2 - result_1)
    print("diff v2 with v1: ", diff)

    t3_start = time.time()
    result_v3 = tm.target_mean_v3(data, 'y', 'x')
    t3_end = time.time()
    print(f"target_mean_v3 using time: {round(t3_end - t3_start, 6)} s, compared with v1, speed up: {round(v1_time/(t3_end-t3_start), 6)}")
    diff = np.linalg.norm(result_v3 - result_1)
    print(f"diff v3 with v1: {diff}\n")

    print("Homework:")
    for i in range(4, 8):
        start = time.time()
        result = tm.target_mean(data, 'y', 'x', x_label, i)
        end = time.time()
        print(f"target_mean_v{i} using time: {round(end - start, 6)} s, compared with v1, speed up: {round(v1_time/(end-start), 6)}")
        diff = np.linalg.norm(result - result_1)
        print(f"diff v{i} with v1: ", diff)


if __name__ == '__main__':
    main()