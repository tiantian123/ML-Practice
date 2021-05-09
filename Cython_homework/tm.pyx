# distutils: language=c++
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log
from libcpp.unordered_map cimport unordered_map


cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result


cpdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    
    i = 0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]] - 1)


cpdef target_mean(data, y_name, x_name, x_label, version):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name], dtype=np.int64)

    if version == 4:
        target_mean_v4_impl(result, y, x, nrow)
    elif version == 5:
        target_mean_v5_impl(result, y, x, nrow)
    elif version == 6:
        target_mean_v6_impl(result, y, x, nrow)
    elif version == 7:
        target_mean_v7_impl(result, y, x, nrow, x_label)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void target_mean_v4_impl(double[:] result, double[:] y, long[:] x, const long nrow):
    # v4:change the x data type
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    
    i = 0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]] - 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void target_mean_v5_impl(double[:] result, double[:] y, long[:] x, const long nrow):
    # v5: using unordered_map replace the dict.
    cdef unordered_map[int, float] value_map
    cdef unordered_map[int, int] count_map

    cdef long i
    for i in range(nrow):
        if not value_map.count(x[i]):
            value_map[x[i]] = y[i]
            count_map[x[i]] = 1
        else:
            value_map[x[i]] += y[i]
            count_map[x[i]] += 1 
    i = 0
    for i in range(nrow):
        result[i] = (value_map[x[i]] - y[i]) / (count_map[x[i]] - 1)


cdef struct value_count:
    float value
    int count

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void target_mean_v6_impl(double[:] result, double[:] y, long[:] x, const long nrow):
    # v6: using one unordered_map replace the dict.
    cdef unordered_map[int, value_count] value_map

    cdef long i
    for i in range(nrow):
        if not value_map.count(x[i]):
            value_map[x[i]].value = y[i]
            value_map[x[i]].count = 1
        else:
            value_map[x[i]].value += y[i]
            value_map[x[i]].count += 1
    i = 0
    for i in range(nrow):
        result[i] = (value_map[x[i]].value - y[i]) / (value_map[x[i]].count - 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void target_mean_v7_impl(double[:] result, double[:] y, long[:] x, const long nrow, const long x_label):
    # v7: using array replace the dict
    cdef np.ndarray[double] value = np.asfortranarray(np.zeros(x_label), dtype=np.float64)
    cdef np.ndarray[long] count = np.asfortranarray(np.zeros(x_label), dtype=np.int64)

    cdef long i
    for i in range(nrow):
            value[x[i]] += y[i]
            count[x[i]] += 1
    i = 0
    for i in range(nrow):
        result[i] = (value[x[i]] - y[i]) / (count[x[i]] - 1)