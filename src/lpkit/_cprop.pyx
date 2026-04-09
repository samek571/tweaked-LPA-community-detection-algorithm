import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, int64_t, uint64_t

#tie_break: 0 = first, 1 = min, 2 = max
#returns dict[u -> new_label]
def lpa_block(
        edges, #numpy array (m,2) uint32
        labels_snapshot, #numpy array (n,) int64
        int tie_break,
        unsigned long long seed):

    #conversion of Python objects to typed memoryviews
    cdef np.ndarray[np.uint32_t, ndim=2] edges_arr = edges
    cdef np.ndarray[np.int64_t, ndim=1] labels_arr = labels_snapshot

    cdef uint32_t[:, :] e = edges_arr  # typed memoryview
    cdef int64_t[:] lab = labels_arr   # typed memoryview

    cdef Py_ssize_t m = e.shape[0]
    cdef uint32_t u, v, cur_u = <uint32_t>-1

    cdef int have_u = 0
    cdef int num_labels = 0
    cdef int MAX_DEG = 4096
    cdef int64_t label_ids[4096]
    cdef int counts[4096]

    cdef dict result = {}
    cdef Py_ssize_t i, j
    cdef int found
    cdef int best_idx
    cdef int best_count
    cdef int64_t new_lab, old_lab

    #local helper
    def finalize():
        nonlocal have_u, num_labels, cur_u
        cdef Py_ssize_t j
        cdef int idx = 0
        cdef int cnt = counts[0]

        if not have_u or num_labels == 0:
            return

        for j in range(1, num_labels):
            if counts[j] > cnt:
                cnt = counts[j]
                idx = j
            elif counts[j] == cnt:
                if tie_break == 1 and label_ids[j] < label_ids[idx]:
                    idx = j
                elif tie_break == 2 and label_ids[j] > label_ids[idx]:
                    idx = j

        new_lab = label_ids[idx]
        old_lab = lab[cur_u]

        if new_lab != old_lab:
            result[<int>cur_u] = <int>new_lab

    #main loop
    for i in range(m):
        u = e[i, 0]
        v = e[i, 1]

        if not have_u or u != cur_u:
            finalize()
            cur_u = u
            have_u = 1
            num_labels = 0

        found, newlab = 0, lab[v]
        for j in range(num_labels):
            if label_ids[j] == newlab:
                counts[j] += 1
                found = 1
                break

        if not found:
            if num_labels < MAX_DEG:
                label_ids[num_labels] = newlab
                counts[num_labels] = 1
                num_labels += 1

    finalize()
    return result

def accumulate_source_run(
        edges,           # numpy array (m,2) uint32
        labels_live,     # numpy array / memmap (n,) uint64
        unsigned int target_u,
        Py_ssize_t start_idx,
        dict counts):
    """
    Scan the contiguous slice of rows with source == target_u starting at start_idx,
    increment counts[label[v]] in-place, and return the first index after the run.

    This is used by the oracle-permuted block-based streaming path.
    """
    cdef np.ndarray[np.uint32_t, ndim=2] edges_arr = edges
    cdef np.ndarray[np.uint64_t, ndim=1] labels_arr = labels_live

    cdef const uint32_t[:, :] e = edges_arr
    cdef const uint64_t[:] lab = labels_arr

    cdef Py_ssize_t m = e.shape[0]
    cdef Py_ssize_t i = start_idx
    cdef uint64_t lv

    while i < m and e[i, 0] == target_u:
        lv = lab[e[i, 1]]
        counts[lv] = counts.get(lv, 0) + 1
        i += 1

    return i