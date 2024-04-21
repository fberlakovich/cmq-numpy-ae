#ifndef CMLQ_H
#define CMLQ_H

#define CMLQ_CACHE_SIZE 512
#define SUBSCRIPT_CACHE_SIZE 512
#include <mapping.h>

typedef struct _CMLQIterCache {
    /* Initial fixed position data */
    npy_uint32 itflags;
    npy_uint8 ndim, nop;
    npy_int8 maskop;
    npy_intp itersize, iterstart, iterend;
    /* iterindex is only used if RANGED or BUFFERED is set */
    npy_intp iterindex;
} CMLQIterCache;

enum CMLQCacheState {
    UNUSED = 0,
    TRIVIAL,
    ITERATOR,
    BROADCAST,
    DISABLED
};

typedef struct _CMLQIteratorPathCache {
    NpyIter *cached_iter;
    NpyIter_IterNextFunc *iter_next;
    npy_intp *countptr;
    char **dataptr;
    npy_intp *strides;
} CMLQIteratorPathCache;

typedef struct _CMLQTrivialPathCache {
    npy_intp fixed_strides[3];
    npy_intp count;
} CMLQTrivialPathCache;

typedef union _CMLQCacheData {
    CMLQTrivialPathCache trivial;
    CMLQIteratorPathCache iterator;
} CMLQCacheData;


typedef struct _CMLQCacheStatsElem {
    char *opname;
    _Py_CODEUNIT *instr_ptr;
    uint64_t op_exec_count;
    uint64_t trivial_cache_hits;
    uint64_t trivial_cache_misses;
    uint64_t iterator_cache_hits;
    uint64_t iterator_cache_misses;
    uint64_t result_cache_hits;
    uint64_t result_cache_misses;
    uint64_t refcnt_misses;
    uint64_t ndims_misses;
    uint64_t shape_misses;
    uint64_t temp_elision_hits;
    uint64_t trivial_case;
    uint64_t iterator_case;
    uint64_t left_type_misses;
    uint64_t right_type_misses;
    uint64_t exponent_type_misses;
    uint64_t ufunc_type_misses;
    uint64_t trivial_cache_collisions;
    uint64_t iterator_cache_collisions;
    uint64_t function_end_clear;
    uint64_t trivial_cache_init;
    uint64_t iterator_cache_init;
    enum CMLQCacheState last_state;
} CMLQCacheStatsElem;


typedef struct _CMLQLocalityCacheElem {
    enum CMLQCacheState state;
    PyArrayObject *result;
    int16_t miss_counter;
    union {
        CMLQTrivialPathCache trivial;
        CMLQIteratorPathCache iterator;
    };
#ifdef CMLQ_STATS
    CMLQCacheStatsElem stats;
#endif

} CMLQLocalityCacheElem;

#define RESULT_CACHE_WARMUP 6
#define IS_RESULT_CACHE_UNSTABLE(elem) elem->miss_counter >= 100

// a negative counter means we are still warming up the cache
#define RESET_CACHE_COUNTER(elem) elem->miss_counter = -RESULT_CACHE_WARMUP

typedef struct _CMLQSubscriptCacheElem {
    _Py_CODEUNIT *instr;
    npy_index_info *indices;
    int index_type;
    int index_num;
    int array_ndim;
} CMLQSubscriptCacheElem;

#endif  // CMLQ_H
