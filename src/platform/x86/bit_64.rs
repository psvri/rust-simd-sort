/*
 * Constants used in sorting 8 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ZMM                  7, 6, 5, 4, 3, 2, 1, 0
//define NETWORK_64BIT_1 4, 5, 6, 7, 0, 1, 2, 3
//define NETWORK_64BIT_2 0, 1, 2, 3, 4, 5, 6, 7
//define NETWORK_64BIT_3 5, 4, 7, 6, 1, 0, 3, 2
//define NETWORK_64BIT_4 3, 2, 1, 0, 7, 6, 5, 4

/*impl SimdSortType for i64 {
    
}*/