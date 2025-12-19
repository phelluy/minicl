// OpenCL kernels for sorting integer list
// global parameters for the radix sort kernels
// they are replaced by string manipulation before compilation
///////////////////////////////////////////////////////
// number of items in a group
#define _ITEMS <ITEMS>
// the number of virtual processors is _ITEMS * _GROUPS
#define _GROUPS <GROUPS>
// number of splits of the histogram
#define _HISTOSPLIT <HISTOSPLIT>
// number of bits for the integer in the list (max=32)
#define _TOTALBITS <TOTALBITS>
// number of bits in the radix
#define _BITS <BITS>
// maximal size of the list
// it has to be divisible by _ITEMS * _GROUPS
// (for other sizes, pad the list with big values)
#define _N <N>
// max local memory for scan kernel
#define _MAX_LOC_SCAN <MAX_LOC_SCAN>
#define VERBOSE 1
// transpose the initial vector (faster memory access)
// #define TRANSPOSE
// store the final permutation
// #define PERMUT
////////////////////////////////////////////////////////

// the following parameters are computed from the previous
#define _RADIX (1 << _BITS)        //  radix  = 2^_BITS
#define _PASS (_TOTALBITS / _BITS) // number of needed passes to sort the list
#define _HISTOSIZE (_ITEMS * _GROUPS * _RADIX) // size of the histogram
// maximal value of integers for the sort to be correct
#define _MAXINT (1 << (_TOTALBITS - 1))

// OpenCL kernel sources
// <ITEMS>, <GROUPS>, etc. are replaced by actual values before compilation

// compute the histogram for each radix and each virtual processor for the pass
// compute the histogram for each radix and each virtual processor for the pass
__kernel void histogram(const __global int *d_Keys, __global int *d_Histograms,
                        const int pass, const int n) {

  __local int loc_histo[_RADIX * _ITEMS];

  int it = get_local_id(0);  // i local number of the processor
  int ig = get_global_id(0); // global number = i + g I

  int gr = get_group_id(0); // g group number

  int groups = get_num_groups(0);
  int items = get_local_size(0);

  // set the local histograms to zero
  for (int ir = 0; ir < _RADIX; ir++) {
    loc_histo[ir * items + it] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // range of keys that are analyzed by the work item
  int size = n / groups / items; // size of the sub-list
  int start = ig * size;         // beginning of the sub-list

  int key, shortkey, k;

  // compute the index
  // the computation depends on the transposition
  for (int j = 0; j < size; j++) {
#ifdef TRANSPOSE
    k = groups * items * j + ig;
#else
    k = j + start;
#endif

    key = d_Keys[k];

    // extract the group of _BITS bits of the pass
    // the result is in the range 0.._RADIX-1
    shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1));

    // increment the local histogram
    loc_histo[shortkey * items + it]++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // copy the local histogram to the global one
  for (int ir = 0; ir < _RADIX; ir++) {
    d_Histograms[items * (ir * groups + gr) + it] = loc_histo[ir * items + it];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
}

// initial transpose of the list for improving
// coalescent memory access
__kernel void transpose(const __global int *invect, __global int *outvect,
                        const int nbcol, const int nbrow,
                        const __global int *inperm, __global int *outperm,
                        __local int *blockmat, __local int *blockperm,
                        const int tilesize) {

  // Flattened 1D launch emulation
  // Global Size = (nbrow / tilesize) * nbcol
  // We map global ID to (block_row, col)
  // i0 corresponds to `get_global_id(0)` in 2D which was `block_row * tilesize`
  // j corresponds to `get_global_id(1)` in 2D which was `col`

  // Here global_id(0) covers the whole 2D range
  int gid = get_global_id(0);

  // nbcol is the width of the matrix
  // j is the column index
  // But wait, the launch geometry in 2D was:
  // dim0: nbrow / tilesize
  // dim1: nbcol
  // So gid = (i_block_idx) * nbcol + j
  // i_block_idx = gid / nbcol
  // j = gid % nbcol

  int i_block_idx = gid / nbcol;
  int j = gid % nbcol;

  int i0 = i_block_idx * tilesize; // first row index

  // jloc was get_local_id(1). In 2D launch, local size was likely (1,
  // tilesize). So local_id(1) varied 0..tilesize-1 along the column dimension.
  // In 1D launch, if local_size is `tilesize`, then local_id(0) is
  // 0..tilesize-1. BUT `j` index advances by 1 for each thread. `gid` advances
  // by 1. `j` advances by 1 (modulo nbcol). If `nbcol` is multiple of
  // `tilesize`, this works perfectly. jloc = j % tilesize? Or simplify
  // get_local_id(0)? If we launch with local_size = tilesize, and `nbcol` is
  // multiple of `tilesize`: work-group handles a specific strip of columns for
  // a specific row-block. Yes.

  int jloc = get_local_id(0); // local column index

  // fill the cache
  for (int iloc = 0; iloc < tilesize; iloc++) {
    int k = (i0 + iloc) * nbcol + j; // position in the matrix
    blockmat[iloc * tilesize + jloc] = invect[k];
#ifdef PERMUT
    blockperm[iloc * tilesize + jloc] = inperm[k];
#endif
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // first row index in the transpose (corresponding to Input Tile Column)
  int j0 = j - jloc;

  // put the cache at the good place
  for (int iloc = 0; iloc < tilesize; iloc++) {
    int kt = (j0 + iloc) * nbrow + i0 + jloc; // position in the transpose
    outvect[kt] = blockmat[jloc * tilesize + iloc];
#ifdef PERMUT
    outperm[kt] = blockperm[jloc * tilesize + iloc];
#endif
  }
}

// each virtual processor reorders its data using the scanned histogram
// each virtual processor reorders its data using the scanned histogram
__kernel void reorder(const __global int *d_inKeys, __global int *d_outKeys,
                      __global int *d_Histograms, const int pass,
                      __global int *d_inPermut, __global int *d_outPermut,
                      const int n) {

  __local int loc_histo[_RADIX * _ITEMS];

  int it = get_local_id(0);
  int ig = get_global_id(0);

  int gr = get_group_id(0);
  int groups = get_num_groups(0);
  int items = get_local_size(0);

  int start = ig * (n / groups / items);
  int size = n / groups / items;

  // take the histogram in the cache
  for (int ir = 0; ir < _RADIX; ir++) {
    loc_histo[ir * items + it] = d_Histograms[items * (ir * groups + gr) + it];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int newpos, key, shortkey, k, newpost;

  for (int j = 0; j < size; j++) {
#ifdef TRANSPOSE
    k = groups * items * j + ig;
#else
    k = j + start;
#endif
    key = d_inKeys[k];
    shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1));

    newpos = loc_histo[shortkey * items + it];

#ifdef TRANSPOSE
    int ignew, jnew;
    ignew = newpos / (n / groups / items);
    jnew = newpos % (n / groups / items);
    newpost = jnew * (groups * items) + ignew;
#else
    newpost = newpos;
#endif

    d_outKeys[newpost] = key; // killing line !!!

#ifdef PERMUT
    d_outPermut[newpost] = d_inPermut[k];
#endif

    newpos++;
    loc_histo[shortkey * items + it] = newpos;
  }
}

// perform a parallel prefix sum (a scan) on the local histograms
// (see Blelloch 1990) each workitem worries about two memories
// see also http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// perform a parallel prefix sum (a scan) on the local histograms
// (see Blelloch 1990) each workitem worries about two memories
// see also http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__kernel void scanhistograms(__global int *histo, __global int *globsum) {

  __local int temp[_MAX_LOC_SCAN];

  int it = get_local_id(0);
  int ig = get_global_id(0);
  int decale = 1;
  int n = get_local_size(0) * 2;
  int gr = get_group_id(0);

  // load input into local memory
  // up sweep phase
  temp[2 * it] = histo[2 * ig];
  temp[2 * it + 1] = histo[2 * ig + 1];

  // parallel prefix sum (algorithm of Blelloch 1990)
  for (int d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (it < d) {
      int ai = decale * (2 * it + 1) - 1;
      int bi = decale * (2 * it + 2) - 1;
      temp[bi] += temp[ai];
    }
    decale *= 2;
  }

  // store the last element in the global sum vector
  // (maybe used in the next step for constructing the global scan)
  // clear the last element
  if (it == 0) {
    globsum[gr] = temp[n - 1];
    temp[n - 1] = 0;
  }

  // down sweep phase
  for (int d = 1; d < n; d *= 2) {
    decale >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (it < d) {
      int ai = decale * (2 * it + 1) - 1;
      int bi = decale * (2 * it + 2) - 1;

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // write results to device memory

  histo[2 * ig] = temp[2 * it];
  histo[2 * ig + 1] = temp[2 * it + 1];

  barrier(CLK_GLOBAL_MEM_FENCE);
}

// use the global sum for updating the local histograms
// each work item updates two values
__kernel void pastehistograms(__global int *histo, __global int *globsum) {

  int ig = get_global_id(0);
  int gr = get_group_id(0);

  int s;

  s = globsum[gr];

  // write results to device memory
  histo[2 * ig] += s;
  histo[2 * ig + 1] += s;

  barrier(CLK_GLOBAL_MEM_FENCE);
}
