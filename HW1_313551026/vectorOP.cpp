#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  int iter = N/VECTOR_WIDTH;
  __pp_mask mask_all = _pp_init_ones(VECTOR_WIDTH);
  __pp_mask mask_all_not = _pp_init_ones(-1);
  __pp_mask mask_enlane;
  __pp_vec_int vec_ones = _pp_vset_int(1.0f);
  __pp_vec_int vec_zeros = _pp_vset_int(0.0f);
  __pp_vec_float vec_clamp = _pp_vset_float(9.999999f);
  __pp_vec_float base, vec_result;
  __pp_vec_int exp;

  for(int i = 0; i < iter; i++){
    _pp_vset_float(vec_result, 1.0f, mask_all);
    _pp_vload_float(base, values, mask_all);
    _pp_vload_int(exp, exponents, mask_all);
    while(1){
      _pp_vgt_int(mask_enlane, exp, vec_zeros, mask_all);
      if(_pp_cntbits(mask_enlane) == 0){
        break;
      }
      _pp_vmult_float(vec_result, vec_result, base, mask_enlane);
      _pp_vsub_int(exp, exp, vec_ones, mask_enlane);
    }
    _pp_vgt_float(mask_enlane, vec_result, vec_clamp, mask_all);
    _pp_vmove_float(vec_result, vec_clamp, mask_enlane);
    _pp_vstore_float(output, vec_result, mask_all);
    values += VECTOR_WIDTH;
    exponents += VECTOR_WIDTH;
    output += VECTOR_WIDTH;
  }

  if(N%VECTOR_WIDTH > 0){
    mask_enlane = mask_all_not;
    mask_all = _pp_init_ones(N%VECTOR_WIDTH);
    _pp_vset_float(vec_result, 1.0, mask_all);
    _pp_vload_float(base, values, mask_all);
    _pp_vload_int(exp, exponents, mask_all);
    while(1){
      _pp_vgt_int(mask_enlane, exp, vec_zeros, mask_all);
      if(_pp_cntbits(mask_enlane) == 0){
        break;
      }
      _pp_vmult_float(vec_result, vec_result, base, mask_enlane);
      _pp_vsub_int(exp, exp, vec_ones, mask_enlane);
    }
    _pp_vgt_float(mask_enlane, vec_result, vec_clamp, mask_all);
    _pp_vmove_float(vec_result, vec_clamp, mask_enlane);
    _pp_vstore_float(output, vec_result, mask_all);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float ans = 0.0f;
  __pp_vec_float vec_result;
  __pp_mask mask_all = _pp_init_ones(VECTOR_WIDTH);

  for(int i = 0; i < N/VECTOR_WIDTH; i++){
    _pp_vload_float(vec_result, values, mask_all);
    int remain_length = VECTOR_WIDTH;
    while(remain_length > 1){
      _pp_interleave_float(vec_result, vec_result);
      _pp_hadd_float(vec_result, vec_result);
      remain_length /= 2;
    }
    ans += vec_result.value[0];
    values += VECTOR_WIDTH;
  }

  return ans;
}