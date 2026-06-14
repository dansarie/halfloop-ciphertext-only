/* Implementation of attacks on HALFLOOP-24.

   Copyright (C) 2022 Marcus Dansarie, Patrick Derbez, Gregor Leander, and
   Lukas Stennes.
   Copyright (C) 2025-2026 Marcus Dansarie, Gregor Leander, Shahram Rasoolzadeh,
   Lukas Stennes, and Cihangir Tezcan.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>. */

#ifndef HALFLOOP_CUDA_H_
#define HALFLOOP_CUDA_H_

#ifdef CUDA_ENABLED

#ifdef __cplusplus
#include <cuda.h>
#endif

#include "halfloop-common.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef __cplusplus
#define RETURN_ON_CUDA_ERROR(R) {\
  cudaError_t e = R;\
  if (e != cudaSuccess) {\
    fprintf(\
        stderr,\
        "CUDA error \"%s\" on line %d in %s.\n",\
        cudaGetErrorName(e),\
        __LINE__,\
        __FILE__);\
    err = HALFLOOP_INTERNAL_ERROR;\
    goto error;\
  }\
}

#define RETURN_ON_CUDA_RESULT(R) {\
  CUresult e = R;\
  if (e != CUDA_SUCCESS) {\
    const char *errstr = NULL;\
    cuGetErrorName(e, &errstr);\
    fprintf( \
        stderr,\
        "CUDA error \"%s\" on line %d in %s.\n",\
        errstr,\
        __LINE__,\
        __FILE__);\
    err = HALFLOOP_INTERNAL_ERROR;\
    goto error;\
  }\
}
#endif /* __cplusplus */

/**
 * Initializes the CUDA API.
 */
halfloop_result_t halfloop_init_cuda(void);

/**
 * @brief Tests the HALFLOOP-24 CUDA bitslice implementation.
 *
 * @return halfloop_result_t HALFLOOP_SUCCESS on success.
 */
halfloop_result_t test_halfloop_cuda_bitslice(void);

/**
 * Returns a list of available CUDA devices.
 *
 * @param num_devices return variable for the number of CUDA devices.
 * @param device_name return variable for the device names. Device name n starts
 * at index n * 256 in the returned array. The array must be freed by the
 * caller.
 */
halfloop_result_t halfloop_list_cuda_devices(
    int *num_devices,
    char **device_names);

/**
 * @brief Searches through 2^32 candidate keys for ones that cause the two
 * ciphertexts to be decrypted to the same value. The input ciphertexts come
 * from a pair of identical plaintexts that have been encrypted with tweaks that
 * only differ in the least significant bit in the tweak's word number field.
 *
 * @param ct0 the first 24-bit ciphertext.
 * @param ct1 the second 24-bit ciphertext.
 * @param tw0 the tweak for the first ciphertext.
 * @param rk7 round key 7, normalized.
 * @param rk8 round key 8, normalized.
 * @param rk9 round key 9, normalized.
 * @param rk10 round key 10, normalized.
 * @param found return pointer to a list of found matches. Each match contains
 * rk 6 in the least significant 24 bits and the least significant byte of rk 5
 * in the most significant 8 bits.
 * @param num_found returns the number of keys in found.
 * @return halfloop_result_t HALFLOOP_SUCCESS on success.
 */
halfloop_result_t halfloop_cuda_bitslice(
    u32 ct0,
    u32 ct1,
    u64 tw0,
    u32 rk7,
    u32 rk8,
    u32 rk9,
    u32 rk10,
    hlkey **found,
    int *num_found);

/**
 * Recovers the unknown 48 bits of the key through a brute force attack.
 * @param ct0 a ciphertext from a 2G ALE "TO" word.
 * @param ct1 a second ciphertext, which is highly likely to have the same
 * plaintext as ct0 and where the only tweak difference is that the tweak for
 * ct0 has word number 1 and the tweak for ct2 has word number 2.
 * @param tw0 the tweak for ct0.
 * @param rk7 the most significant byte of LL^-1(round key 7), normalized.
 * @param rk8 round key 8, normalized.
 * @param rk9 round key 9, normalized.
 * @param rk10 round key 10, normalized.
 * @param devices a list of CUDA device ids to use for the search, or NULL to
 * use all available devices.
 * @param num_devices number of device ids in the devices list.
 * @param verbose increases verbosity when set to true.
 * @param found return variable for the found key.
 */
halfloop_result_t halfloop_cuda_bitslice_all(
    const u32 ct0,
    const u32 ct1,
    const u64 tw0,
    const tuple_t *tuples,
    int num_tuples,
    u8 rk7,
    u32 rk8,
    u32 rk9,
    u32 rk10,
    int *devices,
    int num_devices,
    bool verbose,
    hlkey *found);

/**
 * Runs a ciphertext-only attack on HALFLOOP, recovering rk8, rk9, rk10, and one
 * byte from LL^-1(rk7).
 * @param algo Specifies which attack algorithm to use. Currently, only
 * GPU_ATTACK3 is supported.
 * @param ct a list of ciphertext-tweak tuples.
 * @param num_ct number of ciphertext-tweak tuples in ct.
 * @param pairs a list of pairs with the required tweak difference, where the
 * indexes correspond to indexes in the ct list.
 * @param num_pairs number of pairs in the pairs list.
 * @param tau1 threshold value in the first part of the algorithm.
 * @param tau2 threshold value in the second part of the algorithm.
 * @param blockmul number of CUDA blocks per multiprocessor.
 * @param fixed_bits User-set job bits. Filled from MSB down to LSB. The eight
 * most significant bits represent one byte of rk9. The remaining bits represent
 * rk10.
 * @param num_fixed Number of valid bits in fixed_bits.
 * @param candidates return variable for identified candidate keys. The caller
 * must return the array using free.
 * @param num_candidates return variable for the number of candidate keys in the
 * candidates list.
 * @param verbose if set to true, the search will output more information to
 * stdout.
 * @param profile if set to true, only a limited search will be performed. Used
 * for performance measurement and profiling.
 * @param devices a list of CUDA device ids to use. Set to NULL to use all
 * available devices.
 * @param num_devices the number of CUDA device ids in the devices list.
 * @param found return variable for found 80-bit candidate keys.
 * @param num_found return variable for number of keys in found array.
 */
halfloop_result_t cuda_ct_attack(
    halfloop_algorithm_t algo,
    const tuple_t *ct,
    int num_ct,
    const tuple_pair_t *pairs,
    int num_pairs,
    u32 tau1,
    u32 tau2,
    u32 blockmul,
    u32 fixed_bits,
    int num_fixed,
    candidate_key_t **candidates,
    u32 *num_candidates,
    bool verbose,
    bool profile,
    int *devices,
    int num_devices);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CUDA_ENABLED */

#endif /* HALFLOOP_CUDA_H_ */
