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

#ifndef HALFLOOP_BITSLICE_H_
#define HALFLOOP_BITSLICE_H_

#include "halfloop-common.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * Recovers the original 128-bit key using 128 bits of key material from
 * rk5-rk10.
 * @param rk56 the least significant byte of round key 5 in the most significant
 * byte, and round key 6 in the remaining bytes.
 * @param rk7 round key 7.
 * @param rk8 round key 8.
 * @param rk9 round key 9.
 * @param rk10 round key 10.
 * @return the recovered original key.
 */
hlkey halfloop_bitslice_revert_key(
    u32 rk56,
    u32 rk7,
    u32 rk8,
    u32 rk9,
    u32 rk10);

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
 * @param found return pointer to a list of matching keys.
 * @param num_found returns the number of keys in found.
 * @return halfloop_result_t HALFLOOP_SUCCESS on success.
 */
halfloop_result_t halfloop_bitslice(
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
 * Runs a benchmark on the bitslice implementation.
 */
halfloop_result_t halfloop_benchmark_bitslice(void);

/**
 * @brief Tests the HALFLOOP-24 bitslice implementation.
 *
 * @return halfloop_result_t HALFLOOP_SUCCESS on success.
 */
halfloop_result_t test_halfloop_bitslice(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HALFLOOP_BITSLICE_H_ */
