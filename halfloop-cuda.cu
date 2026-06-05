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

#include <locale.h>
#include <barrier>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "halfloop-common.h"
#include "halfloop-cuda.h"

/**
 * Broadcasts the Bth bit of a 32-bit unsigned variable V to all positions in a
 * 32-bit unsigned variable. The bit number B is 0 for the MSB and 31 for the
 * LSB.
 */
#define GET_U32_BIT(V, B) (0ULL - (((V) >> (31  - B)) & 1))

/**
 * Shorthand for calling create_cuda_device_table that casts the arguments D and
 * H to void pointers.
 */
#define CREATE_CUDA_TABLE(D, H, S) \
    create_cuda_device_table((void**)(D), (void*)(H), (S))

/** Size of key candidate output buffer. */
#define MAX_CANDIDATES (10000000)

/** Size of partial key buffer. */
#define MAX_KEYS (2724)

/** Size of match buffers. */
#define MAX_MATCHES (4096)

/* Number of 32-bit counter registers used by each block in algorithm 2. */
#define ALG2_COUNTERS (0x200000)

/**
 * Pointers to lookup tables for the CUDA implementation. The structure is
 * initialized by init_cuda_tables and freed by free_cuda_tables.
 */
typedef struct {
  u8* __restrict__ sbox;      /**< HALFLOOP/AES S-box. */
  u8 * __restrict__ inv_sbox; /**< HALFLOOP/AES inverse S-box. */
  u32 * __restrict__ y0;      /**< y0 to x2 difference LUT. */
  u32 * __restrict__ y1;      /**< y1 to x2 difference LUT. */
  u32 * __restrict__ y2;      /**< y2 to x2 difference LUT. */
  u8 * __restrict__ mul2;     /**< LUT for finite field multiplication by 2. */
  u8 * __restrict__ mul6;     /**< LUT for finite field multiplication by 6. */
  u8 * __restrict__ mul8;     /**< LUT for finite field multiplication by 8. */
  u8 * __restrict__ mul9;     /**< LUT for finite field multiplication by 9. */
  u8 * __restrict__ mul39;    /**< LUT for finite field multiplication by 39. */
  u16 * __restrict__ ddt;     /**< DDT output value LUT. */
} CudaTables;

/** Attach thread arguments. */
typedef struct {
  halfloop_algorithm_t algorithm; /**< Selected algorithm. */
  const tuple_t *ct;              /**< Ciphertext-tweak tuples. */
  const tuple_pair_t *pairs;      /**< Pair indexes. */
  std::mutex mutex;               /**< Worker synchronization mutex. */
  std::unique_ptr<std::barrier<>> barrier; /**< Worker synchronization
                                                barrier. */
  candidate_key_t *validated;     /**< Validated candidate keys. */
  u64 next_job;      /**< Value of non-fixed bits in next job. */
  u32 fixed_bits;    /**< User-set job bits. Filled from MSB down to LSB. */
  u32 threadnum;     /**< Counter for individual thread number assignment. */
  u32 tau1;          /**< Theshold value 1. */
  u32 tau2;          /**< Threshold value 2. */
  u32 blockmul;      /**< Number of blocks per multiprocessor. */
  u32 num_validated; /**< Number of candidate keys in validated array. */
  int devices[MAX_DEVICES]; /**< List of CUDA devices to use. */
  int num_devices;   /**< Number of devices in devices list. */
  int num_ct;        /**< Number of tuples in ct list. */
  int num_pairs;     /**< Number of pairs in pairs list. */
  int num_fixed;     /**< Number of valid bits in fixed_bits. */
  bool run;          /**< Signals workers to quit when set to false. */
  bool profile;      /**< Runs workers in profiling mode when set to true. */
  bool verbose;      /**< Verbosity flag. */
} ThreadArg;

/**
 * @brief Represents a byte in the bitslice implementation.
 */
typedef struct {
  u32 b0; /**< MSB */
  u32 b1;
  u32 b2;
  u32 b3;
  u32 b4;
  u32 b5;
  u32 b6;
  u32 b7; /**< LSB */
} Eightbits;

/**
 * @brief Represents three bytes in the bitslice implementation.
 */
typedef struct {
  Eightbits msb;
  Eightbits mid;
  Eightbits lsb;
} Twentyfourbits;

/**
 * Bitsliced XOR of an 8-bit state.
 */
__device__ Eightbits xor8(Eightbits a, Eightbits b) {
  Eightbits out = {
    .b0 = a.b0 ^ b.b0,
    .b1 = a.b1 ^ b.b1,
    .b2 = a.b2 ^ b.b2,
    .b3 = a.b3 ^ b.b3,
    .b4 = a.b4 ^ b.b4,
    .b5 = a.b5 ^ b.b5,
    .b6 = a.b6 ^ b.b6,
    .b7 = a.b7 ^ b.b7
  };
  return out;
}

/**
 * Bitsliced XOR of a 24-bit state.
 */
__device__ Twentyfourbits xor24(Twentyfourbits a, Twentyfourbits b) {
  Twentyfourbits out = {
    .msb = xor8(a.msb, b.msb),
    .mid = xor8(a.mid, b.mid),
    .lsb = xor8(a.lsb, b.lsb)
  };
  return out;
}

/**
 * Broadcasts an eight-bit value to all bitslice lanes.
 */
__device__ Eightbits get_eightbits(u8 bits) {
  Eightbits out = {
    .b0 = 0U - (bits >> 7),
    .b1 = 0U - ((bits >> 6) & 1),
    .b2 = 0U - ((bits >> 5) & 1),
    .b3 = 0U - ((bits >> 4) & 1),
    .b4 = 0U - ((bits >> 3) & 1),
    .b5 = 0U - ((bits >> 2) & 1),
    .b6 = 0U - ((bits >> 1) & 1),
    .b7 = 0U - (bits & 1)
  };
  return out;
}

/**
 * @brief Gate network implementation of the inverse Rijndael S-box.
 * @see https://cs-www.cs.yale.edu/homes/peralta/CircuitStuff/AESReverseDepth.txt
 */
__device__ Eightbits bitslice_inv_sub_bytes(Eightbits in) {
  u32 T23 = in.b0 ^ in.b3;
  u32 T22 = ~(in.b1 ^ in.b3);
  u32 T2  = ~(in.b0 ^ in.b1);
  u32 T1  = in.b3 ^ in.b4;
  u32 T24 = ~(in.b4 ^ in.b7);
  u32 R5  = in.b6 ^ in.b7;
  u32 T8  = ~(in.b1 ^ T23);
  u32 T19 = T22 ^ R5;
  u32 T9  = ~(in.b7 ^ T1);
  u32 T10 = T2 ^ T24;
  u32 T13 = T2 ^ R5;
  u32 T3  = T1 ^ R5;
  u32 T25 = ~(in.b2 ^ T1);
  u32 R13 = in.b1 ^ in.b6;
  u32 T17 = ~(in.b2 ^ T19);
  u32 T20 = T24 ^ R13;
  u32 T4  = in.b4 ^ T8;
  u32 R17 = ~(in.b2 ^ in.b5);
  u32 R18 = ~(in.b5 ^ in.b6);
  u32 R19 = ~(in.b2 ^ in.b4);
  u32 Y5  = in.b0 ^ R17;
  u32 T6  = T22 ^ R17;
  u32 T16 = R13 ^ R19;
  u32 T27 = T1 ^ R18;
  u32 T15 = T10 ^ T27;
  u32 T14 = T10 ^ R18;
  u32 T26 = T3 ^ T16;
  u32 M1  = T13 & T6;
  u32 M2  = T23 & T8;
  u32 M3  = T14 ^ M1;
  u32 M4  = T19 & Y5;
  u32 M5  = M4 ^ M1;
  u32 M6  = T3 & T16;
  u32 M7  = T22 & T9;
  u32 M8  = T26 ^ M6;
  u32 M9  = T20 & T17;
  u32 M10 = M9 ^ M6;
  u32 M11 = T1 & T15;
  u32 M12 = T4 & T27;
  u32 M13 = M12 ^ M11;
  u32 M14 = T2 & T10;
  u32 M15 = M14 ^ M11;
  u32 M16 = M3 ^ M2;
  u32 M17 = M5 ^ T24;
  u32 M18 = M8 ^ M7;
  u32 M19 = M10 ^ M15;
  u32 M20 = M16 ^ M13;
  u32 M21 = M17 ^ M15;
  u32 M22 = M18 ^ M13;
  u32 M23 = M19 ^ T25;
  u32 M24 = M22 ^ M23;
  u32 M25 = M22 & M20;
  u32 M26 = M21 ^ M25;
  u32 M27 = M20 ^ M21;
  u32 M28 = M23 ^ M25;
  u32 M29 = M28 & M27;
  u32 M30 = M26 & M24;
  u32 M31 = M20 & M23;
  u32 M32 = M27 & M31;
  u32 M33 = M27 ^ M25;
  u32 M34 = M21 & M22;
  u32 M35 = M24 & M34;
  u32 M36 = M24 ^ M25;
  u32 M37 = M21 ^ M29;
  u32 M38 = M32 ^ M33;
  u32 M39 = M23 ^ M30;
  u32 M40 = M35 ^ M36;
  u32 M41 = M38 ^ M40;
  u32 M42 = M37 ^ M39;
  u32 M43 = M37 ^ M38;
  u32 M44 = M39 ^ M40;
  u32 M45 = M42 ^ M41;
  u32 M46 = M44 & T6;
  u32 M47 = M40 & T8;
  u32 M48 = M39 & Y5;
  u32 M49 = M43 & T16;
  u32 M50 = M38 & T9;
  u32 M51 = M37 & T17;
  u32 M52 = M42 & T15;
  u32 M53 = M45 & T27;
  u32 M54 = M41 & T10;
  u32 M55 = M44 & T13;
  u32 M56 = M40 & T23;
  u32 M57 = M39 & T19;
  u32 M58 = M43 & T3;
  u32 M59 = M38 & T22;
  u32 M60 = M37 & T20;
  u32 M61 = M42 & T1;
  u32 M62 = M45 & T4;
  u32 M63 = M41 & T2;
  u32 P0  = M52 ^ M61;
  u32 P1  = M58 ^ M59;
  u32 P2  = M54 ^ M62;
  u32 P3  = M47 ^ M50;
  u32 P4  = M48 ^ M56;
  u32 P5  = M46 ^ M51;
  u32 P6  = M49 ^ M60;
  u32 P7  = P0 ^ P1;
  u32 P8  = M50 ^ M53;
  u32 P9  = M55 ^ M63;
  u32 P10 = M57 ^ P4;
  u32 P11 = P0 ^ P3;
  u32 P12 = M46 ^ M48;
  u32 P13 = M49 ^ M51;
  u32 P14 = M49 ^ M62;
  u32 P15 = M54 ^ M59;
  u32 P16 = M57 ^ M61;
  u32 P17 = M58 ^ P2;
  u32 P18 = M63 ^ P5;
  u32 P19 = P2 ^ P3;
  u32 P20 = P4 ^ P6;
  u32 P22 = P2 ^ P7;
  u32 P23 = P7 ^ P8;
  u32 P24 = P5 ^ P7;
  u32 P25 = P6 ^ P10;
  u32 P26 = P9 ^ P11;
  u32 P27 = P10 ^ P18;
  u32 P28 = P11 ^ P25;
  u32 P29 = P15 ^ P20;
  Eightbits out;
  out.b0  = P13 ^ P22;
  out.b1  = P26 ^ P29;
  out.b2  = P17 ^ P28;
  out.b3  = P12 ^ P22;
  out.b4  = P23 ^ P27;
  out.b5  = P19 ^ P24;
  out.b6  = P14 ^ P23;
  out.b7  = P9 ^ P16;
  return out;
}

/**
 * @brief Performs the rotate left operation six steps.
 *
 * @param in input value.
 * @return the input value rotated right six steps.
 */
__device__ Eightbits bitslice_inv_rotate_rows_6(Eightbits in) {
  Eightbits out = {
    .b0 = in.b2,
    .b1 = in.b3,
    .b2 = in.b4,
    .b3 = in.b5,
    .b4 = in.b6,
    .b5 = in.b7,
    .b6 = in.b0,
    .b7 = in.b1
  };
  return out;
}

/**
 * @brief Performs the rotate left operation four steps.
 *
 * @param in input value.
 * @return the input value rotated right four steps.
 */
__device__ Eightbits bitslice_inv_rotate_rows_4(Eightbits in) {
  Eightbits out = {
    .b0 = in.b4,
    .b1 = in.b5,
    .b2 = in.b6,
    .b3 = in.b7,
    .b4 = in.b0,
    .b5 = in.b1,
    .b6 = in.b2,
    .b7 = in.b3
  };
  return out;
}

/**
 * Performs bitsliced finite field multiplication by 6.
 */
__device__ Eightbits multiply6(Eightbits in) {
  Eightbits out = {
    .b0 = in.b1 ^ in.b2,
    .b1 = in.b2 ^ in.b3,
    .b2 = in.b0 ^ in.b3 ^ in.b4,
    .b3 = in.b1 ^ in.b4 ^ in.b5,
    .b4 = in.b0 ^ in.b1 ^ in.b5 ^ in.b6,
    .b5 = in.b0 ^ in.b6 ^ in.b7,
    .b6 = in.b1 ^ in.b7,
    .b7 = in.b0 ^ in.b1
  };
  return out;
}

/**
 * Performs bitsliced finite field multiplication by 8.
 */
__device__ Eightbits multiply8(Eightbits in) {
  Eightbits out = {
    .b0 = in.b3,
    .b1 = in.b0 ^ in.b4,
    .b2 = in.b0 ^ in.b1 ^ in.b5,
    .b3 = in.b1 ^ in.b2 ^ in.b6,
    .b4 = in.b0 ^ in.b2 ^ in.b7,
    .b5 = in.b0 ^ in.b1,
    .b6 = in.b1 ^ in.b2,
    .b7 = in.b2
  };
  return out;
}

/**
 * Performs bitsliced finite field multiplication by 39.
 */
__device__ Eightbits multiply39(Eightbits in) {
  Eightbits out = {
    .b0 = in.b2 ^ in.b5,
    .b1 = in.b3 ^ in.b6,
    .b2 = in.b4 ^ in.b7,
    .b3 = in.b5,
    .b4 = in.b2 ^ in.b5 ^ in.b6,
    .b5 = in.b0 ^ in.b2 ^ in.b3 ^ in.b5 ^ in.b6 ^ in.b7,
    .b6 = in.b0 ^ in.b1 ^ in.b3 ^ in.b4 ^ in.b6 ^ in.b7,
    .b7 = in.b1 ^ in.b4 ^ in.b7
  };
  return out;
}

/**
 * @brief Performs the HALFLOOP-24 inverted mix columns operation.
 *
 * @param in input value.
 * @return the output of the inverted mix columns operation on
 * the input value.
 */
__device__ Twentyfourbits bitslice_inv_mix_columns(Twentyfourbits in) {
  Twentyfourbits mul6 = {
    .msb = multiply6(in.msb),
    .mid = multiply6(in.mid),
    .lsb = multiply6(in.lsb)
  };
  Twentyfourbits mul8 = {
    .msb = multiply8(in.msb),
    .mid = multiply8(in.mid),
    .lsb = multiply8(in.lsb)
  };
  Twentyfourbits mul39 = {
    .msb = multiply39(in.msb),
    .mid = multiply39(in.mid),
    .lsb = multiply39(in.lsb)
  };
  Twentyfourbits out = {
    .msb = xor8(xor8(mul6.msb,  mul8.mid),  mul39.lsb),
    .mid = xor8(xor8(mul39.msb, mul6.mid),  mul8.lsb),
    .lsb = xor8(xor8(mul8.msb,  mul39.mid), mul6.lsb)
  };
  return out;
}

/**
 * Bitsliced implementation of one HALFLOOP decryption round.
 * @param state a 24-bit HALFLOOP state.
 * @param rk a 24-bit round key.
 */
__device__ Twentyfourbits bitslice_inv_round(
    Twentyfourbits state,
    Twentyfourbits rk) {
  state = bitslice_inv_mix_columns(xor24(state, rk));
  Twentyfourbits out = {
    .msb = bitslice_inv_sub_bytes(state.msb),
    .mid = bitslice_inv_sub_bytes(bitslice_inv_rotate_rows_6(state.mid)),
    .lsb = bitslice_inv_sub_bytes(bitslice_inv_rotate_rows_4(state.lsb))
  };
  return out;
}

/**
 * Performs the HALFLOOP SubBytes operation.
 * @param state the S-box input value.
 * @param SBOX_d the S-box lookup table. Must be a pointer to global device
 * memory.
 * @return the S-box output value.
 */
__device__ u32 sub_bytes_d(u32 state, u8 *SBOX_d) {
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  state = (__ldg(SBOX_d + a0) << 16) ^ (__ldg(SBOX_d + a1) << 8)
      ^ __ldg(SBOX_d + a2);
  return state;
}

/**
 * Performs the inverse HALFLOOP SubBytes operation.
 * @param state the S-box output value.
 * @param inv_SBOX_d the inverse S-box lookup table. Must be a pointer to global
 * device memory.
 * @return the S-box input value.
 */
__device__ u32 inv_sub_bytes_d(u32 state, u8 *inv_SBOX_d) {
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  state = (__ldg(inv_SBOX_d + a0) << 16) | (__ldg(inv_SBOX_d + a1) << 8)
      | __ldg(inv_SBOX_d + a2);
  return state;
}

/**
 * Performs the HALFLOOP MixColumns operation.
 * @param state the 24-bit input value.
 * @param table_2d lookup table for finite field multiplication by 2. Must be a
 * pointer to global device memory.
 * @param table_9d lookup table for finite field multiplication by 9. Must be a
 * pointer to global device memory.
 * @return the MixColumns output value.
 */
__device__ u32 mix_columns_d(u32 in, u8 *table_2d, u8 *table_9d) {
  u32 a = in >> 16;
  u32 b = (in >> 8) & 0xff;
  u32 c = in & 0xff;
  u32 out = (__ldg(table_9d + a) ^ b ^ __ldg(table_2d + c)) << 16;
  out    |= (__ldg(table_2d + a) ^ __ldg(table_9d + b) ^ c)  << 8;
  out    |= a ^ __ldg(table_2d + b) ^ __ldg(table_9d + c);
  return out;
}

/**
 * Performs the inverse HALFLOOP MixColumns operation.
 * @param state the 24-bit output value.
 * @param table_6d lookup table for finite field multiplication by 6. Must be a
 * pointer to global device memory.
 * @param table_8d lookup table for finite field multiplication by 8. Must be a
 * pointer to global device memory.
 * @param table_39d lookup table for finite field multiplication by 39. Must be
 * a pointer to global device memory.
 * @return the MixColumns input value.
 */
__device__ u32 inv_mix_columns_d(u32 in, u8 *table_6d, u8 *table_8d,
    u8 *table_39d) {
  u32 a = in >> 16;
  u32 b = (in >> 8) & 0xff;
  u32 c = in & 0xff;
  u32 out = (__ldg(table_6d + a)  ^ __ldg(table_8d + b)  ^ __ldg(table_39d + c))
      << 16;
  out    |= (__ldg(table_39d + a) ^ __ldg(table_6d + b)  ^ __ldg(table_8d + c))
      << 8;
  out    |= (__ldg(table_8d + a)  ^ __ldg(table_39d + b) ^ __ldg(table_6d + c));
  return out;
}

/**
 * Performs the HALFLOOP RotateRows operation.
 * @param state the 24-bit input value.
 * @return the RotateRows output value.
 */
__device__ u32 rotate_rows_d(u32 state) {
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  a1 = (a1 << 6) | (a1 >> 2);
  a2 = (a2 << 4) | (a2 >> 4);
  state = (a0 << 16) ^ (a1 << 8) ^ a2;
  return state;
}

/**
 * Performs the inverse HALFLOOP RotateRows operation.
 * @param state the 24-bit output value.
 * @return the RotateRows input value.
 */
__device__ u32 inv_rotate_rows_d(u32 state) {
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  a1 = (a1 >> 6) | (a1 << 2);
  a2 = (a2 >> 4) | (a2 << 4);
  state = (a0 << 16) ^ (a1 << 8) ^ a2;
  return state;
}

/**
 * @brief The kernel attempts to find value of round key 6 and the least
 * siginificant byte of round key 5 such that the two provided x6 states decrypt
 * to identical plaintexts. The tweaks for the two parallell encryptions must
 * differ only in the least significant bit of the tweak's word number field.
 * For correct operation, the kernel must be launched as 65536 simultaneous
 * threads.
 *
 * @param x6a x6 state for ciphertext A.
 * @param x6b x6 state for ciphertext B.
 * @param twa Array of 11 round tweaks for ciphertext A.
 * @param twb Array of 11 round tweaks for ciphertext B.
 * @param rk3 Round key 3.
 * @param rk4 Round key 4.
 * @param rk5 Round key 5.
 * @param rk7 Round key 7.
 * @param rk8a Round key 8.
 * @param found Output array for found matching keys. The LSB of round key 5 is
 * stored in the most significant 8 bits and round key 6 is stored in the
 * remaining 24 bits.
 * @param num_found Output variable for number of found keys in found.
 * @param t Lookup tables for HALFLOOP.
 */
__global__ void __launch_bounds__(128, 5) bitslice_kernel(
    u32 x6a,
    u32 x6b,
    u32 *twa,
    u32 *twb,
    u32 rk3,
    u32 rk4,
    u32 rk5,
    u32 rk7,
    u32 rk8a,
    u32 *found,
    int *num_found,
    CudaTables t) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  /* Iterate over all possible values of normalized rk6. In each iteration, the
   * lower 13 bits of rk6 are equal to the most significant 13 bits of idx. The
   * remaining 3 index bits are used as the 3 most significant bits of the least
   * significant byte of round key 5. */
  for (u64 rk6 = 0; rk6 < 0x1000000; rk6 += 8192) {
    /* x5 values are the same for all lanes. */
    u32 rk6a = (rk6 | (idx >> 3)) ^ twa[6];
    u32 rk6b = (rk6 | (idx >> 3)) ^ twb[6];
    u32 x5a = inv_sub_bytes_d(inv_rotate_rows_d(inv_mix_columns_d(x6a ^ rk6a,
        t.mul6, t.mul8, t.mul39)), t.inv_sbox);
    u32 x5b = inv_sub_bytes_d(inv_rotate_rows_d(inv_mix_columns_d(x6b ^ rk6b,
        t.mul6, t.mul8, t.mul39)), t.inv_sbox);

    /* Convert x5 to bitsliced representation. */
    Twentyfourbits statea = {
      .msb = get_eightbits(x5a >> 16),
      .mid = get_eightbits((x5a >> 8) & 0xff),
      .lsb = get_eightbits(x5a & 0xff)
    };
    Twentyfourbits stateb = {
      .msb = get_eightbits(x5b >> 16),
      .mid = get_eightbits((x5b >> 8) & 0xff),
      .lsb = get_eightbits(x5b & 0xff)
    };

    /* rk5 */
    Twentyfourbits rk;
    rk.msb = get_eightbits(rk5 >> 16);
    rk.mid = get_eightbits((rk5 >> 8) & 0xff);
    rk.lsb = get_eightbits(twa[5]);
    rk.lsb.b0 ^= GET_U32_BIT(idx, 29);
    rk.lsb.b1 ^= GET_U32_BIT(idx, 30);
    rk.lsb.b2 ^= GET_U32_BIT(idx, 31);
    rk.lsb.b3 ^= 0xFFFF0000U;
    rk.lsb.b4 ^= 0xFF00FF00U;
    rk.lsb.b5 ^= 0xF0F0F0F0U;
    rk.lsb.b6 ^= 0xCCCCCCCCU;
    rk.lsb.b7 ^= 0xAAAAAAAAU;
    statea = bitslice_inv_round(statea, rk);
    stateb = bitslice_inv_round(stateb, rk);

    /* rk4 */
    rk.msb = get_eightbits(rk4 >> 16);
    rk.mid = get_eightbits((rk4 >> 8) & 0xff);
    rk.lsb = get_eightbits(rk4 & 0xff);
    statea = bitslice_inv_round(statea, rk);
    stateb = bitslice_inv_round(stateb, rk);

    /* rk3 */
    rk.msb = get_eightbits(rk3 >> 16);
    rk.mid = get_eightbits((rk3 >> 8) & 0xff);
    rk.lsb = get_eightbits(rk3 & 0xff);
    statea = bitslice_inv_round(statea, rk);
    stateb = bitslice_inv_round(stateb, rk);

    /* rk2 */
    u8 rk2_0 = ((rk7 >> 8) ^ (rk6a >> 16)) & 0xff;
    u8 rk2_1 = (rk7 ^ (rk6a >> 8)) & 0xff;
    u8 rk2_2 = ((rk8a >> 16) ^ rk6a) & 0xff;
    rk.msb = get_eightbits(rk2_0);
    rk.mid = get_eightbits(rk2_1);
    rk.lsb = get_eightbits(rk2_2);
    statea = bitslice_inv_round(statea, rk);
    stateb = bitslice_inv_round(stateb, rk);

    /* We can check for equality of the plaintexts without adding rk0 and rk1 by
     * flipping the single bit that differs between the tweaks. */
    stateb.mid.b1 = ~stateb.mid.b1;

    /* Compare plaintexts. */
    Twentyfourbits diff = xor24(statea, stateb);
    u32 cmp;
    cmp  = diff.msb.b0;
    cmp |= diff.msb.b1;
    cmp |= diff.msb.b2;
    cmp |= diff.msb.b3;
    cmp |= diff.msb.b4;
    cmp |= diff.msb.b5;
    cmp |= diff.msb.b6;
    cmp |= diff.msb.b7;
    cmp |= diff.mid.b0;
    cmp |= diff.mid.b1;
    cmp |= diff.mid.b2;
    cmp |= diff.mid.b3;
    cmp |= diff.mid.b4;
    cmp |= diff.mid.b5;
    cmp |= diff.mid.b6;
    cmp |= diff.mid.b7;
    cmp |= diff.lsb.b0;
    cmp |= diff.lsb.b1;
    cmp |= diff.lsb.b2;
    cmp |= diff.lsb.b3;
    cmp |= diff.lsb.b4;
    cmp |= diff.lsb.b5;
    cmp |= diff.lsb.b6;
    cmp |= diff.lsb.b7;
    cmp = ~cmp;

    while (cmp != 0) {
      u32 low5 = 31 - __clz(cmp);
      cmp ^= 1U << low5;
      u32 hi3 = (idx & 0x7) << 29;
      found[atomicAdd(num_found, 1)] = hi3 | (low5 << 24) | (rk6a ^ twa[6]);
      *num_found += 1;
    }
  }
}

/**
 * Atomically increases a 4-bit counter using saturating arithmetic.
 * @param arr pointer to the array.
 * @param idx counter index.
 * @return the counter value after incrementing. A return value of 16 indicates
 * overflow and that the counter value has saturated at 115
 */
__device__ u32 atomicInc4(u32 *arr, u32 idx) {
  u32 q = idx >> 3;
  u32 *r = arr + q;
  u32 shift = idx * 4 - q * 32;
  u32 compare;
  u32 ret;
  u32 nr;
  do {
    compare = *r;
    ret = ((compare >> shift) & 0xf) + 1;
    u32 nval = ret > 15 ? 15 : ret;
    nr = (compare & ~(0xf << shift)) | (nval << shift);
  } while (atomicCAS(r, compare, nr) != compare);
  return ret;
}

#if __CUDA_ARCH__ < 800
/**
 * Atomically increases a 5-bit counter using saturating arithmetic.
 * @param arr pointer to the array.
 * @param idx counter index.
 * @return the counter value after incrementing. A return value of 32 indicates
 * overflow and that the counter value has saturated at 31.
 */
__device__ u32 atomicInc5(u32 *arr, u32 idx) {
  u32 q = (idx * 0xaaab) >> 18; /* Optimized 16-bit division by 6. */
  u32 *r = arr + q;
  u32 shift = idx * 5 - q * 30;
  u32 compare;
  u32 ret;
  u32 nr;
  do {
    compare = *r;
    ret = ((compare >> shift) & 0x1f) + 1;
    u32 nval = ret > 31 ? 31 : ret;
    nr = (compare & ~(0x1f << shift)) | (nval << shift);
  } while (atomicCAS(r, compare, nr) != compare);
  return ret;
}

#else /* _CUDA_ARCH__ >= 800 */
/**
 * Atomically increases a 8-bit counter using saturating arithmetic.
 * @param arr pointer to the array.
 * @param idx counter index.
 * @return the counter value after incrementing. A return value of 256 indicates
 * overflow and that the counter value has saturated at 255.
 */
__device__ u32 atomicInc8(u32 *arr, u32 idx) {
  u32 q = idx >> 2;
  u32 *r = arr + q;
  u32 shift = idx * 8 - q * 32;
  u32 compare;
  u32 ret;
  u32 nr;
  do {
    compare = *r;
    ret = ((compare >> shift) & 0xff) + 1;
    u32 nval = ret > 255 ? 255 : ret;
    nr = (compare & ~(0xff << shift)) | (nval << shift);
  } while (atomicCAS(r, compare, nr) != compare);
  return ret;
}
#endif /* __CUDA_ARCH__ */

/**
 * Recovers rk8 and rk9.
 *
 * @param x9 an array of precomputed x9 values for each ciphertext.
 * @param tw8 round 8 tweak for each ciphertext.
 * @param tw9 round 9 tweak for each ciphertext.
 * @param twd tweak delta for each ciphertext.
 * @param num_ct size of the x9 and tw arrays.
 * @param pairs pairs of ciphertexts with the required tweak difference.
 * Encoded as indexes in x9 and tw. One index is in the top 16 bits and one
 * index is in the bottom 16 bits.
 * @param num_pairs number of pairs in the pairs array.
 * @param rk9n2 the least significant byte of rk9.
 * @param tau1 threshold value.
 * @param v8s array of temporary storage for v8 values. The start of the array
 * for a particular block is at blockIdx * num_ct.
 * @param counts array of counters for the 24-bit candidate key.
 * @param candidates output array for found candidate keys. Round key 9 is
 * stored in the least significant 24 bits. Round key 8 is stored in the
 * following 24 bits.
 * @param num_candidates output variable for the number of keys in candidates.
 * @param warn set to true to indicate that matching keys may have been missed
 * due to internal match buffers overflowing.
 * @param t pointers to various tables needed by the algorithm.
 * @param profile if true, only a limited set of keys are searched. Used for
 * fast profiling.
 */
__global__ void __launch_bounds__(512, 2) ct_attack2(
    u32 * __restrict__ x9,
    u32 * __restrict__ tw8,
    u32 * __restrict__ tw9,
    u8 * __restrict__ twd,
    u32 num_ct,
    u32 * __restrict__ pairs,
    u32 num_pairs,
    u8 rk9n2, u32 tau1,
    u32 * __restrict__ v8s,
    u32 * __restrict__ counts,
    u64 * __restrict__ candidates,
    u32 * __restrict__ num_candidates,
    bool * warn,
    CudaTables t,
    bool profile) {

  /* Block storage in global memory. */
  u32 *v8 = v8s + blockIdx.x * num_ct;
  u32 *count = counts + blockIdx.x * ALG2_COUNTERS;

  /* Decrease search space when profiling. */
  u32 rk9n01max = 0x10000;
  if (profile) {
    rk9n01max = gridDim.x * 4;
  }

  for (u32 rk9n01 = blockIdx.x; rk9n01 < rk9n01max; rk9n01 += gridDim.x) {
    u32 rk9n = (rk9n01 << 8) | rk9n2; /* Normalized rk9. */
    /* Zeroize counters. */
    for (int i = threadIdx.x; i < ALG2_COUNTERS; i += blockDim.x) {
      count[i] = 0;
    }
    /* Precalculate v8 for all ciphertexts. */
    for (int i = threadIdx.x; i < num_ct; i += blockDim.x) {
      /* Generate rk9 from normalized rk9. */
      u32 rk9 = __ldg(tw9 + i) ^ rk9n;
      u32 x8 = inv_mix_columns_d(__ldg(x9 + i) ^ rk9, t.mul6, t.mul8, t.mul39);
      x8 = inv_rotate_rows_d(x8);
      x8 = inv_sub_bytes_d(x8, t.inv_sbox);
      u32 vv = inv_mix_columns_d(x8 ^ __ldg(tw8 + i), t.mul6, t.mul8, t.mul39);
      v8[i] = inv_rotate_rows_d(vv);
    }
    __syncthreads();
    /* Iterate over all pairs. */
    for (int p = threadIdx.x; p < num_pairs; p += blockDim.x) {
      u32 pair = __ldg(pairs + p);
      u32 pa = pair >> 16;
      u32 pb = pair & 0xffff;
      u32 delta = __ldg(twd + pa) ^ __ldg(twd + pb);
      u32 dy7 = v8[pa] ^ v8[pb]; /* y7 difference. */
      u32 y0idx = (dy7 >> 16) * 8;
      u32 y1idx = (((delta - 1) * 256) | ((dy7 >> 8) & 0xff)) * 8;
      u32 y2idx = (dy7 & 0xff) * 8;
      u32 w = __ldg(t.y0 + y0idx) & __ldg(t.y1 + y1idx) & __ldg(t.y2 + y2idx);
      u32 i = 0;
      while (true) {
        /* Ensure that each thread has a non-zero w variable as far as
         * possible. */
        while (w == 0) {
          i += 1;
          if (i == 8) {
            break;
          }
          w = __ldg(t.y0 + y0idx + i) & __ldg(t.y1 + y1idx + i)
              & __ldg(t.y2 + y2idx + i);
        }
        if (i == 8) {
          break;
        }
        u32 bit = 31 - __clz(w);
        w ^= 1U << bit;
        u32 dx72 = bit + i * 32; /* x7_2 difference. */
        u32 dx70 = FFMUL9(dx72);
        u32 dx71 = FFMUL2(dx72) ^ delta;
        u32 ddt0 = __ldg(t.ddt + (((dy7 & 0xff0000) >> 8) | dx70));
        u32 ddt1 = __ldg(t.ddt +  ((dy7 & 0x00ff00)       | dx71));
        u32 ddt2 = __ldg(t.ddt + (((dy7 & 0x0000ff) << 8) | dx72));
        /* Impossible differences. */
        if (ddt0 == 0xffff || ddt1 == 0xffff || ddt2 == 0xffff) {
          continue;
        }
        /* Each value in the DDT LUT represents two output values with a
         * difference that is implicit from the index. The diffs array is used
         * to generate all eight combinations. */
        u32 diffs[8] = {
          0,              dy7 & 0x0000ff, dy7 & 0x00ff00, dy7 & 0x00ffff,
          dy7 & 0xff0000, dy7 & 0xff00ff, dy7 & 0xffff00, dy7 & 0xffffff
        };
        do {
          u32 y72a = ddt2 & 0xff; /* Look up y7_2 value. */
          ddt2 >>= 8;
          u32 ddt0x = ddt0;
          do {
            u32 y70 = (ddt0x & 0xff) << 16; /* Look up y7_0 value. */
            ddt0x >>= 8;
            u32 ddt1x = ddt1;
            do {
              /* Look up y7_1 value. */
              u32 y701 = y70 | ((ddt1x & 0xff) << 8);
              ddt1x >>= 8;
              /* Generate candidate key. */
              u32 key = (y701 | y72a) ^ v8[pa];
              for (u32 j = 0; j < 8; j++) {
                u32 k = key ^ diffs[j];
                u32 val = atomicInc4(count, k);
                if (val == tau1) {
                  u32 idx = atomicAdd(num_candidates, 1);
                  if (idx < MAX_CANDIDATES) {
                    u64 rk8 = mix_columns_d(rotate_rows_d(k), t.mul2, t.mul9);
                    candidates[idx] = (rk8 << 24) | rk9n;
                  } else {
                    *warn = true;
                  }
                }
              }
            } while (ddt1x != 0);
          } while (ddt0x != 0);
        } while (ddt2 != 0);
      }
    }
    __syncthreads();
  }
}

/**
 * Recovers rk8 and rk9.
 *
 * @param x9 an array of precomputed x9 values for each ciphertext.
 * @param tw8 round 8 tweak for each ciphertext.
 * @param tw9 round 9 tweak for each ciphertext.
 * @param twd tweak delta for each ciphertext.
 * @param num_ct size of the x9 and tw arrays.
 * @param pairs pairs of ciphertexts with the required tweak difference.
 * Encoded as indexes in x9 and tw. One index is in the top 16 bits and one
 * index is in the bottom 16 bits.
 * @param num_pairs number of pairs in the pairs array.
 * @param rk9n2 the least significant byte of rk9.
 * @param tau1 threshold value.
 * @param tau2 threshold value.
 * @param v8s array of temporary storage for v8 values. The start of the array
 * for a particular block is at blockIdx * num_ct.
 * @param candidates output array for found candidate keys. Round key 9 is
 * stored in the least significant 24 bits. Round key 8 is stored in the
 * following 24 bits. The most significant 16 bits contains the number of
 * matching pairs for the key. The count of matching pairs saturates at 0xffff.
 * @param num_candidates output variable for the number of keys in candidates.
 * @param warn set to true to indicate that matching keys may have been missed
 * due to internal match buffers overflowing.
 * @param t pointers to various tables needed by the algorithm.
 * @param profile if true, only a limited set of keys are searched. Used for
 * fast profiling.
 */
__global__ void __launch_bounds__(1024, 2) ct_attack3(
    u32 * __restrict__ x9,
    u32 * __restrict__ tw8,
    u32 * __restrict__ tw9,
    u8 * __restrict__ twd,
    u32 num_ct,
    u32 * __restrict__ pairs,
    u32 num_pairs,
    u8 rk9n2,
    u32 tau1,
    u32 tau2,
    u32 * __restrict__ v8s,
    u64 * __restrict__ candidates,
    u32 * __restrict__ num_candidates,
    bool * warn,
    CudaTables t,
    bool profile) {

#if __CUDA_ARCH__ < 800
  constexpr int ncounters = 10923;
#else
  constexpr int ncounters = 16384;
#endif
  __shared__ u32 count[ncounters];   /* Counters for candidate keys. */
  __shared__ u16 keys[MAX_KEYS]; /* List of partial keys that reached the tau1
                                    threshold. */
  __shared__ u32 nkeys;          /* Length of keys list. */
  __shared__ u32 nmatch;         /* Length of matchp list. */
  __shared__ bool flag;          /* True if at least one key reached the tau2
                                    threshold. */

  /* Reuse count array memory. These are not use at the same time as count. */
  u32 *k1count = count;         /* Counter for matching middle key bytes. */
  u32 *matchp = k1count + 256;  /* List of matching pairs for candidate key. */
  u16 *x702 = (u16*)(matchp + MAX_MATCHES); /* x702 cache. */
  u8 *dx71x = (u8*)(x702 + num_ct); /* x7_1 difference without tweak difference
                                       removed for each value in matchp. */

  /* Block storage in global memory. */
  u32 *v8 = v8s + blockIdx.x * num_ct;

  /* Decrease search space when profiling. */
  u32 rk9n01max = 0x10000;
  if (profile) {
    rk9n01max = gridDim.x * 128;
  }

  for (u32 rk9n01 = blockIdx.x; rk9n01 < rk9n01max; rk9n01 += gridDim.x) {
    u32 rk9n = (rk9n01 << 8) | rk9n2; /* Normalized rk9. */
    /* Zeroize counters. */
    for (int i = threadIdx.x; i < ncounters; i += blockDim.x) {
      count[i] = 0;
    }
    if (threadIdx.x == 0) {
      nkeys = 0;
    }
    /* Precalculate v8 for all ciphertexts. */
    for (int i = threadIdx.x; i < num_ct; i += blockDim.x) {
      /* Generate rk9 from normalized rk9. */
      u32 rk9 = __ldg(tw9 + i) ^ rk9n;
      u32 x8 = inv_mix_columns_d(__ldg(x9 + i) ^ rk9, t.mul6, t.mul8, t.mul39);
      x8 = inv_rotate_rows_d(x8);
      x8 = inv_sub_bytes_d(x8, t.inv_sbox);
      u32 vv = inv_mix_columns_d(x8 ^ __ldg(tw8 + i), t.mul6, t.mul8, t.mul39);
      v8[i] = inv_rotate_rows_d(vv);
    }
    __syncthreads();
    /* Iterate over all pairs. */
    for (int p = threadIdx.x; p < num_pairs; p += blockDim.x) {
      u32 pair = __ldg(pairs + p);
      u32 pa = pair >> 16;
      u32 pb = pair & 0xffff;
      u32 delta = __ldg(twd + pa) ^ __ldg(twd + pb);
      u32 dy7 = v8[pa] ^ v8[pb]; /* y7 difference. */
      u32 v8x = ((v8[pa] << 8) | (v8[pa] >> 16)) & 0xffff;
      u32 y0idx = (dy7 >> 16) * 8;
      u32 y1idx = (((delta - 1) * 256) | ((dy7 >> 8) & 0xff)) * 8;
      u32 y2idx = (dy7 & 0xff) * 8;
      u32 w = __ldg(t.y0 + y0idx) & __ldg(t.y1 + y1idx) & __ldg(t.y2 + y2idx);
      u32 i = 0;
      while (true) {
        /* Ensure that each thread has a non-zero w variable as far as
         * possible. */
        while (w == 0) {
          i += 1;
          if (i == 8) {
            break;
          }
          w = __ldg(t.y0 + y0idx + i) & __ldg(t.y1 + y1idx + i)
              & __ldg(t.y2 + y2idx + i);
        }
        if (i == 8) {
          break;
        }
        u32 bit = 31 - __clz(w);
        w ^= 1U << bit;
        u32 dx72 = bit + i * 32; /* x7_2 difference. */
        u32 dx70 = FFMUL9(dx72);
        u32 dout0 = dy7 >> 16;
        u32 dout2 = dy7 & 0xff;
        u32 ddt0 = __ldg(t.ddt + ((dout0 << 8) | dx70));
        u32 ddt2 = __ldg(t.ddt + ((dout2 << 8) | dx72));
        if (ddt0 == 0xffff || ddt2 == 0xffff) { /* Impossible differences. */
          continue;
        }
        /* Each value in the DDT LUT represents two output values with a
         * difference that is implicit from the index. The diffs array
         * is used to generate all four combinations of a pair of values. */
        u32 diffs[4] = {0, dout2 << 8, dout0, (dout2 << 8) | dout0};
        do {
          u32 y72 = (ddt2 & 0xff) << 8;
          ddt2 >>= 8;
          u32 d0 = ddt0;
          do {
            u32 y70 = (d0 & 0xff);
            d0 >>= 8;
            /* Generate candidate key. */
            u32 key = (y72 | y70) ^ v8x;
            for (u32 j = 0; j < 4; j++) {
              u32 k = key ^ diffs[j];
#if __CUDA_ARCH__ < 800
              u32 val = atomicInc5(count, k);
#else
              u32 val = atomicInc8(count, k);
#endif
              if (val == tau1) {
                u32 keyidx = atomicAdd(&nkeys, 1);
                if (keyidx < MAX_KEYS) {
                  keys[keyidx] = k;
                } else {
                  *warn = true;
                }
              }
            }
          } while (d0 != 0);
        } while (ddt2 != 0);
      }
    }
    __syncthreads();
    /* Iterate over all partial LL^-1(rk8) that reached the tau1 threshold. */
    for (u32 i = 0; i < nkeys && i < MAX_KEYS; i++) {
      if (threadIdx.x == 0) {
        nmatch = 0;
      }
      u32 k0 = keys[i] & 0xff;
      u32 k2 = (keys[i] >> 8);
      /* Precalculate x7_0 and x7_2 for each ciphertext. */
      for (int j = threadIdx.x; j < num_ct; j += blockDim.x) {
        x702[j] = (u16)__ldg(t.inv_sbox + ((v8[j] & 0xff) ^ k2))
            | ((u16)__ldg(t.inv_sbox + ((v8[j] >> 16) ^ k0)) << 8);
      }
      __syncthreads();
      /* Find which pairs match for the candidate key. */
      for (int p = threadIdx.x; p < num_pairs; p += blockDim.x) {
        u32 pair = __ldg(pairs + p);
        u32 pa = pair >> 16;
        u32 pb = pair & 0xffff;
        u32 x702a = x702[pa];
        u32 x702b = x702[pb];
        u32 dx702 = x702a ^ x702b;
        u32 dx72 = dx702 & 0xff;
        u32 dx70 = FFMUL9(dx72);
        if ((dx702 >> 8) == dx70) {
          u32 idx = atomicAdd(&nmatch, 1);
          if (idx < MAX_MATCHES) {
            matchp[idx] = pair;
            dx71x[idx] = FFMUL2(dx72);
          } else {
            *warn = true;
          }
        }
      }
      for (int j = threadIdx.x; j < 256; j += blockDim.x) {
        k1count[j] = 0;
      }
      if (threadIdx.x == 0) {
        flag = false;
      }
      __syncthreads();
      /* For each matching pair, test the middle key bytes. */
      for (int j = threadIdx.x; j < nmatch && j < MAX_MATCHES;
          j += blockDim.x) {
        u32 pa = matchp[j] >> 16;
        u32 pb = matchp[j] & 0xffff;
        u32 delta = __ldg(twd + pa) ^ __ldg(twd + pb);
        u32 dx71 = dx71x[j] ^ delta;
        u32 dy71 = ((v8[pa] ^ v8[pb]) >> 8) & 0xff;
        u32 ddt1 = __ldg(t.ddt + dy71 * 256 + dx71);
        if (ddt1 == 0xffff) { /* Impossible difference. */
          continue;
        }
        do {
          u32 y71 = ddt1 & 0xff;
          ddt1 >>= 8;
          u32 k1 = y71 ^ ((v8[pa] >> 8) & 0xff);
          for (int i = 0; i < 2; i++) {
            /* Each DDT LUT entry represents two values with a known
             * difference. */
            u32 k = k1 ^ i * dy71;
            u32 val = atomicAdd(k1count + k, 1);
            if (val == tau2) {
              flag = true;
            }
          }
        } while (ddt1 != 0);
      }
      __syncthreads();
      for (int k1 = threadIdx.x; flag && k1 < 256; k1 += blockDim.x) {
        if (k1count[k1] >= tau2) {
          u32 idx = atomicAdd(num_candidates, 1);
          if (idx < MAX_CANDIDATES) {
            u64 rk8 = rotate_rows_d((k0 << 16) | (k1 << 8) | k2);
            rk8 = mix_columns_d(rk8, t.mul2, t.mul9);
            if (k1count[k1] > 0xffff) {
              k1count[k1] = 0xffff;
            }
            candidates[idx] = (u64)(k1count[k1]) << 48 | (rk8 << 24) | rk9n;
          } else {
            *warn = true;
          }
        }
      }
    }
  }
}

/**
 * Initializes a lookup table entry for looking up viable x2 differences from
 * y0 differences.
 *
 * @param y2delta the y0 output difference to create a table entry for.
 * @param lut output pointer for the created table entry.
 */
static halfloop_result_t init_y0_lut_cuda(u8 y0delta, u32 *lut) {
  CHECK_BAD_ARGUMENT(lut == NULL);

  memset(lut, 0, sizeof(u32) * 8);

  if (y0delta == 0) {
    return HALFLOOP_SUCCESS;
  }

  /* Temporary lookup table for converting x0 to x2. */
  u8 w[256] = {0};
  for (u32 gamma = 1; gamma < 256; gamma++) {
    u32 v = mix_columns(gamma << 16);
    w[v >> 16] = v & 0xff;
  }

  for (u32 i = 0; i < 256; i++) {
    u32 v = w[inv_SBOX[i] ^ inv_SBOX[i ^ y0delta]];
    lut[v >> 5] |= 1U << (v & 0x1f);
  }

  return HALFLOOP_SUCCESS;
}

/**
 * Initializes a lookup table entry for looking up viable x2 differences from
 * y1 differences.
 * @param y1delta the y1 output difference to create a table entry for.
 * @param delta the known tweak difference to create a table entry for.
 * @param lut output pointer for the created table entry.
 */
static halfloop_result_t init_y1_lut_cuda(u8 y1delta, u8 delta, u32 *lut) {
  CHECK_BAD_ARGUMENT(delta == 0);
  CHECK_BAD_ARGUMENT(lut == NULL);

  memset(lut, 0, sizeof(u32) * 8);

  /* Temporary lookup table for converting x1 to x2. */
  u8 w[256] = {0};
  for (u32 gamma = 1; gamma < 256; gamma++) {
    u32 v = mix_columns(gamma << 16);
    w[(v >> 8) & 0xff] = v & 0xff;
  }

  u32 l[8] = {0};
  for (u32 i = 0; i < 256; i++) {
    u32 v = w[inv_SBOX[i] ^ inv_SBOX[i ^ y1delta] ^ delta];
    l[v >> 5] |= 1U << (v & 0x1f);
  }

  /* Calculate viable gamma values. */
  u32 g[8] = {0};
  for (u32 i = 0; i < 256; i++) {
    u32 gamma = (u32)(SBOX[i] ^ SBOX[i ^ delta]);
    g[gamma >> 5] |= 1U << (gamma & 0x1f);
  }

  for (int i = 0; i < 8; i++) {
    lut[i] = l[i] & g[i];
  }

  return HALFLOOP_SUCCESS;
}

/**
 * Initializes a lookup table entry for looking up viable x2 differences from
 * y2 differences.
 *
 * @param y0delta the y2 output difference to create a table entry for.
 * @param lut output pointer for the created table entry.
 */
static halfloop_result_t init_y2_lut_cuda(u8 y2delta, u32 *lut) {
  CHECK_BAD_ARGUMENT(lut == NULL);

  memset(lut, 0, sizeof(u32) * 8);
  for (u32 i = 0; i < 256; i++) {
    u32 v = (u32)(inv_SBOX[i] ^ inv_SBOX[i ^ y2delta]);
    lut[v >> 5] |= 1U << (v & 0x1f);
  }

  return HALFLOOP_SUCCESS;
}

/**
 * Creates a data structure on the device and initializes it with data from host
 * memory.
 * @param device return pointer for the allocated device memory.
 * @param host pointer to the host memory buffer that will be copied to the
 * allocated device memory. If NULL, the memory will not be initialized.
 * @param size the size of the allocated buffer.
 */
static halfloop_result_t create_cuda_device_table(
    void **device,
    const void *host,
    size_t size) {
  CHECK_BAD_ARGUMENT(device == NULL);
  CHECK_BAD_ARGUMENT(size == 0);
  *device = NULL;
  halfloop_result_t err = HALFLOOP_SUCCESS;
  RETURN_ON_CUDA_ERROR(cudaMalloc(device, size));
  if (host != NULL) {
    RETURN_ON_CUDA_ERROR(cudaMemcpy(*device, host, size,
        cudaMemcpyHostToDevice));
  }
error:
  if (err != HALFLOOP_SUCCESS) {
    if (*device != NULL) {
      cudaFree(*device);
      *device = NULL;
    }
  }
  return err;
}

/** Initializes the CUDA driver API by calling cuInit. */
halfloop_result_t halfloop_init_cuda(void) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  RETURN_ON_CUDA_RESULT(cuInit(0));
error:
  return err;
}

/**
 * Initializes a CudaTables struct.
 * @param t the struct to initialize.
 */
static halfloop_result_t init_cuda_tables(CudaTables *t) {
  CHECK_BAD_ARGUMENT(t == NULL);
  u32 *y0_lut = NULL;
  u32 *y1_lut = NULL;
  u32 *y2_lut = NULL;
  u16 *ddt = NULL;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  y0_lut = (u32*)calloc(256 * 8,       sizeof(u32));
  y1_lut = (u32*)calloc(256 * 256 * 8, sizeof(u32));
  y2_lut = (u32*)calloc(256 * 8,       sizeof(u32));
  ddt    = (u16*)calloc(256 * 256,     sizeof(u16));
  RETURN_IF(y0_lut == NULL || y1_lut == NULL || y2_lut == NULL || ddt == NULL,
      HALFLOOP_MEMORY_ERROR);

  RETURN_ON_ERROR(init_ddt(ddt));

  for (int ydelta = 0; ydelta < 256; ydelta++) {
    RETURN_ON_ERROR(init_y0_lut_cuda((u8)ydelta, y0_lut + ydelta * 8));
    RETURN_ON_ERROR(init_y2_lut_cuda((u8)ydelta, y2_lut + ydelta * 8));
  }

  for (int delta = 1; delta < 256; delta++) {
    for (int y1delta = 0; y1delta < 256; y1delta++) {
      RETURN_ON_ERROR(init_y1_lut_cuda((u8)y1delta, (u8)delta,
          y1_lut + ((delta - 1) * 256 + y1delta) * 8));
    }
  }

  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->sbox,     SBOX,           0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->inv_sbox, inv_SBOX,       0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->mul2,     ffmul_table_2,  0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->mul6,     ffmul_table_6,  0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->mul8,     ffmul_table_8,  0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->mul9,     ffmul_table_9,  0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->mul39,    ffmul_table_39, 0x100));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->y0,    y0_lut, sizeof(u32) * 2048));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->y1,    y1_lut, sizeof(u32) * 522240));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->y2,    y2_lut, sizeof(u32) * 2048));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&t->ddt,   ddt, 256 * 256 * sizeof(u16)));
error:
  free(y0_lut);
  free(y1_lut);
  free(y2_lut);
  free(ddt);
  return err;
}

/**
 * Frees a CudaTables struct.
 * @param t the struct to free.
 */
static halfloop_result_t free_cuda_tables(CudaTables *t) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  RETURN_ON_CUDA_ERROR(cudaFree(t->sbox));
  t->sbox = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->inv_sbox));
  t->inv_sbox = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->mul2));
  t->mul2 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->mul6));
  t->mul6 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->mul8));
  t->mul8 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->mul9));
  t->mul9 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->mul39));
  t->mul39 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->y0));
  t->y0 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->y1));
  t->y1 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->y2));
  t->y2 = NULL;
  RETURN_ON_CUDA_ERROR(cudaFree(t->ddt));
  t->ddt = NULL;
error:
  return err;
}

halfloop_result_t test_halfloop_cuda_bitslice(void) {
  hltimer timer;
  u32 pt = 0;
  u32 ct0 = 0;
  u32 ct1 = 0;
  u64 tweak = 0;
  hlkey key = {0};
  u32 rk[11];
  u32 tw0[11];
  u32 tw1[11];
  u32 *found = NULL;
  u32 rk56;
  int num_found = 0;
  double elapsed;
  bool ok = false;
  hlkey zerokey = { 0 };
  halfloop_result_t err = HALFLOOP_SUCCESS;

  RETURN_ON_ERROR(random_bytes(&pt, sizeof(u32)));
  RETURN_ON_ERROR(random_bytes(&tweak, sizeof(u64)));
  RETURN_ON_ERROR(random_bytes(&key, sizeof(hlkey)));
  pt &= 0xffffff;

  RETURN_ON_ERROR(halfloop_encrypt(pt, key, tweak, &ct0));
  RETURN_ON_ERROR(halfloop_encrypt(pt, key, tweak ^ (1 << 30), &ct1));
  RETURN_ON_ERROR(key_schedule(rk, key, 0));
  RETURN_ON_ERROR(key_schedule(tw0, zerokey, tweak));
  RETURN_ON_ERROR(key_schedule(tw1, zerokey, 0));
  rk56 = ((rk[5] & 0xff) << 24) | rk[6];

  print_message("Benchmarking CUDA bitslice algorithm.", WHITE);
  TIMER_START(&timer);
  RETURN_ON_ERROR(halfloop_cuda_bitslice(
      ct0,
      ct1,
      tweak,
      rk[7],
      rk[8],
      rk[9],
      rk[10],
      &found,
      &num_found));
  TIMER_STOP(&timer);
  elapsed = timer_elapsed(timer);
  print_message(
      "Number of keys found during bitslice test: %d.",
      WHITE,
      num_found);
  setlocale(LC_NUMERIC, "");
  print_message(
      "Test took %.2f seconds: %'lld keys/second.",
      WHITE,
      elapsed,
      (u64)(0x100000000ULL / elapsed));

  for (int i = 0; i < num_found && !ok; i++) {
    if (rk56 == found[i]) {
      ok = true;
    }
  }
  RETURN_IF(!ok, HALFLOOP_INTERNAL_ERROR);
  print_message("Bitslice implementation ok.", WHITE);
error:
  if (err != HALFLOOP_SUCCESS) {
    if (err != HALFLOOP_SUCCESS) {
      print_message(
          "CUDA bitslice benchmark failed. PT=%06x tweak=%016" PRIx64
              " Key=%016" PRIx64 "%016" PRIx64,
          RED,
          pt,
          tweak,
          key.hi,
          key.lo);
    }
  }
  free(found);
  return err;
}

halfloop_result_t halfloop_list_cuda_devices(
    int *num_devices,
    char **device_names) {
  CHECK_BAD_ARGUMENT(num_devices == NULL);
  CHECK_BAD_ARGUMENT(device_names == NULL);

  *num_devices = 0;
  *device_names = NULL;
  cudaDeviceProp prop = {0};
  halfloop_result_t err = HALFLOOP_SUCCESS;


  RETURN_ON_CUDA_ERROR(cudaGetDeviceCount(num_devices));
  *device_names = (char*)calloc(*num_devices, (size_t)256);
  RETURN_IF(*device_names == NULL, HALFLOOP_MEMORY_ERROR);
  for (int i = 0; i < *num_devices; i++) {
    RETURN_ON_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
    memcpy(*device_names + 256 * i, prop.name, 256);
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*device_names);
    *num_devices = 0;
  }
  return err;
}

halfloop_result_t halfloop_cuda_bitslice(
    u32 cta,
    u32 ctb,
    u64 tw,
    u32 rk7n,
    u32 rk8n,
    u32 rk9n,
    u32 rk10n,
    u32 **found,
    int *num_found) {
  CHECK_BAD_ARGUMENT(cta   & 0xff000000);
  CHECK_BAD_ARGUMENT(ctb   & 0xff000000);
  CHECK_BAD_ARGUMENT(rk7n  & 0xff000000);
  CHECK_BAD_ARGUMENT(rk8n  & 0xff000000);
  CHECK_BAD_ARGUMENT(rk9n  & 0xff000000);
  CHECK_BAD_ARGUMENT(rk10n & 0xff000000);
  CHECK_BAD_ARGUMENT(found == NULL);
  CHECK_BAD_ARGUMENT(num_found == NULL);

  *found = NULL;
  *num_found = 0;
  u32 *found_d = NULL;
  int *num_found_d = NULL;
  u32 *twa_d = NULL;
  u32 *twb_d = NULL;
  u32 rk10;
  u32 rk9a;
  u32 rk9b;
  u32 rk8a;
  u32 rk8b;
  u32 twa[11];
  u32 twb[11];
  u32 tw0[11];
  u32 rk7;
  u32 x9a;
  u32 x9b;
  u32 x8a;
  u32 x8b;
  u32 x7a;
  u32 x7b;
  u32 x6a;
  u32 x6b;
  u32 rk3;
  u32 rk4;
  u32 rk50;
  u32 rk51;
  u32 rk5;
  CudaTables tables = {0};
  hlkey zerokey = {0};
  halfloop_result_t err = HALFLOOP_SUCCESS;

  RETURN_ON_CUDA_ERROR(cudaMalloc(&found_d, 1024 * sizeof(u32)));
  RETURN_ON_CUDA_ERROR(cudaMalloc(&num_found_d, sizeof(int)));
  RETURN_ON_CUDA_ERROR(cudaMalloc(&twa_d, 11 * sizeof(u32)));
  RETURN_ON_CUDA_ERROR(cudaMalloc(&twb_d, 11 * sizeof(u32)));

  /* Initialize lookup tables. */
  RETURN_ON_ERROR(init_cuda_tables(&tables));

  /* Prepare round tweaks. */
  RETURN_ON_ERROR(key_schedule(twa, zerokey, tw));
  RETURN_ON_ERROR(key_schedule(twb, zerokey, tw ^ (1 << 30)));
  RETURN_ON_ERROR(key_schedule(tw0, zerokey, 0));
  for (int i = 0; i < 11; i++) {
    twa[i] ^= tw0[i];
    twb[i] ^= tw0[i];
  }
  RETURN_ON_ERROR(halfloop_round10_tweak(tw, rk9n & 0xff, twa + 10));
  RETURN_ON_ERROR(halfloop_round10_tweak(
      tw ^ (1 << 30),
      rk9n & 0xff,
      twb + 10));
  RETURN_ON_CUDA_ERROR(cudaMemcpy(
      twa_d,
      twa,
      11 * sizeof(u32),
      cudaMemcpyHostToDevice));
  RETURN_ON_CUDA_ERROR(cudaMemcpy(
      twb_d,
      twb,
      11 * sizeof(u32),
      cudaMemcpyHostToDevice));
  RETURN_ON_CUDA_ERROR(cudaMemcpy(
      num_found_d,
      num_found,
      sizeof(int),
      cudaMemcpyHostToDevice));
  /* Calculate x6 and known round keys. */
  rk10 = rk10n ^ twa[10];
  rk9a = rk9n  ^ twa[9];
  rk9b = rk9n  ^ twb[9];
  rk8a = rk8n  ^ twa[8];
  rk8b = rk8n  ^ twb[8];
  rk7  = rk7n  ^ twa[7];
  x9a = inv_sub_bytes(inv_rotate_rows(cta ^ rk10));
  x9b = inv_sub_bytes(inv_rotate_rows(ctb ^ rk10));
  x8a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9a ^ rk9a)));
  x8b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9b ^ rk9b)));
  x7a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x8a ^ rk8a)));
  x7b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x8b ^ rk8b)));
  x6a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x7a ^ rk7)));
  x6b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x7b ^ rk7)));
  rk3 = (((rk8a << 8) & 0xffff00) | (rk9a >> 16)) ^ rk7;
  rk4 = (((rk9a << 8) & 0xffff00) | (rk10 >> 16)) ^ rk8a;
  rk50 = ((rk10 << 8) ^ rk9a) & 0xff0000;
  rk51 = ((rk10 & 0xff) ^ 2 ^ SBOX[rk9a & 0xff]) << 8;
  rk5 = rk50 | rk51;

  bitslice_kernel<<<1024,64>>>(
      x6a,
      x6b,
      twa_d,
      twb_d,
      rk3,
      rk4,
      rk5,
      rk7,
      rk8a,
      found_d,
      num_found_d,
      tables);
  RETURN_ON_CUDA_ERROR(cudaGetLastError());

  RETURN_ON_CUDA_ERROR(cudaMemcpy(
      num_found,
      num_found_d,
      sizeof(int),
      cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  *found = (u32*)malloc(*num_found * sizeof(u32));
  RETURN_IF(num_found != 0 && *found == NULL, HALFLOOP_MEMORY_ERROR);
  RETURN_ON_CUDA_ERROR(cudaMemcpy(
      *found,
      found_d,
      *num_found * sizeof(u32),
      cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*found);
    *num_found = 0;
  }
  if (found_d != NULL) {
    cudaFree(found_d);
  }
  if (num_found_d != NULL) {
    cudaFree(num_found_d);
  }
  if (twa_d != NULL) {
    cudaFree(twa_d);
  }
  if (twb_d != NULL) {
    cudaFree(twb_d);
  }
  free_cuda_tables(&tables);
  return err;
}

/**
 * Fetches the next job for an attack worker thread.
 * @param arg pointer to a ThreadArg struct that contains the search state.
 * @param job return pointer for the job. The lower 24 bits will contain the
 * rk10 to search and the upper 8 bits will contain one byte from rk9. If the
 * function returns false, the contents of job are undefined.
 * @return true if a new job is available and false if there are no more jobs to
 * perform.
 */
static bool get_next_job(ThreadArg *arg, u32 *job) {
  CHECK_BAD_ARGUMENT(arg == NULL);
  CHECK_BAD_ARGUMENT(job == NULL);
  arg->mutex.lock();
  if (arg->num_fixed == 32) {
    if (arg->next_job == 0) {
      *job = arg->fixed_bits;
      arg->next_job = 1;
      arg->mutex.unlock();
      return true;
    }
    arg->mutex.unlock();
    return false;
  }
  if (arg->next_job >= (1ULL << (32 - arg->num_fixed))) {
    arg->mutex.unlock();
    return false;
  }
  *job = (u32)((arg->next_job << arg->num_fixed) | arg->fixed_bits);
  arg->next_job += 1;
  arg->mutex.unlock();
  return true;
}

/**
 * Calculates the x9 states for a list of ciphertext-tweak tuples.
 * @param ct a list of ciphertext-tweak tuples.
 * @param num_ct the number of tuples in ct.
 * @param rk10n normalized round key for the last round.
 * @param rk9n2 last byte of the normalized round key for the second last round.
 * @param x9 output list for the calculated x9 states. Must have room for at
 * least num_ct items.
 */
static halfloop_result_t precalculate_x9(
    const tuple_t * __restrict__ ct,
    int num_ct,
    u32 rk10n,
    u8 rk9n2,
    u32 * __restrict__ x9) {
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(num_ct < 2);
  CHECK_BAD_ARGUMENT(x9 == NULL);
  halfloop_result_t err = HALFLOOP_SUCCESS;
  for (int i = 0; i < num_ct; i++) {
    /* Generate rk10 from normalized rk10 and LSB of rk9. */
    u32 rk10;
    RETURN_ON_ERROR(halfloop_round10_tweak(ct[i].tweak, rk9n2, &rk10));
    rk10 ^= rk10n;
    x9[i] = inv_sub_bytes(inv_rotate_rows(ct[i].ct ^ rk10));
  }
error:
  return err;
}

/**
 * Validates a list of candidate keys returned from the attack kernel.
 * @param arg pointer to a ThreadArg struct that contains the search state.
 * @param candidates a list of candidate keys. Round key 9 is stored in the
 * least significant 24 bits. Round key 8 is stored in the following 24 bits.
 * The most significant 16 bits contains the number of matching pairs for the
 * key.
 * @param num_candidates the number of keys in the candidates list.
 * @param tau2 threshold value.
 * @param x9 a list of precalculated x9 states.
 * @param rk10 round key 10.
 */
static halfloop_result_t validate_candidates(
    ThreadArg *arg,
    u64 * __restrict__ candidates,
    u32 num_candidates,
    u32 tau2,
    const u32 * __restrict__ x9,
    u32 rk10,
    candidate_key_t ** __restrict__ validated,
    u32 * __restrict__ num_validated) {
  CHECK_BAD_ARGUMENT(arg == NULL);
  CHECK_BAD_ARGUMENT(candidates == NULL);
  CHECK_BAD_ARGUMENT(num_candidates == 0);
  CHECK_BAD_ARGUMENT(x9 == NULL);
  CHECK_BAD_ARGUMENT(validated == NULL);
  CHECK_BAD_ARGUMENT(num_validated == NULL);
  u32 *v8 = NULL;
  *validated = NULL;
  *num_validated = 0;
  u32 validated_alloc = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  v8 = (u32*)malloc(sizeof(u32) * arg->num_ct);
  RETURN_IF(v8 == NULL, HALFLOOP_MEMORY_ERROR);

  for (u32 i = 0; i < num_candidates; i++) {
    u8 rk7 = 0;
    u32 keycount = 0;
    u32 rk9n = candidates[i] & 0xffffff;
    u32 rk8 = (candidates[i] >> 24) & 0xffffff;
    for (int j = 0; j < arg->num_ct; j++) {
      /* Generate rk9 from normalized rk9. */
      u32 rk9;
      RETURN_ON_ERROR(halfloop_round_tweak(arg->ct[j].tweak, 9, &rk9));
      rk9 ^= rk9n;
      /* Calculate and store v8. */
      u32 x8 = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9[j] ^ rk9)));
      u32 tw8;
      RETURN_ON_ERROR(halfloop_round_tweak(arg->ct[j].tweak, 8, &tw8));
      v8[j] = inv_rotate_rows(inv_mix_columns(x8 ^ tw8));
    }
    /* Validate the candidate key against the expected v7 difference and
     * recover the MSB of LL^-1(rk7). */
    RETURN_ON_ERROR(validate_rk8(
        arg->ct,
        v8,
        arg->num_ct,
        arg->pairs,
        arg->num_pairs,
        rk8,
        &rk7,
        &keycount));
    if (keycount >= tau2) {
      if (validated_alloc == *num_validated) {
        validated_alloc += 10;
        candidate_key_t *tmp = (candidate_key_t*)realloc(
            *validated,
            sizeof(candidate_key_t) * validated_alloc);
        RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
        *validated = tmp;
      }
      candidate_key_t *k = *validated + *num_validated;
      k->pairs = keycount;
      k->rk7 = rk7;
      k->rk8 = rk8;
      k->rk9 = rk9n;
      k->rk10 = rk10;
      *num_validated += 1;
      arg->mutex.lock();
      print_message(
          "Found candidate key: %02x %06x %06x %06x (%d matching pairs)",
          GREEN,
          rk7,
          rk8,
          rk9n,
          rk10,
          keycount);
      arg->mutex.unlock();
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*validated);
    *num_validated = 0;
  }
  free(v8);
  return err;
}

static void* ct_attack_thread(void *a) {
  ThreadArg *arg = (ThreadArg*)a;
  CudaTables tables = {0};

  u32 *tw8_h = NULL;    /* Round 8 tweak for each ciphertext. */
  u32 *tw8_d = NULL;    /* Copy destination for tw8_h. */
  u32 *tw9_h = NULL;    /* Round 9 tweak for each ciphertext. */
  u32 *tw9_d = NULL;    /* Copy destination for tw9_h. */
  u8 *twd_h = NULL;     /* Tweak delta for each ciphertext. */
  u8 *twd_d = NULL;     /* Copy destination for twd_h. */
  u32 *x9_h = NULL;     /* x9 state for each ciphertext. */
  u32 *x9_d = NULL;     /* Copy destination for x9_h. */
  u32 *pairs_h = NULL;  /* Ciphertext indexes for each pair.*/
  u32 *pairs_d = NULL;  /* Copy destination for pairs_h. */
  u32 *v8s_d = NULL;    /* Device memory buffer for precalculated v8 states. */
  u32 *count_d = NULL;  /* Counters for algorithm 2. */
  u64 *candidates_h = NULL; /* Copy destination for candidates_d. */
  u64 *candidates_d = NULL; /* Device memory buffer for candidate keys. */
  u32 num_candidates_h = 0; /* Copy destination for num_candidates_d. */
  u32 *num_candidates_d = NULL; /* Counter for number of candidate keys. */
  u32 num_validated = 0;        /* Number of keys in validated array. */
  candidate_key_t *validated = NULL; /* Array of validated candidate keys. */
  bool warn = false;    /* CUDA kernel warning flag. */
  bool *warn_d = NULL;  /* CUDA kernel warning flag on device. */
  int nthreads;         /* Number of threads per block. */
  int blocks;           /* Number of blocks. */
  cudaStream_t stream = {0}; /* Stream used for the various CUDA operations. */
  cudaDeviceProp deviceProp = {0}; /* CUDA device properties. */
  u32 tau1 = arg->tau1; /* Tau1 value. */
  halfloop_result_t err = HALFLOOP_SUCCESS;

  arg->mutex.lock();
  int threadnum = (int)arg->threadnum;
  arg->threadnum += 1;
  arg->mutex.unlock();
  int devicenum = threadnum >> 1;
  if (devicenum >= arg->num_devices) {
    return NULL;
  }
  cudaSetDevice(arg->devices[devicenum]);
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  RETURN_ON_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, devicenum));
  if (arg->verbose) {
    print_message(
        "Thread %d: Device: %s CC: %d.%d Shared mem: %d Multiprocessors: %d",
        WHITE,
        threadnum,
        deviceProp.name,
        deviceProp.major,
        deviceProp.minor,
        deviceProp.sharedMemPerBlock,
        deviceProp.multiProcessorCount);
  }

  nthreads = arg->algorithm == GPU_ATTACK2 ? 512 : 1024;
  blocks = MIN(deviceProp.multiProcessorCount * arg->blockmul, 65536);
  if (deviceProp.major < 8 && arg->tau1 > 31) {
    if ((threadnum & 1) == 0) { /* Only print one warning per device. */
      print_message(
          "WARNING: Adjusting tau1 to 31 on device %d",
          RED,
          devicenum);
    }
    tau1 = 31;
  }

  RETURN_ON_ERROR(init_cuda_tables(&tables));

  tw8_h           = (u32*)malloc(sizeof(u32) * arg->num_ct);
  tw9_h           = (u32*)malloc(sizeof(u32) * arg->num_ct);
  twd_h           = (u8*)malloc(sizeof(u8) * arg->num_ct);
  x9_h            = (u32*)malloc(sizeof(u32) * arg->num_ct);
  pairs_h         = (u32*)malloc(sizeof(u32) * arg->num_pairs);
  candidates_h = (u64*)malloc(sizeof(u64) * MAX_CANDIDATES);
  RETURN_IF(tw8_h == NULL
            || tw9_h == NULL
            || twd_h == NULL
            || pairs_h == NULL
            || x9_h == NULL
            || pairs_h == NULL
            || candidates_h == NULL,
      HALFLOOP_MEMORY_ERROR);

  /* Initialize constant tweaks and pairs arrays. */
  for (int i = 0; i < arg->num_ct; i++) {
    u64 tw = arg->ct[i].tweak;
    tw8_h[i] = ((tw >> 40) ^ (tw >> 8)) & 0xffffff;
    tw9_h[i] = (((tw << 16) & 0xff0000) | ((tw >> 48) & 0x00ffff))
        ^ ((tw >> 16) & 0xffffff);
    twd_h[i] = (tw >> 40) & 0xff;
  }
  for (int i = 0; i < arg->num_pairs; i++) {
    RETURN_IF(arg->pairs[i].a > 65535, HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(arg->pairs[i].b > 65535, HALFLOOP_INTERNAL_ERROR);
    pairs_h[i] = (u32)((arg->pairs[i].a << 16) | arg->pairs[i].b);
  }
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&tw8_d, tw8_h, sizeof(u32) * arg->num_ct));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&tw9_d, tw9_h, sizeof(u32) * arg->num_ct));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&twd_d, twd_h, sizeof(u8) * arg->num_ct));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(
      &pairs_d,
      pairs_h,
      sizeof(u32) * arg->num_pairs));
  FREE_AND_NULL(tw8_h);
  FREE_AND_NULL(tw9_h);
  FREE_AND_NULL(twd_h);
  FREE_AND_NULL(pairs_h);

  /* Allocate device memory. */
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&x9_d, NULL, sizeof(u32) * arg->num_ct));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(
      &v8s_d,
      NULL,
      sizeof(u32) * arg->num_ct * blocks));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(
      &candidates_d,
      NULL,
      sizeof(u64) * MAX_CANDIDATES));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(
      &num_candidates_d,
      &num_candidates_h,
      sizeof(u32)));
  RETURN_ON_ERROR(CREATE_CUDA_TABLE(&warn_d, &warn, sizeof(bool)));
  if (arg->algorithm == GPU_ATTACK2) {
    RETURN_ON_ERROR(CREATE_CUDA_TABLE(
        &count_d,
        NULL,
        sizeof(u32) * ALG2_COUNTERS * blocks));
  }
  RETURN_ON_CUDA_ERROR(cudaStreamCreate(&stream));

  arg->barrier->arrive_and_wait();

  while (arg->run) {
    u32 job;
    if (!get_next_job(arg, &job)) {
      if (arg->verbose) {
        arg->mutex.lock();
        print_message("No more jobs for thread %d.", WHITE, threadnum);
        arg->mutex.unlock();
      }
      break;
    }
    if (arg->verbose) {
      arg->mutex.lock();
      print_message("Thread %d got job %08x.", WHITE, threadnum, job);
      arg->mutex.unlock();
    }
    u32 rk9n2 = job >> 24;
    u32 rk10n = job & 0xffffff;
    RETURN_ON_ERROR(precalculate_x9(
        arg->ct,
        arg->num_ct,
        rk10n,
        (u8)rk9n2,
        x9_h));
    RETURN_ON_CUDA_ERROR(cudaMemcpyAsync(
        x9_d,
        x9_h,
        sizeof(u32) * arg->num_ct,
        cudaMemcpyHostToDevice,
        stream));
    if (arg->algorithm == GPU_ATTACK2) {
      ct_attack2<<<blocks, nthreads, 0, stream>>>(
          x9_d,
          tw8_d,
          tw9_d,
          twd_d,
          arg->num_ct,
          pairs_d,
          arg->num_pairs,
          rk9n2,
          tau1,
          v8s_d,
          count_d,
          candidates_d,
          num_candidates_d,
          warn_d,
          tables,
          arg->profile);
    } else {
      ct_attack3<<<blocks, nthreads, 0, stream>>>(
          x9_d,
          tw8_d,
          tw9_d,
          twd_d,
          arg->num_ct,
          pairs_d,
          arg->num_pairs,
          rk9n2,
          tau1,
          arg->tau2,
          v8s_d,
          candidates_d,
          num_candidates_d,
          warn_d,
          tables,
          arg->profile);
    }
    RETURN_ON_CUDA_ERROR(cudaMemcpyAsync(
        &num_candidates_h,
        num_candidates_d,
        sizeof(u32),
        cudaMemcpyDeviceToHost,
        stream));
    RETURN_ON_CUDA_ERROR(cudaMemcpyAsync(
        candidates_h,
        candidates_d,
        sizeof(u64) * MAX_CANDIDATES,
        cudaMemcpyDeviceToHost,
        stream));
    RETURN_ON_CUDA_ERROR(cudaMemcpyAsync(
        &warn,
        warn_d,
        sizeof(bool),
        cudaMemcpyDeviceToHost,
        stream));
    cudaStreamSynchronize(stream);
    if (warn) {
      arg->mutex.lock();
      print_message("WARNING: Candidate keys may have been missed.", RED);
      arg->mutex.unlock();
    }
    if (arg->verbose) {
      arg->mutex.lock();
      print_message(
          "Thread %d job %08x found %d candidate%s.",
          WHITE,
          threadnum,
          job,
          num_candidates_h,
          num_candidates_h == 1 ? "" : "s");
      arg->mutex.unlock();
    }
    if (num_candidates_h > 0) {
      if (num_candidates_h > MAX_CANDIDATES) {
        num_candidates_h = MAX_CANDIDATES;
      }
      num_validated = 0;
      RETURN_ON_ERROR(validate_candidates(
          arg,
          candidates_h,
          num_candidates_h,
          arg->tau2,
          x9_h,
          rk10n,
          &validated,
          &num_validated));
      if (num_validated > 0) {
        arg->mutex.lock();
        candidate_key_t *tmp = (candidate_key_t*)realloc(
            arg->validated,
            sizeof(candidate_key_t) * (arg->num_validated + num_validated));
        if (tmp == NULL) {
          arg->mutex.unlock();
          RETURN_IF(true, HALFLOOP_MEMORY_ERROR);
        }
        arg->validated = tmp;
        memcpy(
            arg->validated + arg->num_validated,
            validated,
            sizeof(candidate_key_t) * num_validated);
        arg->num_validated += num_validated;
        arg->mutex.unlock();
        FREE_AND_NULL(validated);
      }
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    arg->run = false;
  }
  free_cuda_tables(&tables);
  cudaFree(tw8_d);
  cudaFree(tw9_d);
  cudaFree(twd_d);
  cudaFree(x9_d);
  cudaFree(pairs_d);
  cudaFree(v8s_d);
  cudaFree(candidates_d);
  cudaFree(num_candidates_d);
  cudaFree(count_d);
  free(tw8_h);
  free(tw9_h);
  free(twd_h);
  free(x9_h);
  free(pairs_h);
  free(candidates_h);
  free(validated);
  cudaStreamDestroy(stream);
  return NULL;
}

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
    int num_devices) {
  CHECK_BAD_ARGUMENT(algo != GPU_ATTACK2 && algo != GPU_ATTACK3);
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(num_ct < 2);
  CHECK_BAD_ARGUMENT(pairs == NULL);
  CHECK_BAD_ARGUMENT(num_pairs < 1);
  CHECK_BAD_ARGUMENT(tau1 < 1);
  CHECK_BAD_ARGUMENT(tau2 < 1 && algo == GPU_ATTACK3);
  CHECK_BAD_ARGUMENT(blockmul == 0);
  CHECK_BAD_ARGUMENT(candidates == NULL);
  CHECK_BAD_ARGUMENT(num_candidates == NULL);

  ThreadArg arg = {
    .algorithm = algo,
    .ct = ct,
    .pairs = pairs,
    .validated = NULL,
    .next_job = 0,
    .fixed_bits = fixed_bits,
    .threadnum = 0,
    .tau1 = tau1,
    .tau2 = tau2,
    .blockmul = blockmul,
    .num_validated = 0,
    .devices = {0},
    .num_devices = 0,
    .num_ct = num_ct,
    .num_pairs = num_pairs,
    .num_fixed = num_fixed,
    .run = true,
    .profile = profile,
    .verbose = verbose
  };
  std::vector<std::thread> threads;
  int ndevs = 0;
  *candidates = NULL;
  *num_candidates = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;
  RETURN_ON_CUDA_ERROR(cudaGetDeviceCount(&ndevs));
  RETURN_IF(ndevs == 0, HALFLOOP_FAILURE);
  if (devices == NULL || num_devices == 0) {
    for (int i = 0; i < MAX_DEVICES && i < ndevs; i++) {
      arg.devices[arg.num_devices++] = i;
    }
  } else {
    for (int i = 0; i < MAX_DEVICES && i < num_devices; i++) {
      if (devices[i] < ndevs) {
        arg.devices[arg.num_devices++] = devices[i];
      }
    }
  }
  arg.barrier = std::make_unique<std::barrier<>>(arg.num_devices * 2);

  if (verbose) {
    print_message("Starting %d threads.", WHITE, arg.num_devices * 2);
  }

  for (int i = 0; i < arg.num_devices * 2; i++) {
    threads.emplace_back(ct_attack_thread, &arg);
  }
  for (int i = 0; i < arg.num_devices * 2; i++) {
    threads[(u64)i].join();
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    free(arg.validated);
  } else {
    *candidates = arg.validated;
    *num_candidates = arg.num_validated;
  }
  return err;
}
