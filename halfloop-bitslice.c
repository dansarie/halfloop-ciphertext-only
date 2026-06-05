/* Bitslice implementation of attacks on HALFLOOP-24.

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

#define _GNU_SOURCE
#include <immintrin.h>
#include <locale.h>

#include "halfloop-bitslice.h"
#include "halfloop-common.h"

typedef __m256i u256;

/**
 * @brief Represents a byte in the bitslice implementation.
 */
typedef struct {
  u256 b0; /* MSB */
  u256 b1;
  u256 b2;
  u256 b3;
  u256 b4;
  u256 b5;
  u256 b6;
  u256 b7; /* LSB */
} eightbits;

/**
 * @brief Represents three bytes in the bitslice implementation.
 */
typedef struct {
  eightbits msb;
  eightbits mid;
  eightbits lsb;
} twentyfourbits;

/**
 * Bitsliced XOR of an 8-bit state.
 */
static eightbits xor8(eightbits a, eightbits b) {
  eightbits out = {
    .b0 = _mm256_xor_si256(a.b0, b.b0),
    .b1 = _mm256_xor_si256(a.b1, b.b1),
    .b2 = _mm256_xor_si256(a.b2, b.b2),
    .b3 = _mm256_xor_si256(a.b3, b.b3),
    .b4 = _mm256_xor_si256(a.b4, b.b4),
    .b5 = _mm256_xor_si256(a.b5, b.b5),
    .b6 = _mm256_xor_si256(a.b6, b.b6),
    .b7 = _mm256_xor_si256(a.b7, b.b7)
  };
  return out;
}

/**
 * Bitsliced XOR of a 24-bit state.
 */
static twentyfourbits xor24(twentyfourbits a, twentyfourbits b) {
  twentyfourbits out = {
    .msb = xor8(a.msb, b.msb),
    .mid = xor8(a.mid, b.mid),
    .lsb = xor8(a.lsb, b.lsb)
  };
  return out;
}

/**
 * Broadcasts an eight-bit value to all bitslice lanes.
 */
static eightbits get_eightbits(u8 bits) {
  eightbits out = {
    .b0 = _mm256_set1_epi64x(0ULL - (bits >> 7)),
    .b1 = _mm256_set1_epi64x(0ULL - ((bits >> 6) & 1)),
    .b2 = _mm256_set1_epi64x(0ULL - ((bits >> 5) & 1)),
    .b3 = _mm256_set1_epi64x(0ULL - ((bits >> 4) & 1)),
    .b4 = _mm256_set1_epi64x(0ULL - ((bits >> 3) & 1)),
    .b5 = _mm256_set1_epi64x(0ULL - ((bits >> 2) & 1)),
    .b6 = _mm256_set1_epi64x(0ULL - ((bits >> 1) & 1)),
    .b7 = _mm256_set1_epi64x(0ULL - (bits & 1))
  };
  return out;
}

/**
 * @brief Gate network implementation of the inverse Rijndael S-box.
 * @see https://cs-www.cs.yale.edu/homes/peralta/CircuitStuff/AESReverseDepth.txt
 */
static eightbits bitslice_inv_sub_bytes(eightbits in) {
  u256 ones = _mm256_set1_epi32(0xffffffff);
  u256 T23 = _mm256_xor_si256(in.b0, in.b3);
  u256 T22 = _mm256_xor_si256(_mm256_xor_si256(in.b1, in.b3), ones);
  u256 T2  = _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b1), ones);
  u256 T1  = _mm256_xor_si256(in.b3, in.b4);
  u256 T24 = _mm256_xor_si256(_mm256_xor_si256(in.b4, in.b7), ones);
  u256 R5  = _mm256_xor_si256(in.b6, in.b7);
  u256 T8  = _mm256_xor_si256(_mm256_xor_si256(in.b1, T23), ones);
  u256 T19 = _mm256_xor_si256(T22,   R5);
  u256 T9  = _mm256_xor_si256(_mm256_xor_si256(in.b7, T1), ones);
  u256 T10 = _mm256_xor_si256(T2,    T24);
  u256 T13 = _mm256_xor_si256(T2,    R5);
  u256 T3  = _mm256_xor_si256(T1,    R5);
  u256 T25 = _mm256_xor_si256(_mm256_xor_si256(in.b2, T1), ones);
  u256 R13 = _mm256_xor_si256(in.b1, in.b6);
  u256 T17 = _mm256_xor_si256(_mm256_xor_si256(in.b2, T19), ones);
  u256 T20 = _mm256_xor_si256(T24,   R13);
  u256 T4  = _mm256_xor_si256(in.b4, T8);
  u256 R17 = _mm256_xor_si256(_mm256_xor_si256(in.b2, in.b5), ones);
  u256 R18 = _mm256_xor_si256(_mm256_xor_si256(in.b5, in.b6), ones);
  u256 R19 = _mm256_xor_si256(_mm256_xor_si256(in.b2, in.b4), ones);
  u256 Y5  = _mm256_xor_si256(in.b0, R17);
  u256 T6  = _mm256_xor_si256(T22,   R17);
  u256 T16 = _mm256_xor_si256(R13,   R19);
  u256 T27 = _mm256_xor_si256(T1,    R18);
  u256 T15 = _mm256_xor_si256(T10,   T27);
  u256 T14 = _mm256_xor_si256(T10,   R18);
  u256 T26 = _mm256_xor_si256(T3,    T16);
  u256 M1  = _mm256_and_si256(T13,   T6);
  u256 M2  = _mm256_and_si256(T23,   T8);
  u256 M3  = _mm256_xor_si256(T14,   M1);
  u256 M4  = _mm256_and_si256(T19,   Y5);
  u256 M5  = _mm256_xor_si256(M4,    M1);
  u256 M6  = _mm256_and_si256(T3,    T16);
  u256 M7  = _mm256_and_si256(T22,   T9);
  u256 M8  = _mm256_xor_si256(T26,   M6);
  u256 M9  = _mm256_and_si256(T20,   T17);
  u256 M10 = _mm256_xor_si256(M9,    M6);
  u256 M11 = _mm256_and_si256(T1,    T15);
  u256 M12 = _mm256_and_si256(T4,    T27);
  u256 M13 = _mm256_xor_si256(M12,   M11);
  u256 M14 = _mm256_and_si256(T2,    T10);
  u256 M15 = _mm256_xor_si256(M14,   M11);
  u256 M16 = _mm256_xor_si256(M3,    M2);
  u256 M17 = _mm256_xor_si256(M5,    T24);
  u256 M18 = _mm256_xor_si256(M8,    M7);
  u256 M19 = _mm256_xor_si256(M10,   M15);
  u256 M20 = _mm256_xor_si256(M16,   M13);
  u256 M21 = _mm256_xor_si256(M17,   M15);
  u256 M22 = _mm256_xor_si256(M18,   M13);
  u256 M23 = _mm256_xor_si256(M19,   T25);
  u256 M24 = _mm256_xor_si256(M22,   M23);
  u256 M25 = _mm256_and_si256(M22,   M20);
  u256 M26 = _mm256_xor_si256(M21,   M25);
  u256 M27 = _mm256_xor_si256(M20,   M21);
  u256 M28 = _mm256_xor_si256(M23,   M25);
  u256 M29 = _mm256_and_si256(M28,   M27);
  u256 M30 = _mm256_and_si256(M26,   M24);
  u256 M31 = _mm256_and_si256(M20,   M23);
  u256 M32 = _mm256_and_si256(M27,   M31);
  u256 M33 = _mm256_xor_si256(M27,   M25);
  u256 M34 = _mm256_and_si256(M21,   M22);
  u256 M35 = _mm256_and_si256(M24,   M34);
  u256 M36 = _mm256_xor_si256(M24,   M25);
  u256 M37 = _mm256_xor_si256(M21,   M29);
  u256 M38 = _mm256_xor_si256(M32,   M33);
  u256 M39 = _mm256_xor_si256(M23,   M30);
  u256 M40 = _mm256_xor_si256(M35,   M36);
  u256 M41 = _mm256_xor_si256(M38,   M40);
  u256 M42 = _mm256_xor_si256(M37,   M39);
  u256 M43 = _mm256_xor_si256(M37,   M38);
  u256 M44 = _mm256_xor_si256(M39,   M40);
  u256 M45 = _mm256_xor_si256(M42,   M41);
  u256 M46 = _mm256_and_si256(M44,   T6);
  u256 M47 = _mm256_and_si256(M40,   T8);
  u256 M48 = _mm256_and_si256(M39,   Y5);
  u256 M49 = _mm256_and_si256(M43,   T16);
  u256 M50 = _mm256_and_si256(M38,   T9);
  u256 M51 = _mm256_and_si256(M37,   T17);
  u256 M52 = _mm256_and_si256(M42,   T15);
  u256 M53 = _mm256_and_si256(M45,   T27);
  u256 M54 = _mm256_and_si256(M41,   T10);
  u256 M55 = _mm256_and_si256(M44,   T13);
  u256 M56 = _mm256_and_si256(M40,   T23);
  u256 M57 = _mm256_and_si256(M39,   T19);
  u256 M58 = _mm256_and_si256(M43,   T3);
  u256 M59 = _mm256_and_si256(M38,   T22);
  u256 M60 = _mm256_and_si256(M37,   T20);
  u256 M61 = _mm256_and_si256(M42,   T1);
  u256 M62 = _mm256_and_si256(M45,   T4);
  u256 M63 = _mm256_and_si256(M41,   T2);
  u256 P0  = _mm256_xor_si256(M52,   M61);
  u256 P1  = _mm256_xor_si256(M58,   M59);
  u256 P2  = _mm256_xor_si256(M54,   M62);
  u256 P3  = _mm256_xor_si256(M47,   M50);
  u256 P4  = _mm256_xor_si256(M48,   M56);
  u256 P5  = _mm256_xor_si256(M46,   M51);
  u256 P6  = _mm256_xor_si256(M49,   M60);
  u256 P7  = _mm256_xor_si256(P0,    P1);
  u256 P8  = _mm256_xor_si256(M50,   M53);
  u256 P9  = _mm256_xor_si256(M55,   M63);
  u256 P10 = _mm256_xor_si256(M57,   P4);
  u256 P11 = _mm256_xor_si256(P0,    P3);
  u256 P12 = _mm256_xor_si256(M46,   M48);
  u256 P13 = _mm256_xor_si256(M49,   M51);
  u256 P14 = _mm256_xor_si256(M49,   M62);
  u256 P15 = _mm256_xor_si256(M54,   M59);
  u256 P16 = _mm256_xor_si256(M57,   M61);
  u256 P17 = _mm256_xor_si256(M58,   P2);
  u256 P18 = _mm256_xor_si256(M63,   P5);
  u256 P19 = _mm256_xor_si256(P2,    P3);
  u256 P20 = _mm256_xor_si256(P4,    P6);
  u256 P22 = _mm256_xor_si256(P2,    P7);
  u256 P23 = _mm256_xor_si256(P7,    P8);
  u256 P24 = _mm256_xor_si256(P5,    P7);
  u256 P25 = _mm256_xor_si256(P6,    P10);
  u256 P26 = _mm256_xor_si256(P9,    P11);
  u256 P27 = _mm256_xor_si256(P10,   P18);
  u256 P28 = _mm256_xor_si256(P11,   P25);
  u256 P29 = _mm256_xor_si256(P15,   P20);
  eightbits out;
  out.b0  = _mm256_xor_si256(P13,    P22);
  out.b1  = _mm256_xor_si256(P26,    P29);
  out.b2  = _mm256_xor_si256(P17,    P28);
  out.b3  = _mm256_xor_si256(P12,    P22);
  out.b4  = _mm256_xor_si256(P23,    P27);
  out.b5  = _mm256_xor_si256(P19,    P24);
  out.b6  = _mm256_xor_si256(P14,    P23);
  out.b7  = _mm256_xor_si256(P9,     P16);
  return out;
}

/**
 * @brief Performs the rotate left operation six steps.
 *
 * @param in input value.
 * @return eightbits the input value rotated right six steps.
 */
static eightbits bitslice_inv_rotate_rows_6(eightbits in) {
  eightbits out = {
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
 * @return eightbits the input value rotated right four steps.
 */
static eightbits bitslice_inv_rotate_rows_4(eightbits in) {
  eightbits out = {
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
static eightbits multiply6(eightbits in) {
  eightbits out = {
    .b0 = _mm256_xor_si256(in.b1, in.b2),
    .b1 = _mm256_xor_si256(in.b2, in.b3),
    .b2 = _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b3), in.b4),
    .b3 = _mm256_xor_si256(_mm256_xor_si256(in.b1, in.b4), in.b5),
    .b4 = _mm256_xor_si256(
        _mm256_xor_si256(in.b0, in.b1),
        _mm256_xor_si256(in.b5, in.b6)),
    .b5 = _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b6), in.b7),
    .b6 = _mm256_xor_si256(in.b1, in.b7),
    .b7 = _mm256_xor_si256(in.b0, in.b1)
  };
  return out;
}

/**
 * Performs bitsliced finite field multiplication by 8.
 */
static eightbits multiply8(eightbits in) {
  eightbits out = {
    .b0 = in.b3,
    .b1 = _mm256_xor_si256(in.b0, in.b4),
    .b2 = _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b1), in.b5),
    .b3 = _mm256_xor_si256(_mm256_xor_si256(in.b1, in.b2), in.b6),
    .b4 = _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b2), in.b7),
    .b5 = _mm256_xor_si256(in.b0, in.b1),
    .b6 = _mm256_xor_si256(in.b1, in.b2),
    .b7 = in.b2
  };
  return out;
}

/**
 * Performs bitsliced finite field multiplication by 39.
 */
static eightbits multiply39(eightbits in) {
  eightbits out = {
    .b0 = _mm256_xor_si256(in.b2, in.b5),
    .b1 = _mm256_xor_si256(in.b3, in.b6),
    .b2 = _mm256_xor_si256(in.b4, in.b7),
    .b3 = in.b5,
    .b4 = _mm256_xor_si256(_mm256_xor_si256(in.b2, in.b5), in.b6),
    .b5 = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b2),
                         _mm256_xor_si256(in.b3, in.b5)),
        _mm256_xor_si256(in.b6, in.b7)),
    .b6 = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(in.b0, in.b1),
                         _mm256_xor_si256(in.b3, in.b4)),
        _mm256_xor_si256(in.b6, in.b7)),
    .b7 = _mm256_xor_si256(_mm256_xor_si256(in.b1, in.b4), in.b7)
  };
  return out;
}

/**
 * @brief Performs the HALFLOOP-24 inverted mix columns operation.
 *
 * @param in input value.
 * @return twentyfourbits the output of the inverted mix columns operation on
 * the input value.
 */
static twentyfourbits bitslice_inv_mix_columns(twentyfourbits in) {
  twentyfourbits mul6 = {
    .msb = multiply6(in.msb),
    .mid = multiply6(in.mid),
    .lsb = multiply6(in.lsb)
  };
  twentyfourbits mul8 = {
    .msb = multiply8(in.msb),
    .mid = multiply8(in.mid),
    .lsb = multiply8(in.lsb)
  };
  twentyfourbits mul39 = {
    .msb = multiply39(in.msb),
    .mid = multiply39(in.mid),
    .lsb = multiply39(in.lsb)
  };
  twentyfourbits out = {
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
static twentyfourbits bitslice_inv_round(twentyfourbits state,
    twentyfourbits rk) {
  state = bitslice_inv_mix_columns(xor24(state, rk));
  twentyfourbits out = {
    .msb = bitslice_inv_sub_bytes(state.msb),
    .mid = bitslice_inv_sub_bytes(bitslice_inv_rotate_rows_6(state.mid)),
    .lsb = bitslice_inv_sub_bytes(bitslice_inv_rotate_rows_4(state.lsb))
  };
  return out;
}

halfloop_result_t halfloop_bitslice(u32 cta, u32 ctb, u64 tw, u32 rk7n,
    u32 rk8n, u32 rk9n, u32 rk10n, u32 **found, int *num_found) {
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
  halfloop_result_t err = HALFLOOP_SUCCESS;

  int alloc = 300;
  *found = malloc(alloc * sizeof(u32));
  RETURN_IF(*found == NULL, HALFLOOP_MEMORY_ERROR);

  /* Prepare round tweaks. */
  u32 twa[11];
  u32 twb[11];
  u32 tw0[11];
  hlkey zero_key = {0};
  RETURN_ON_ERROR(key_schedule(twa, zero_key, tw));
  RETURN_ON_ERROR(key_schedule(twb, zero_key, tw ^ (1 << 30)));
  RETURN_ON_ERROR(key_schedule(tw0, zero_key, 0));
  for (int i = 0; i < 11; i++) {
    twa[i] ^= tw0[i];
    twb[i] ^= tw0[i];
  }
  RETURN_ON_ERROR(halfloop_round10_tweak(tw, rk9n & 0xff, twa + 10));
  RETURN_ON_ERROR(halfloop_round10_tweak(tw ^ (1 << 30), rk9n & 0xff,
      twb + 10));

  /* Calculate x6 and known round keys. */
  u32 rk10 = rk10n ^ twa[10];
  u32 rk9a = rk9n  ^ twa[9];
  u32 rk9b = rk9n  ^ twb[9];
  u32 rk8a = rk8n  ^ twa[8];
  u32 rk8b = rk8n  ^ twb[8];
  u32 rk7  = rk7n  ^ twa[7];
  u32 x9a = inv_sub_bytes(inv_rotate_rows(cta ^ rk10));
  u32 x9b = inv_sub_bytes(inv_rotate_rows(ctb ^ rk10));
  u32 x8a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9a ^ rk9a)));
  u32 x8b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9b ^ rk9b)));
  u32 x7a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x8a ^ rk8a)));
  u32 x7b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x8b ^ rk8b)));
  u32 x6a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x7a ^ rk7)));
  u32 x6b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x7b ^ rk7)));
  u32 rk3 = (((rk8a << 8) & 0xffff00) | (rk9a >> 16)) ^ rk7;
  u32 rk4 = (((rk9a << 8) & 0xffff00) | (rk10 >> 16)) ^ rk8a;
  u32 rk50 = ((rk10 << 8) ^ rk9a) & 0xff0000;
  u32 rk51 = ((rk10 & 0xff) ^ 2 ^ SBOX[rk9a & 0xff]) << 8;
  u32 rk5 = rk50 | rk51;

  /* Iterate over all possible values of normalized rk6. */
  for (u64 rk6 = 0; rk6 < 0x1000000; rk6++) {
    /* x5 values are the same for all lanes. */
    u32 rk6a = (u32)rk6 ^ twa[6];
    u32 rk6b = (u32)rk6 ^ twb[6];
    u32 x5a = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x6a ^ rk6a)));
    u32 x5b = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x6b ^ rk6b)));

    /* Convert x5 to bitsliced representation. */
    twentyfourbits statea = {
      .msb = get_eightbits((u8)(x5a >> 16)),
      .mid = get_eightbits((u8)(x5a >> 8)),
      .lsb = get_eightbits((u8)(x5a & 0xff))
    };
    twentyfourbits stateb = {
      .msb = get_eightbits((u8)(x5b >> 16)),
      .mid = get_eightbits((u8)(x5b >> 8)),
      .lsb = get_eightbits((u8)(x5b & 0xff))
    };

    /* rk5 */
    twentyfourbits rk;
    rk.msb = get_eightbits((u8)(rk5 >> 16));
    rk.mid = get_eightbits((u8)(rk5 >> 8));
    rk.lsb = get_eightbits((u8)twa[5]);
    rk.lsb.b0 = _mm256_xor_si256(
        rk.lsb.b0,
        _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFULL,
                          0xFFFFFFFFFFFFFFFFULL,
                          0x0000000000000000ULL,
                          0x0000000000000000ULL));
    rk.lsb.b1 = _mm256_xor_si256(
        rk.lsb.b1,
        _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFULL,
                          0x0000000000000000ULL,
                          0xFFFFFFFFFFFFFFFFULL,
                          0x0000000000000000ULL));
    rk.lsb.b2 = _mm256_xor_si256(
        rk.lsb.b2,
        _mm256_set1_epi64x(0xFFFFFFFF00000000ULL));
    rk.lsb.b3 = _mm256_xor_si256(
        rk.lsb.b3,
        _mm256_set1_epi64x(0xFFFF0000FFFF0000ULL));
    rk.lsb.b4 = _mm256_xor_si256(
        rk.lsb.b4,
        _mm256_set1_epi64x(0xFF00FF00FF00FF00ULL));
    rk.lsb.b5 = _mm256_xor_si256(
        rk.lsb.b5,
        _mm256_set1_epi64x(0xF0F0F0F0F0F0F0F0ULL));
    rk.lsb.b6 = _mm256_xor_si256(
        rk.lsb.b6,
        _mm256_set1_epi64x(0xCCCCCCCCCCCCCCCCULL));
    rk.lsb.b7 = _mm256_xor_si256(
        rk.lsb.b7,
        _mm256_set1_epi64x(0xAAAAAAAAAAAAAAAAULL));
    statea = bitslice_inv_round(statea, rk);
    stateb = bitslice_inv_round(stateb, rk);

    /* rk4 */
    rk.msb = get_eightbits((u8)(rk4 >> 16));
    rk.mid = get_eightbits((u8)(rk4 >> 8));
    rk.lsb = get_eightbits((u8)rk4);
    statea = bitslice_inv_round(statea, rk);
    stateb = bitslice_inv_round(stateb, rk);

    /* rk3 */
    rk.msb = get_eightbits((u8)(rk3 >> 16));
    rk.mid = get_eightbits((u8)(rk3 >> 8));
    rk.lsb = get_eightbits((u8)rk3);
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
    stateb.mid.b1 = _mm256_xor_si256(
        stateb.mid.b1,
        _mm256_set1_epi32(0xffffffff));

    /* Compare plaintexts. */
    twentyfourbits diff = xor24(statea, stateb);
    u256 cmp = diff.msb.b0;
    cmp = _mm256_or_si256(cmp, diff.msb.b1);
    cmp = _mm256_or_si256(cmp, diff.msb.b2);
    cmp = _mm256_or_si256(cmp, diff.msb.b3);
    cmp = _mm256_or_si256(cmp, diff.msb.b4);
    cmp = _mm256_or_si256(cmp, diff.msb.b5);
    cmp = _mm256_or_si256(cmp, diff.msb.b6);
    cmp = _mm256_or_si256(cmp, diff.msb.b7);
    cmp = _mm256_or_si256(cmp, diff.mid.b0);
    cmp = _mm256_or_si256(cmp, diff.mid.b1);
    cmp = _mm256_or_si256(cmp, diff.mid.b2);
    cmp = _mm256_or_si256(cmp, diff.mid.b3);
    cmp = _mm256_or_si256(cmp, diff.mid.b4);
    cmp = _mm256_or_si256(cmp, diff.mid.b5);
    cmp = _mm256_or_si256(cmp, diff.mid.b6);
    cmp = _mm256_or_si256(cmp, diff.mid.b7);
    cmp = _mm256_or_si256(cmp, diff.lsb.b0);
    cmp = _mm256_or_si256(cmp, diff.lsb.b1);
    cmp = _mm256_or_si256(cmp, diff.lsb.b2);
    cmp = _mm256_or_si256(cmp, diff.lsb.b3);
    cmp = _mm256_or_si256(cmp, diff.lsb.b4);
    cmp = _mm256_or_si256(cmp, diff.lsb.b5);
    cmp = _mm256_or_si256(cmp, diff.lsb.b6);
    cmp = _mm256_or_si256(cmp, diff.lsb.b7);
    cmp = _mm256_xor_si256(cmp, _mm256_set1_epi32(0xffffffff));

    u64 cmpa[4];
    _mm256_storeu_si256((u256*)cmpa, cmp);
    for (int i = 0; i < 4; i++) {
      while (cmpa[i] != 0) {
        if (*num_found == alloc) {
          alloc += 300;
          u32 *tmp = realloc(*found, alloc * sizeof(u32));
          RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
          *found = tmp;
        }
        int low6 = FFSL(cmpa[i]) - 1;
        (*found)[*num_found] = (((i << 6) | low6) << 24) | (u32)rk6;
        cmpa[i] ^= 1ULL << low6;
        *num_found += 1;
      }
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*found);
    *num_found = 0;
  }
  return err;
}

/**
 * @brief Returns the least significant bit in each of the eight variables in
 * the input struct. Used for testing.
 */
static u8 get_lower_bits(eightbits in) {
  u8 ret = (_mm256_cvtsi256_si32(in.b0) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b1) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b2) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b3) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b4) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b5) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b6) & 1);
  ret = (ret << 1) | (_mm256_cvtsi256_si32(in.b7) & 1);
  return ret;
}

/**
 * @brief Returns true if, for each of the eight variables in the input struct,
 * all 256 bits are equal. Used for testing.
 */
static bool check_equal_bits(eightbits v) {
  u256 ones = _mm256_set1_epi32(0xffffffff);
  eightbits inv = {
    .b0 = _mm256_xor_si256(v.b0, ones),
    .b1 = _mm256_xor_si256(v.b1, ones),
    .b2 = _mm256_xor_si256(v.b2, ones),
    .b3 = _mm256_xor_si256(v.b3, ones),
    .b4 = _mm256_xor_si256(v.b4, ones),
    .b5 = _mm256_xor_si256(v.b5, ones),
    .b6 = _mm256_xor_si256(v.b6, ones),
    .b7 = _mm256_xor_si256(v.b7, ones),
  };
  return (_mm256_testz_si256(v.b0, v.b0) || _mm256_testz_si256(inv.b0, inv.b0))
      && (_mm256_testz_si256(v.b1, v.b1) || _mm256_testz_si256(inv.b1, inv.b1))
      && (_mm256_testz_si256(v.b2, v.b2) || _mm256_testz_si256(inv.b2, inv.b2))
      && (_mm256_testz_si256(v.b3, v.b3) || _mm256_testz_si256(inv.b3, inv.b3))
      && (_mm256_testz_si256(v.b4, v.b4) || _mm256_testz_si256(inv.b4, inv.b4))
      && (_mm256_testz_si256(v.b5, v.b5) || _mm256_testz_si256(inv.b5, inv.b5))
      && (_mm256_testz_si256(v.b6, v.b6) || _mm256_testz_si256(inv.b6, inv.b6))
      && (_mm256_testz_si256(v.b7, v.b7) || _mm256_testz_si256(inv.b7, inv.b7));
}

/**
 * @brief Returns true if the bitslice implementation of the inverse Rijndael
 * S-box is correct. Used for testing.
 */
static halfloop_result_t test_bitslice_inv_sbox(void) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  for (u32 i = 0; i < 0x100; i++) {
    eightbits out = bitslice_inv_sub_bytes(get_eightbits((u8)i));
    RETURN_IF(!check_equal_bits(out), HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(get_lower_bits(out) != inv_SBOX[i], HALFLOOP_INTERNAL_ERROR);
  }
error:
  return err;
}

/**
 * @brief Returns true if the bitslice implementation of the inverse rotate rows
 * operations is correct. Used for testing.
 */
static halfloop_result_t test_bitslice_inv_rotate_rows(void) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  for (u32 i = 0; i < 0x100; i++) {
    eightbits in = get_eightbits((u8)i);
    eightbits out4 = bitslice_inv_rotate_rows_4(in);
    eightbits out6 = bitslice_inv_rotate_rows_6(in);
    RETURN_IF(!check_equal_bits(out4), HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(!check_equal_bits(out6), HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(get_lower_bits(out4) != (u8)((i >> 4) | (i << 4)),
        HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(get_lower_bits(out6) != (u8)((i >> 6) | (i << 2)),
        HALFLOOP_INTERNAL_ERROR);
  }
error:
  return err;
}

/**
 * @brief Returns true if the bitslice implementation of the HALFLOOP-24 mix
 * columns operation is correct. Used for testing.
 */
static halfloop_result_t test_bitslice_inv_mix_columns(void) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  for (u32 i = 0; i < 0x1000000; i++) {
    twentyfourbits in = {
      .msb = get_eightbits((u8)(i >> 16)),
      .mid = get_eightbits((u8)(i >> 8)),
      .lsb = get_eightbits((u8)i)
    };
    twentyfourbits out = bitslice_inv_mix_columns(in);
    RETURN_IF(!check_equal_bits(out.msb), HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(!check_equal_bits(out.mid), HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(!check_equal_bits(out.lsb), HALFLOOP_INTERNAL_ERROR);
    u32 res = (get_lower_bits(out.msb) << 16) | (get_lower_bits(out.mid) << 8)
        | get_lower_bits(out.lsb);
    RETURN_IF(res != inv_mix_columns(i), HALFLOOP_INTERNAL_ERROR);
  }
error:
  return err;
}

halfloop_result_t halfloop_benchmark_bitslice(void) {
  u32 pt = 0;
  u32 ct0 = 0;
  u32 ct1 = 0;
  u64 tweak = 0;
  hlkey key = {0};
  u32 rk[11];
  u32 *found = NULL;
  int num_found = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  RETURN_ON_ERROR(random_bytes(&pt, sizeof(u32)));
  RETURN_ON_ERROR(random_bytes(&tweak, sizeof(u64)));
  RETURN_ON_ERROR(random_bytes(&key, sizeof(hlkey)));
  pt &= 0xffffff;
  RETURN_ON_ERROR(halfloop_encrypt(pt, key, tweak, &ct0));
  RETURN_ON_ERROR(halfloop_encrypt(pt, key, tweak ^ (1 << 30), &ct1));
  RETURN_ON_ERROR(key_schedule(rk, key, 0));
  u32 rk56 = ((rk[5] & 0xff) << 24) | rk[6];

  print_message("Benchmarking CPU bitslice algorithm.", WHITE);
  hltimer timer;
  TIMER_START(&timer);
  RETURN_ON_ERROR(halfloop_bitslice(
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
  double elapsed = timer_elapsed(timer);
  print_message(
      "Number of keys found during bitslice test: %d.",
      WHITE,
      num_found);

  bool ok = false;
  for (int i = 0; i < num_found && !ok; i++) {
    if (rk56 == found[i]) {
      ok = true;
    }
  }
  RETURN_IF(!ok, HALFLOOP_INTERNAL_ERROR);
#ifdef _WIN32
  print_message(
      "Benchmark took %.2f seconds: %lld keys/second.",
      WHITE,
      elapsed,
      (u64)(0x100000000ULL / elapsed));
#else /* _WIN32 */
  setlocale(LC_NUMERIC, "");
  print_message(
      "Benchmark took %.2f seconds: %'lld keys/second.",
      WHITE,
      elapsed,
      (u64)(0x100000000ULL / elapsed));
#endif /* _WIN32 */
error:
  if (err != HALFLOOP_SUCCESS) {
    print_message(
        "Bitslice benchmark failed. PT=%06x tweak=%016" PRIx64 " Key=%016"
            PRIx64 "%016" PRIx64,
        RED,
        pt,
        tweak,
        key.hi,
        key.lo);
  }
  free(found);
  return err;
}

halfloop_result_t test_halfloop_bitslice(void) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  print_message("Testing bitslice subroutines.", WHITE);
  RETURN_ON_ERROR(test_bitslice_inv_sbox());
  RETURN_ON_ERROR(test_bitslice_inv_rotate_rows());
  RETURN_ON_ERROR(test_bitslice_inv_mix_columns());
error:
  if (err != HALFLOOP_SUCCESS) {
    print_message("Bitslice subroutine test failed.", RED);
  }
  return err;
}
