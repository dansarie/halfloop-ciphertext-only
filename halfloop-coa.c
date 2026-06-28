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

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif
#define _GNU_SOURCE
#include <immintrin.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "getopt.h"
#include "halfloop-bitslice.h"
#include "halfloop-common.h"
#include "halfloop-cuda.h"

/** Parsed command line options. */
struct options {
  bool benchmark;   /**< Set by the -b flag to run benchmarks. */
  bool listdevs;    /**< Set by the -l flag to list CUDA devices. */
  bool verbose;     /**< Set by the -v flag to increase verbosity. */
  bool profile;     /**< Set by the -p flag to run a limited search for
                         profiling purposes. */
  bool brute_force; /**< If true, the remaining 48-bits of each key will be
                         found by brute-force search. */
  halfloop_algorithm_t algorithm; /**< Which attack algorithm to use for the
                                       search. */
  char *filename;   /**< Path of file that contains ciphertexts and tweaks. */
  u32 fixed_bits;   /**< The fixed bits of the key. Filled from MSB down to
                         LSB. The eight most significant bits represents one
                         byte of rk9. The remaining bits represent rk10. */
  u32 tau1;         /**< Threshold value 1 set by the -t flag. */
  u32 tau2;         /**< Threshold value 2 set by the -u flag. */
  u32 blockmul;     /**< Number of CUDA blocks per multiprocessor. Set by -m. */
  int devices[MAX_DEVICES]; /**< List of CUDA device ids to use. Set by the -d
                                 flag. */
  int num_devices;  /**< Number of CUDA device ids in devices. Set to 0 to use
                         all available devices. */
  int num_fixed;    /**< Number of fixed bits in fixed_bits. */
  double p_ct;      /**< The probability that two randomly selected callsigns
                         will differ only in the least significant byte. Set by
                         the -c flag. */
  double p_success; /**< Target probability of success. Set by the -s flag. */
};

/** Used by qsort in read_input_tuples. */
static int compare_tuples(const void *tuple1, const void *tuple2) {
  tuple_t *t1 = (tuple_t*)tuple1;
  tuple_t *t2 = (tuple_t*)tuple2;
  if (t1->tweak == t2->tweak) {
    if (t1->ct < t2->ct) {
      return -1;
    }
    if (t1->ct > t2->ct) {
      return 1;
    }
    return 0;
  }
  if (t1->tweak < t2->tweak) {
    return -1;
  }
  if (t1->tweak > t2->tweak) {
    return 1;
  }
  return 0;
}

#ifdef CUDA_ENABLED
/** Used by qsort to sort candidate keys. */
static int compare_candidates(const void *candidate1, const void *candidate2) {
  candidate_key_t *c1 = (candidate_key_t*)candidate1;
  candidate_key_t *c2 = (candidate_key_t*)candidate2;
  if (c1->pairs > c2->pairs) {
    return -1;
  }
  if (c1->pairs < c2->pairs) {
    return 1;
  }
  if (c1->rk7 < c2->rk7) {
    return -1;
  }
  if (c1->rk7 > c2->rk7) {
    return 1;
  }
  if (c1->rk8 < c2->rk8) {
    return -1;
  }
  if (c1->rk8 > c2->rk8) {
    return 1;
  }
  if (c1->rk9 < c2->rk9) {
    return -1;
  }
  if (c1->rk9 > c2->rk9) {
    return 1;
  }
  if (c1->rk10 < c2->rk10) {
    return -1;
  }
  if (c1->rk10 > c2->rk10) {
    return 1;
  }
  return 0;
}
#endif /* CUDA_ENABLED */

/**
 * @brief Reads ciphertext-tweak tuples from a text file.
 *
 * @param fname the file name of the input file.
 * @param tuples return pointer. Will contain a list of tuples on return.
 * @param num_tuples will contain the number of items in the tuples list on
 * return.
 * @return HALFLOOP_SUCCESS on success.
 */
static halfloop_result_t read_input_pairs(
    const char *fname,
    tuple_t **tuples,
    int *num_tuples) {
  CHECK_BAD_ARGUMENT(fname == NULL);
  CHECK_BAD_ARGUMENT(tuples == NULL);
  CHECK_BAD_ARGUMENT(num_tuples == NULL);

  FILE *fp = NULL;
  *tuples = NULL;
  *num_tuples = 0;
  int num_alloc = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  fp = fopen(fname, "r");
  RETURN_IF(fp == NULL, HALFLOOP_FILE_ERROR);

  while (!feof(fp)) {
    tuple_t tuple;
    if (fscanf(fp, "%06x %016" PRIx64 "\n", &tuple.ct, &tuple.tweak) == 2) {
      if (*num_tuples == num_alloc) {
        num_alloc += 1000;
        tuple_t *tmp = realloc(*tuples, sizeof(tuple_t) * num_alloc);
        RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
        *tuples = tmp;
      }
      (*tuples)[*num_tuples] = tuple;
      *num_tuples += 1;
    } else {
      int c;
      while ((c = fgetc(fp)) != '\n' && c != EOF) {
        /* Empty. */
      }
    }
  }

  tuple_t *tmp = realloc(*tuples, sizeof(tuple_t) * *num_tuples);
  if (tmp != NULL) {
    *tuples = tmp;
  }

  /* Remove duplicates. */
  qsort(*tuples, *num_tuples, sizeof(tuple_t), compare_tuples);
  for (int i = 1; i < *num_tuples; i++) {
    if (compare_tuples(*tuples + i - 1, *tuples + i) == 0) {
      *num_tuples -= 1;
      memmove(*tuples + i, *tuples + i + 1,
          sizeof(tuple_t) * (*num_tuples - i));
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*tuples);
    *num_tuples = 0;
  }
  if (fp != NULL) {
    fclose(fp);
  }
  return err;
}

/**
 * Calculates the probability of success of the attack.
 * @param tau threshold value.
 * @param num_pairs number of available pairs.
 * @param num_counters number of counters/bins in the attack.
 * @param keys_per_match average number of keys indicated by a matching pair.
 * @param p_ct probability that two randomly selected ciphertexts will differ
 * only in the last byte.
 * @param p_success output pointer for calculated probability of success.
 */
static halfloop_result_t calc_p_success(
    u32 tau,
    u32 num_pairs,
    u32 num_counters,
    double keys_per_match,
    double p_ct,
    double *p_success) {
  CHECK_BAD_ARGUMENT(num_pairs == 0);
  CHECK_BAD_ARGUMENT(num_counters == 0);
  CHECK_BAD_ARGUMENT(p_ct <= 0.0);
  CHECK_BAD_ARGUMENT(p_ct > 1.0);
  CHECK_BAD_ARGUMENT(p_success == NULL);
  double p_wrong = keys_per_match / num_counters; /* Wrong pair. */
  double p_none = 1.0 - p_ct - p_wrong; /* No hit */
  *p_success = 1.0;
  for (int j = 0; j < (int)tau; j++) {
    for (int k = 0; k <= j; k++) {
      /* Log of multinomial coefficient. */
      double v = lgamma(num_pairs + 1);
      v -= lgamma(k + 1);
      v -= lgamma(j - k + 1);
      v -= lgamma(num_pairs - j + 1);
      /* Log probability of k wrong-pair hits. */
      if (k != 0) {
        v += k * log(p_wrong);
      }
      /* Log probability of j - k right-pair hits. */
      if (j - k != 0) {
        v += (j - k) * log(p_ct);
      }
      /* Log probability of num_pairs - j pairs without hit. */
      if (num_pairs - j != 0) {
        v += (num_pairs - j) * log(p_none);
      }
      *p_success -= exp(v);
    }
  }
  return HALFLOOP_SUCCESS;
}

/**
 * Calculate optimal tau values for a given number of ciphertext pairs and
 * target probability of success.
 * @param num_pairs number of ciphertext pairs.
 * @param num_counters number of counters in algorithm.
 * @param keys_per_match average number of keys per match.
 * @param p_ct probability of a ciphertext pair having the correct difference.
 * @param p_success target probability of success.
 * @param tau return pointer for the optimal tau value.
 * @param p_tau return pointer for calculated probability of success.
 */
static halfloop_result_t get_tau(
    u32 num_pairs,
    u32 num_counters,
    double keys_per_match,
    double p_ct,
    double p_success,
    u32 *tau,
    double *p_tau) {
  CHECK_BAD_ARGUMENT(num_pairs < 1);
  CHECK_BAD_ARGUMENT(num_counters < 1);
  CHECK_BAD_ARGUMENT(keys_per_match <= 0.0);
  CHECK_BAD_ARGUMENT(p_ct <= 0.0);
  CHECK_BAD_ARGUMENT(p_ct > 1.0);
  CHECK_BAD_ARGUMENT(p_success <= 0.0);
  CHECK_BAD_ARGUMENT(p_success > 1.0);
  CHECK_BAD_ARGUMENT(tau == NULL);
  CHECK_BAD_ARGUMENT(p_tau == NULL);
  halfloop_result_t err = HALFLOOP_SUCCESS;
  *p_tau = 0.0;
  for (*tau = 255; *tau > 0 && *p_tau < p_success; (*tau)--) {
    RETURN_ON_ERROR(calc_p_success(
        *tau,
        num_pairs,
        num_counters,
        keys_per_match,
        p_ct,
        p_tau));
  }
  *tau += 1;
error:
  return err;
}

/**
 * Implementation of algorithm 2 for recovering rk8, rk9, rk10, and one byte
 * from LL^-1(rk7).
 * @param ct a list of ciphertext-tweak tuples.
 * @param num_ct number of ciphertext tuples in ct.
 * @param pairs a list of pairs with the required tweak difference, where the
 * indexes correspond to indexes in the ct list.
 * @param num_pairs number of pairs in the pairs list.
 * @param tau1 threshold value 1.
 * @param tau2 threshold value 2.
 * @param fixed_bits User-set job bits. Filled from MSB down to LSB. The eight
 * most significant bits represent one byte of rk9. The remaining bits represent
 * rk10.
 * @param num_fixed Number of valid bits in fixed_bits.
 * @param verbose set to true to print more status updates to stdout.
 * @param candidates return variable for identified candidate keys. Each
 * candidate is stored as (LL^-1(rk7) << 48) | (rk8 << 24) | rk9. Will be set to
 * NULL if no candidates are found. The caller must return the array using free.
 * @param num_candidates return variable for the number of candidate keys in the
 * candidates list.
 */
static halfloop_result_t ct_attack1(
    const tuple_t *restrict ct,
    int num_ct,
    const tuple_pair_t *restrict pairs,
    int num_pairs,
    u32 tau1,
    u32 tau2,
    u32 fixed_bits,
    int num_fixed,
    bool verbose,
    candidate_key_t **restrict candidates,
    u32 *restrict num_candidates) {
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(num_ct < 2);
  CHECK_BAD_ARGUMENT(pairs == NULL);
  CHECK_BAD_ARGUMENT(num_pairs < 1);
  CHECK_BAD_ARGUMENT(tau1 < 1);
  CHECK_BAD_ARGUMENT(tau2 < 1);
  CHECK_BAD_ARGUMENT(num_fixed < 0 || num_fixed > 32);
  CHECK_BAD_ARGUMENT(candidates == NULL);
  CHECK_BAD_ARGUMENT(num_candidates == NULL);

  u16 ddt[256 * 256] = {0}; /* Lookup table for S-box outputs. */
  u32 *v8 = NULL;       /* Array for precomputed v8 states. */
  u32 *x9 = NULL;       /* Array for precomputed x9 states. */
  u8 *count = 0;        /* Counter for candidate rk8 values. */
  *candidates = NULL;
  *num_candidates = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  v8 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(v8 == NULL, HALFLOOP_MEMORY_ERROR);
  x9 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(x9 == NULL, HALFLOOP_MEMORY_ERROR);
  count = malloc(sizeof(u8) * 0x10000);
  RETURN_IF(count == NULL, HALFLOOP_MEMORY_ERROR);

  /* Initialize lookup tables. */
  RETURN_ON_ERROR(init_ddt(ddt));

  for (u64 bits = 0; bits < 1ULL << (32 - num_fixed); bits++) {
    u32 rk10n = ((bits << num_fixed) | fixed_bits) & 0xffffff;
    u32 rk9n2 = (u32)(((bits << num_fixed) | fixed_bits) >> 24);

    if (verbose) {
      print_message("Testing rk9_2: %02x rk10: %06x.", WHITE, rk9n2, rk10n);
    }

    /* Precalculate x9 for all ciphertext tuples. */
    for (int i = 0; i < num_ct; i++) {
      /* Generate rk10 from normalized rk10 and LSB of rk9. */
      u32 rk10;
      RETURN_ON_ERROR(halfloop_round10_tweak(ct[i].tweak, (u8)rk9n2, &rk10));
      rk10 ^= rk10n;
      x9[i] = inv_sub_bytes(inv_rotate_rows(ct[i].ct ^ rk10));
    }

    for (u32 rk9n01 = 0; rk9n01 < 0x10000; rk9n01++) {
      u32 rk9n = (rk9n01 << 8) | rk9n2; /* Normalized rk9. */
      /* Precalculate v8 for all ciphertext tuples. */
      for (int i = 0; i < num_ct; i++) {
        u32 rk9;
        RETURN_ON_ERROR(halfloop_round_tweak(ct[i].tweak, 9, &rk9));
        rk9 ^= rk9n;
        u32 x8 = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9[i] ^ rk9)));
        u32 tw8;
        RETURN_ON_ERROR(halfloop_round_tweak(ct[i].tweak, 8, &tw8));
        v8[i] = inv_rotate_rows(inv_mix_columns(x8 ^ tw8));
      }
      /* Kappa is the MSB of LL^-1(rk8), rk8 normalized. */
      for (u32 kappa = 0; kappa < 256; kappa++) {
        memset(count, 0, sizeof(u8) * 0x10000);
        /* Iterate over all pairs. */
        bool flag = false;
        for (const tuple_pair_t *p = pairs; p < pairs + num_pairs; p++) {
          /* Calculate LL^-1(x8 ^ tw), where tw is the tweak for round 8. */
          u32 tw8a;
          u32 tw8b;
          RETURN_ON_ERROR(halfloop_round_tweak(ct[p->a].tweak, 8, &tw8a));
          RETURN_ON_ERROR(halfloop_round_tweak(ct[p->b].tweak, 8, &tw8b));
          u32 x72a = inv_SBOX[(v8[p->a] ^ kappa) & 0xff];
          u32 x72b = inv_SBOX[(v8[p->b] ^ kappa) & 0xff];
          u32 delta = tw8a ^ tw8b;
          u32 gamma = x72a ^ x72b;
          if (ddt[gamma * 256 + delta] == 0xffff) {
            continue;
          }
          RETURN_ON_ERROR(halfloop_round_tweak(ct[p->a].tweak, 7, &tw8a));
          RETURN_ON_ERROR(halfloop_round_tweak(ct[p->b].tweak, 7, &tw8b));
          u32 dx70 = ffmul_table_9[x72a ^ x72b];
          u32 dx71 = FFMUL2(x72a ^ x72b) ^ delta;
          u32 dv8 = v8[p->a] ^ v8[p->b];
          u32 dout0 = dv8 >> 16;
          u32 dout1 = (dv8 >> 8) & 0xff;
          u32 idx0 = (dout0 << 8) | dx70;
          u32 idx1 = (dout1 << 8) | dx71;
          u32 diffs[4] = {0, dout0 << 8, dout1, (dout0 << 8) | dout1};
          u32 ddt0 = ddt[idx0];
          u32 ddt1 = ddt[idx1];
          if (ddt0 == 0xffff || ddt1 == 0xffff) {
            continue;
          }
          do {
            u32 k0 = ((v8[p->a] >> 16) ^ (ddt0 & 0xff)) << 8;
            ddt0 >>= 8;
            u32 ddt1x = ddt1;
            do {
              u32 key = k0 | (((v8[p->a] >> 8) & 0xff) ^ (ddt1x & 0xff));
              ddt1x >>= 8;
              for (int i = 0; i < 4; i++) {
                u32 k = key ^ diffs[i];
                count[k] += 1;
                if (count[k] >= tau1) {
                  /* Set flag to indicate that a candidate has been found. */
                  flag = true;
                }
              }
            } while (ddt1x != 0);
          } while (ddt0 != 0);
        }
        /* The flag variable ensures that this loop only runs if there is at
         * least one potential candidate in the array. */
        for (int i = 0; flag && i < 0x10000; i++) {
          if (count[i] >= tau1) {
            u32 rk8 = (i << 8) | kappa;
            rk8 = mix_columns(rotate_rows(rk8));
            u8 rk7 = 0;
            u32 keycount = 0;
            RETURN_ON_ERROR(validate_rk8(
                ct,
                v8,
                num_ct,
                pairs,
                num_pairs,
                rk8,
                &rk7,
                &keycount));
            if ((int)keycount >= tau2) {
              print_message(
                  "Found candidate key: %02x %06x %06x %06x "
                  "(%d matching pairs)",
                  GREEN,
                  rk7,
                  rk8,
                  rk9n,
                  rk10n,
                  keycount);
              *num_candidates += 1;
              candidate_key_t *tmp = realloc(*candidates,
                  sizeof(u64) * *num_candidates);
              RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
              *candidates = tmp;
              candidate_key_t *k = *candidates + *num_candidates - 1;
              k->pairs = keycount;
              k->rk7 = rk7;
              k->rk8 = rk8;
              k->rk9 = rk9n;
              k->rk10 = rk10n;
            }
          }
        }
      }
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*candidates);
    *num_candidates = 0;
  }
  free(v8);
  free(x9);
  free(count);
  return err;
}

/**
 * A faster algorithm for recovering rk8, rk9, rk10, and one byte from
 * LL^-1(rk7).
 * @param ct a list of ciphertext-tweak tuples.
 * @param num_ct number of ciphertext tuples in ct.
 * @param pairs a list of pairs with the required tweak difference, where the
 * indexes correspond to indexes in the ct list.
 * @param num_pairs number of pairs in the pairs list.
 * @param tau1 threshold value 1.
 * @param tau2 threshold value 2.
 * @param fixed_bits User-set job bits. Filled from MSB down to LSB. The eight
 * most significant bits represent one byte of rk9. The remaining bits represent
 * rk10.
 * @param num_fixed Number of valid bits in fixed_bits.
 * @param verbose set to true to print more status updates to stdout.
 * @param candidates return variable for identified candidate keys. Each
 * candidate is stored as (LL^-1(rk7) << 48) | (rk8 << 24) | rk9. Will be set to
 * NULL if no candidates are found. The caller must return the array using free.
 * @param num_candidates return variable for the number of candidate keys in the
 * candidates list.
 */
static halfloop_result_t ct_attack2(
    const tuple_t *restrict ct,
    int num_ct,
    const tuple_pair_t *restrict pairs,
    int num_pairs,
    u32 tau1,
    u32 tau2,
    u32 fixed_bits,
    int num_fixed,
    bool verbose,
    candidate_key_t **restrict candidates,
    u32 *restrict num_candidates) {
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(num_ct < 2);
  CHECK_BAD_ARGUMENT(pairs == NULL);
  CHECK_BAD_ARGUMENT(num_pairs < 1);
  CHECK_BAD_ARGUMENT(tau1 < 1);
  CHECK_BAD_ARGUMENT(tau2 < 1);
  CHECK_BAD_ARGUMENT(num_fixed < 0 || num_fixed > 32);
  CHECK_BAD_ARGUMENT(candidates == NULL);
  CHECK_BAD_ARGUMENT(num_candidates == NULL);

  u16 ddt[256 * 256] = {0};          /* Lookup table for S-box outputs.  */
  __m256i y0_table[256] = {0};       /* Lookup table for y0 differences. */
  __m256i y1_table[256 * 255] = {0}; /* Lookup table for y1 differences. */
  __m256i y2_table[256] = {0};       /* Lookup table for y2 differences. */
  u32 *v8 = NULL;       /* Array for precomputed v8 states. */
  u32 *x9 = NULL;       /* Array for precomputed x9 states. */
  u8 *count = 0;        /* Counter for candidate rk8 values. */
  *candidates = NULL;
  *num_candidates = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  v8 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(v8 == NULL, HALFLOOP_MEMORY_ERROR);
  x9 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(x9 == NULL, HALFLOOP_MEMORY_ERROR);
  count = malloc(sizeof(u8) * 0x1000000);
  RETURN_IF(count == NULL, HALFLOOP_MEMORY_ERROR);

  /* Initialize lookup tables. */
  RETURN_ON_ERROR(init_ddt(ddt));
  for (int ydelta = 0; ydelta < 256; ydelta++) {
    RETURN_ON_ERROR(init_y0_lut((u8)ydelta, y0_table + ydelta));
    RETURN_ON_ERROR(init_y2_lut((u8)ydelta, y2_table + ydelta));
  }
  for (int delta = 1; delta < 256; delta++) {
    for (int y1delta = 0; y1delta < 256; y1delta++) {
      RETURN_ON_ERROR(init_y1_lut(
          (u8)y1delta,
          (u8)delta,
          y1_table + (delta - 1) * 256 + y1delta));
    }
  }


  for (u64 bits = 0; bits < 1ULL << (32 - num_fixed); bits++) {
    u32 rk10n = ((bits << num_fixed)| fixed_bits) & 0xffffff;
    u32 rk9n2 = (u32)(((bits << num_fixed) | fixed_bits) >> 24);

    if (verbose) {
      print_message("Testing rk9_2: %02x rk10: %06x.", WHITE, rk9n2, rk10n);
    }

    /* Precalculate x9 for all tuples. */
    for (int i = 0; i < num_ct; i++) {
      /* Generate rk10 from normalized rk10 and LSB of rk9. */
      u32 rk10;
      RETURN_ON_ERROR(halfloop_round10_tweak(ct[i].tweak, (u8)rk9n2, &rk10));
      rk10 ^= rk10n;
      x9[i] = inv_sub_bytes(inv_rotate_rows(ct[i].ct ^ rk10));
    }

    for (u32 rk9n01 = 0; rk9n01 < 0x10000; rk9n01++) {
      u32 rk9n = (rk9n01 << 8) | rk9n2; /* Normalized rk9. */
      /* Precalculate v8 for all tuples. */
      for (int i = 0; i < num_ct; i++) {
        /* Generate rk9 from normalized rk9. */
        u32 rk9;
        RETURN_ON_ERROR(halfloop_round_tweak(ct[i].tweak, 9, &rk9));
        rk9 ^= rk9n;
        /* Calculate and store v8. */
        u32 x8 = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9[i] ^ rk9)));
        u32 tw8;
        RETURN_ON_ERROR(halfloop_round_tweak(ct[i].tweak, 8, &tw8));
        v8[i] = inv_rotate_rows(inv_mix_columns(x8 ^ tw8));
      }
      memset(count, 0, sizeof(u8) * 0x1000000);
      /* Iterate over all pairs. */
      bool flag = false;
      for (const tuple_pair_t *p = pairs; p < pairs + num_pairs; p++) {
        u32 delta = ((ct[p->a].tweak ^ ct[p->b].tweak) >> 40) & 0xff;
        u32 dy7 = v8[p->a] ^ v8[p->b]; /* y7 difference. */
        __m256i w0 = y0_table[dy7 >> 16];
        __m256i w1 = y1_table[((delta - 1) * 256) | ((dy7 >> 8) & 0xff)];
        __m256i w2 = y2_table[dy7 & 0xff];
        __m256i w = _mm256_and_si256(_mm256_and_si256(w0, w1), w2); /* Possible x7_2 differences. */
        u64 ww[4] = {0};
        _mm256_storeu_si256((__m256i*)ww, w);
        /* Iterate over candidate values for y7. */
        for (int i = 0; i < 4; i++) {
          while (ww[i] != 0) {
            u32 bit = (u32)CTZL(ww[i]);
            ww[i] ^= 1ULL << bit;
            u32 dx72 = bit + i * 64; /* x7_2 difference. */
            u32 dx70 = ffmul_table_9[dx72];
            u32 dx71 = FFMUL2(dx72) ^ delta;
            u32 ddt0 = ddt[((dy7 & 0xff0000) >> 8) | dx70];
            u32 ddt1 = ddt[ (dy7 & 0x00ff00)       | dx71];
            u32 ddt2 = ddt[((dy7 & 0x0000ff) << 8) | dx72];
            if (ddt0 == 0xffff || ddt1 == 0xffff || ddt2 == 0xffff) {
              continue;
            }
            /* Each value in the DDT LUT represents two output values with a
             * difference that is implicit from the index. The diffs array is
             * used to generate all eight combinations. */
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
                  u32 key = (y701 | y72a) ^ v8[p->a];
                  for (int j = 0; j < 8; j++) {
                    u32 k = key ^ diffs[j];
                    count[k] += 1;
                    if (count[k] >= tau1) {
                      /* Set flag to indicate that a candidate has been found.*/
                      flag = true;
                    }
                  }
                } while (ddt1x != 0);
              } while (ddt0x != 0);
            } while (ddt2 != 0);
          }
        }
      }
      /* The flag variable ensures that this loop only runs if there is at least
       * one potential candidate in the array. */
      for (int i = 0; flag && i < 0x1000000; i++) {
        if (count[i] >= tau1) {
          u8 rk7 = 0;
          u32 keycount = 0;
          u32 rk8 = mix_columns(rotate_rows(i));
          RETURN_ON_ERROR(validate_rk8(
              ct,
              v8,
              num_ct,
              pairs,
              num_pairs,
              rk8,
              &rk7,
              &keycount));
          if ((int)keycount >= tau2) {
            print_message(
                "Found candidate key: %02x %06x %06x %06x "
                "(%d matching pairs)",
                GREEN,
                rk7,
                rk8,
                rk9n,
                rk10n,
                keycount);
            *num_candidates += 1;
            candidate_key_t *tmp = realloc(*candidates,
                sizeof(u64) * *num_candidates);
            RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
            *candidates = tmp;
            candidate_key_t *k = *candidates + *num_candidates - 1;
            k->pairs = keycount;
            k->rk7 = rk7;
            k->rk8 = rk8;
            k->rk9 = rk9n;
            k->rk10 = rk10n;
          }
        }
      }
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*candidates);
    *num_candidates = 0;
  }
  free(v8);
  free(x9);
  free(count);
  return err;
}

/**
 * A an even faster algorithm for recovering rk8, rk9, rk10, and one byte from
 * LL^-1(rk7).
 * @param ct a list of ciphertext-tweak tuples.
 * @param num_ct number of ciphertext-tweak tuples in ct.
 * @param pairs a list of pairs with the required tweak difference, where the
 * indexes correspond to indexes in the ct list.
 * @param num_pairs number of pairs in the pairs list.
 * @param tau1 threshold value in first part of algorithm.
 * @param tau2 threshold value in second part of algorithm.
 * @param fixed_bits User-set job bits. Filled from MSB down to LSB. The eight
 * most significant bits represent one byte of rk9. The remaining bits represent
 * rk10.
 * @param num_fixed Number of valid bits in fixed_bits.
 * @param verbose set to true to print more status updates to stdout.
 * @param candidates return variable for identified candidate keys. Each
 * candidate is stored as (LL^-1(rk7) << 48) | (rk8 << 24) | rk9. Will be set to
 * NULL if no candidates are found. The caller must return the array using free.
 * @param num_candidates return variable for the number of candidate keys in the
 * candidates list.
 */
static halfloop_result_t ct_attack3(
    const tuple_t *restrict ct,
    int num_ct,
    const tuple_pair_t *restrict pairs,
    int num_pairs,
    u32 tau1,
    u32 tau2,
    u32 fixed_bits,
    int num_fixed,
    bool verbose,
    candidate_key_t **restrict candidates,
    u32 *restrict num_candidates) {
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(num_ct < 2);
  CHECK_BAD_ARGUMENT(pairs == NULL);
  CHECK_BAD_ARGUMENT(num_pairs < 1);
  CHECK_BAD_ARGUMENT(tau1 < 1);
  CHECK_BAD_ARGUMENT(tau2 < 1);
  CHECK_BAD_ARGUMENT(num_fixed < 0 || num_fixed > 32);
  CHECK_BAD_ARGUMENT(candidates == NULL);
  CHECK_BAD_ARGUMENT(num_candidates == NULL);

  u16 ddt[256 * 256] = {0};       /* Lookup table for S-box outputs.  */
  __m256i y0_table[256] = {0};       /* Lookup table for y0 differences. */
  __m256i y1_table[256 * 255] = {0}; /* Lookup table for y1 differences. */
  __m256i y2_table[256] = {0};       /* Lookup table for y2 differences. */
  u8 count[0x10000] = {0};   /* Counters for partial candidate rk8 values. */
  u16 keys[512] = {0};       /* Found candidate keys in first search. */
  u32 nkeys = 0;             /* Number of keys in keys array. */
  u32 *v8 = NULL;            /* Array for precomputed v8 states. */
  u32 *x9 = NULL;            /* Array for precomputed x9 states. */
  u8 *x70 = NULL;            /* Array for precomputed x70 states. */
  u8 *x72 = NULL;            /* Array for precomputed x70 states. */
  *candidates = NULL;
  *num_candidates = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  v8 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(v8 == NULL, HALFLOOP_MEMORY_ERROR);
  x9 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(x9 == NULL, HALFLOOP_MEMORY_ERROR);
  x70 = malloc(sizeof(u8) * num_ct * 256);
  RETURN_IF(x70 == NULL, HALFLOOP_MEMORY_ERROR);
  x72 = malloc(sizeof(u8) * num_ct * 256);
  RETURN_IF(x72 == NULL, HALFLOOP_MEMORY_ERROR);

  /* Initialize lookup tables. */
  RETURN_ON_ERROR(init_ddt(ddt));
  for (int ydelta = 0; ydelta < 256; ydelta++) {
    RETURN_ON_ERROR(init_y0_lut((u8)ydelta, y0_table + ydelta));
    RETURN_ON_ERROR(init_y2_lut((u8)ydelta, y2_table + ydelta));
  }
  for (int delta = 1; delta < 256; delta++) {
    for (int y1delta = 0; y1delta < 256; y1delta++) {
      RETURN_ON_ERROR(init_y1_lut(
          (u8)y1delta,
          (u8)delta,
          y1_table + (delta - 1) * 256 + y1delta));
    }
  }

  for (u64 bits = 0; bits < 1ULL << (32 - num_fixed); bits++) {
    u32 rk10n = ((bits << num_fixed) | fixed_bits) & 0xffffff;
    u32 rk9n2 = (u32)(((bits << num_fixed) | fixed_bits) >> 24);

    if (verbose) {
      print_message("Testing rk9_2: %02x rk10: %06x.", WHITE, rk9n2, rk10n);
    }

    /* Precalculate x9 for all tuples. */
    for (int i = 0; i < num_ct; i++) {
      /* Generate rk10 from normalized rk10 and LSB of rk9. */
      u32 rk10;
      RETURN_ON_ERROR(halfloop_round10_tweak(ct[i].tweak, (u8)rk9n2, &rk10));
      rk10 ^= rk10n;
      x9[i] = inv_sub_bytes(inv_rotate_rows(ct[i].ct ^ rk10));
    }

    /* Loop over the remaining 16 bits of rk9. */
    for (u32 rk9n01 = 0; rk9n01 < 0x10000; rk9n01++) {
      u32 rk9n = (rk9n01 << 8) | rk9n2; /* Normalized rk9. */
      /* Precalculate v8 for all tuples. */
      for (int i = 0; i < num_ct; i++) {
        /* Generate rk9 from normalized rk9. */
        u32 rk9;
        RETURN_ON_ERROR(halfloop_round_tweak(ct[i].tweak, 9, &rk9));
        rk9 ^= rk9n;
        /* Calculate and store v8. */
        u32 x8 = inv_sub_bytes(inv_rotate_rows(inv_mix_columns(x9[i] ^ rk9)));
        u32 tw8;
        RETURN_ON_ERROR(halfloop_round_tweak(ct[i].tweak, 8, &tw8));
        v8[i] = inv_rotate_rows(inv_mix_columns(x8 ^ tw8));
        for (int k = 0; k < 256; k++) {
          x70[k * num_ct + i] = inv_SBOX[(v8[i] >> 16) ^ k];
          x72[k * num_ct + i] = inv_SBOX[(v8[i] & 0xff) ^ k];
        }
      }
      /* Reset counters and counter hit flags. */
      memset(count, 0, sizeof(u8) * 0x10000);
      nkeys = 0;
      /* Iterate over all pairs. */
      for (const tuple_pair_t *p = pairs; p < pairs + num_pairs; p++) {
        u32 delta = ((ct[p->a].tweak ^ ct[p->b].tweak) >> 40) & 0xff;
        u32 dy7 = v8[p->a] ^ v8[p->b]; /* y7 difference. */
        __m256i w0 = y0_table[dy7 >> 16];
        __m256i w1 = y1_table[((delta - 1) * 256) | ((dy7 >> 8) & 0xff)];
        __m256i w2 = y2_table[dy7 & 0xff];
        __m256i w = _mm256_and_si256(_mm256_and_si256(w0, w1), w2); /* Possible x7_2 differences. */
        u64 ww[4] = {0};
        _mm256_storeu_si256((__m256i*)ww, w);
        /* Iterate over candidate values for y7. */
        for (int i = 0; i < 4; i++) {
          while (ww[i] != 0) {
            u32 bit = (u32)CTZL(ww[i]);
            ww[i] ^= 1ULL << bit;
            u32 dx72 = bit + i * 64; /* x7_2 difference. */
            u32 dx70 = ffmul_table_9[dx72];
            u32 dout2 = dy7 & 0xff;
            u32 dout0 = dy7 >> 16;
            u32 idx2 = (dout2 << 8) | dx72;
            u32 idx0 = (dout0 << 8) | dx70;
            if (ddt[idx2] == 0xffff || ddt[idx0] == 0xffff) {
              continue;
            }
            u32 diffs[4] = {0, dout2 << 8, dout0, (dout2 << 8) | dout0};
            u32 ddt2 = ddt[idx2];
            do {
              u32 y72a = ddt2 & 0xff;
              u32 k2 = (y72a ^ (v8[p->a] & 0xff)) << 8; /* LSB of LL^-1(rk8). */
              u32 ddt0 = ddt[idx0];
              do {
                u32 y70 = ddt0 & 0xff;
                /* Calculate candidate key. */
                u32 k0 = y70 ^ (v8[p->a] >> 16); /* MSB of LL^-1(rk8). */
                u32 key = k2 | k0;
                for (int j = 0; j < 4; j++) {
                  u32 k = key ^ diffs[j];
                  count[k] += 1;
                  /* Set flag if counter for the partial key hit the
                     threshold. */
                  if (count[k] == tau1) {
                    keys[nkeys++] = (u16)k;
                  }
                }
                ddt0 >>= 8;
              } while (ddt0 != 0);
              ddt2 >>= 8;
            } while (ddt2 != 0);
          }
        }
      }

      /* Iterate over all partial LL^-1(rk8) where the flag was set. */
      for (int i = 0; i < (int)nkeys; i++) {
        u8 k1count[256] = {0};
        u64 k1flag[4] = {0};
        u32 k0 = keys[i] & 0xff;
        u32 k2 = keys[i] >> 8;
        for (const tuple_pair_t *p = pairs; p < pairs + num_pairs; p++) {
          u32 x70a = x70[k0 * num_ct + p->a];
          u32 x70b = x70[k0 * num_ct + p->b];
          u32 x72a = x72[k2 * num_ct + p->a];
          u32 x72b = x72[k2 * num_ct + p->b];
          u32 dx70 = ffmul_table_9[x72a ^ x72b];
          /* Skip pairs that do not match for this partial key. */
          if ((x70a ^ x70b ^ dx70) != 0) {
            continue;
          }
          u32 delta = ((ct[p->a].tweak ^ ct[p->b].tweak) >> 40) & 0xff;
          u32 dx71 = FFMUL2(x72a ^ x72b) ^ delta;
          u32 dy7 = v8[p->a] ^ v8[p->b];
          u32 diff1 = (dy7 & 0xff00) >> 8;
          u32 idx1 = (diff1 << 8) | dx71;
          u32 ddt1 = ddt[idx1];
          if (ddt1 == 0xffff) {
            continue;
          }
          do {
            u32 y71 = ddt1 & 0xff;
            u8 k1 = (u8)(y71 ^ ((v8[p->a] >> 8)));
            for (int j = 0; j < 2; j++) {
              u8 k = (u8)(k1 ^ j * diff1);
              k1count[k] += 1;
              /* Set flag if counter for the partial key hit the threshold. */
              if (k1count[k] >= tau2) {
                k1flag[k >> 6] |= 1ULL << (k & 0x3f);
              }
            }
            ddt1 >>= 8;
          } while (ddt1 != 0);
        }

        /* Iterate over all k1 values where the flag was set. */
        for (int j = 0; j < 4; j++) {
          while (k1flag[j] != 0) {
            u32 k1bit = (u32)CTZL(k1flag[j]);
            u32 flaggedk1 = (j << 6) | k1bit;
            u32 rk8 = mix_columns(rotate_rows(
                (k0 << 16) | (flaggedk1 << 8) | k2));
            u8 rk7 = 0;
            u32 keycount = 0;
            /* Validate the candidate key against the expected v7 difference and
             * recover the MSB of LL^-1(rk7). */
            RETURN_ON_ERROR(validate_rk8(
                ct,
                v8,
                num_ct,
                pairs,
                num_pairs,
                rk8,
                &rk7,
                &keycount));
            if ((int)keycount >= tau2) {
              print_message(
                  "Found candidate key: %02x %06x %06x %06x "
                  "(%d matching pairs)",
                  GREEN,
                  rk7,
                  rk8,
                  rk9n,
                  rk10n,
                  keycount);
              *num_candidates += 1;
              candidate_key_t *tmp = realloc(*candidates,
                  sizeof(u64) * *num_candidates);
              RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
              *candidates = tmp;
              candidate_key_t *k = *candidates + *num_candidates - 1;
              k->pairs = keycount;
              k->rk7 = rk7;
              k->rk8 = rk8;
              k->rk9 = rk9n;
              k->rk10 = rk10n;
            }
            k1flag[j] ^= 1ULL << k1bit;
          }
        }
      }
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*candidates);
    *num_candidates = 0;
  }
  free(v8);
  free(x9);
  free(x70);
  free(x72);
  return err;
}

/**
 * Generates a list of pairs with the required difference from a list of tuples.
 * @param ct a list of ciphertext-tweak tuples.
 * @param num_ct length of ct.
 * @param pairs output variable for the generated list of tuple pairs,
 * containing indexes of tuples in the paired_tuples list.
 * @param num_pairs output variable for the number of pairs in the list.
 */
static halfloop_result_t generate_pairs(
    const tuple_t *ct,
    int num_ct,
    tuple_pair_t **pairs,
    int *num_pairs) {
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(num_ct <= 2);
  CHECK_BAD_ARGUMENT(pairs == NULL);
  CHECK_BAD_ARGUMENT(num_pairs == NULL);

  *pairs = NULL;
  *num_pairs = 0;
  int pairs_alloc = 0;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  for (int i = 0; i < num_ct; i++) {
    u64 tw0 = ct[i].tweak;
    for (int j = i + 1; j < num_ct; j++) {
      u64 tw1 = ct[j].tweak;
      u64 dtw = tw0 ^ tw1;
      if (dtw >> 48) { /* End of potential matches in list. */
        break;
      }
      if ((dtw & 0xff0000000000ULL) != 0 && (dtw & ~0xff0000000000ULL) == 0) {
        if (*num_pairs == pairs_alloc) {
          pairs_alloc += 1000;
          tuple_pair_t *tmp = realloc(*pairs,
              sizeof(tuple_pair_t) * pairs_alloc);
          RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
          *pairs = tmp;
        }
        (*pairs)[*num_pairs].a = i;
        (*pairs)[*num_pairs].b = j;
        *num_pairs += 1;
      }
    }
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    FREE_AND_NULL(*pairs);
    *num_pairs = 0;
  }
  return err;

}

/**
 * Prints tool help information to stdout.
 * @param name the file name of the tool executable, normally from argv[0].
 */
halfloop_result_t print_help(char *name) {
  CHECK_BAD_ARGUMENT(name == NULL);
  printf("Usage: %s [OPTIONS] <filename> <chunk>\n"
         "   or: %s [OPTIONS]\n"
         "Implementation of ciphertext-only attacks on HALFLOOP-24.\n"
         "\n"
         "Parameters:\n"
         "filename        The path to a file that contains ciphertext and\n"
         "                tweak pairs in hexadecimal format, separated by\n"
         "                whitespace, with one pair per line.\n"
         "chunk           Up to 32 fixed bits of round keys 9 and 10. Entered\n"
         "                as hexadecimal digits, where the most significant\n"
         "                byte represents the least significant byte of rk 9\n"
         "                and the remaining three bytes represent rk 10.\n"
         "\n"
         "Options:\n"
         "  -1            Use CPU algorithm 1.\n"
         "  -2            Use CPU algorithm 2.\n"
         "  -3            Use CPU algorithm 3.\n"
#ifdef CUDA_ENABLED
         "  -4            Use GPU algorithm 2.\n"
         "  -5            Use GPU algorithm 3.\n"
#endif
         "  -b            Runs performance benchmarks.\n"
         "  -c p_ct       Set the probability that two randomly selected\n"
         "                callsigns will differ only in the least significant\n"
         "                byte. Entered as the exponential x in 2^x,\n"
         "                -32 >= x < 0. Default value: -10.22.\n"
#ifdef CUDA_ENABLED
         "  -d devlist    Use the specified CUDA devices. Devlist must be a\n"
         "                comma-separated list of device identifiers. The\n"
         "                default is to use all available devices.\n"
         "  -f            Skip brute force search for the last 48 key bits.\n"
         "  -l            List CUDA devices.\n"
         "  -m            Set number of CUDA blocks per multiprocessor.\n"
         "                Default value: 2\n"
#endif
         "  -p            Performs a partial search for profiling purposes.\n"
         "  -s p_success  Set the target probability of success,\n"
         "                0 < p_success < 1. The default value is 0.5.\n"
         "  -t tau1       Set tau1 value. Overrides values calculated from\n"
         "                p_ct and p_success.\n"
         "  -u tau2       Set tau2 value. The default is set to achieve\n"
         "                p_success >= 0.99 in the validation steps.\n"
         "  -v            Verbose - prints more information.\n"
         "\n",
         name, name);
  return HALFLOOP_SUCCESS;
}

#ifdef CUDA_ENABLED
/**
 * Called by parse_options to parse a list of comma-separated CUDA device ids
 * provided by the user.
 * @param options pointer to an options array, where the devices and num_devices
 * members will be updated.
 * @param dev a comma-separated list of CUDA devices.
 */
static halfloop_result_t parse_devices(
    struct options *options,
    const char *devs) {
  CHECK_BAD_ARGUMENT(options == NULL);
  CHECK_BAD_ARGUMENT(devs == NULL);
  char *buf = NULL;
  halfloop_result_t err = HALFLOOP_SUCCESS;
  size_t devlen = strlen(devs) + 1;
  buf = malloc(sizeof(char) * devlen);
  RETURN_IF(buf == NULL, HALFLOOP_MEMORY_ERROR);
  memcpy(buf, devs, devlen);
  options->num_devices = 0;
  char *saveptr = NULL;
#ifdef _WIN32
#define strtok_r strtok_s
#endif
  char *next = strtok_r(buf, ",", &saveptr);
  while (next != NULL && options->num_devices < MAX_DEVICES) {
    options->devices[options->num_devices] = atoi(next);
    RETURN_IF(options->devices[options->num_devices] < 0, HALFLOOP_FAILURE);
    options->num_devices += 1;
    next = strtok_r(NULL, ",", &saveptr);
  }
#ifdef _WIN32
#undef strtok_r
#endif
  /* Check for duplicates. */
  for (int i = 0; i < options->num_devices; i++) {
    for (int j = i + 1; j < options->num_devices; j++) {
      RETURN_IF(options->devices[i] == options->devices[j], HALFLOOP_FAILURE);
    }
  }
error:
  free(buf);
  return err;
}
#endif /* CUDA_ENABLED */

/**
 * Called by parse_options to parse an integer provided by the user.
 * @param intval return pointer for the integer value.
 * @param param the string to parse the integer from.
 * @return HALFLOOP_SUCCESS on success and HALFLOOP_BAD_ARGUMENT on parse
 * failure.
 */
halfloop_result_t parse_int(u32 *intval, const char *param) {
  CHECK_BAD_ARGUMENT(intval == NULL);
  CHECK_BAD_ARGUMENT(param == NULL);
  halfloop_result_t err = HALFLOOP_SUCCESS;
  *intval = atoi(param);
  RETURN_IF(*intval <= 0, HALFLOOP_BAD_ARGUMENT);
error:
  return err;
}

/**
 * Called parse_options to parse a double provided by the user. Also performs
 * bounds checking on the parsed variable.
 * @param d return pointer for the parsed double.
 * @param param the string to parse the double from.
 * @param min the double minimum value, inclusive.
 * @param max the double maximum value, inclusive.
 * @return HALFLOOP_SUCCESS on success and HALFLOOP_BAD_ARGUMENT on parse
 * failure.
 */
halfloop_result_t parse_double(
    double *d,
    const char *param,
    double min,
    double max) {
  CHECK_BAD_ARGUMENT(d == NULL);
  CHECK_BAD_ARGUMENT(param == NULL);
  CHECK_BAD_ARGUMENT(min >= max);
  halfloop_result_t err = HALFLOOP_SUCCESS;
  *d = atof(param);
  RETURN_IF(*d >= max || *d <= min, HALFLOOP_BAD_ARGUMENT);
error:
  return err;
}

/**
 * Parses the command line options.
 * @param argc the argument count supplied to main.
 * @param argv the argument vector supplied to main.
 * @param options pointer to the options array to set with the results of the
 * options parsing.
 */
halfloop_result_t parse_options(
    int argc,
    char *argv[],
    struct options *options) {
  CHECK_BAD_ARGUMENT(argc < 1);
  CHECK_BAD_ARGUMENT(argv == NULL);
  CHECK_BAD_ARGUMENT(options == NULL);

  halfloop_result_t err = HALFLOOP_SUCCESS;

  RETURN_IF(argc == 1, HALFLOOP_FAILURE);

  int opt;
#ifdef CUDA_ENABLED
  const char *optstring = "12345bc:d:flm:ps:t:u:v";
#else
  const char *optstring = "123bc:fps:t:u:v";
#endif
  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
      case '1': options->algorithm = CPU_ATTACK1; break;
      case '2': options->algorithm = CPU_ATTACK2; break;
      case '3': options->algorithm = CPU_ATTACK3; break;
#ifdef CUDA_ENABLED
      case '4': options->algorithm = GPU_ATTACK2; break;
      case '5': options->algorithm = GPU_ATTACK3; break;
#endif
      case 'b': options->benchmark = true; break;
      case 'c': RETURN_ON_ERROR(parse_double(&options->p_ct, optarg, -32, 0));
                break;
#ifdef CUDA_ENABLED
      case 'd': RETURN_ON_ERROR(parse_devices(options, optarg)); break;
      case 'f': options->brute_force = false; break;
      case 'l': options->listdevs = true; break;
      case 'm': RETURN_ON_ERROR(parse_int(&options->blockmul, optarg)); break;
#endif
      case 'p': options->profile = true; break;
      case 's': RETURN_ON_ERROR(parse_double(
                    &options->p_success,
                    optarg,
                    0,
                    1));
                break;
      case 't': RETURN_ON_ERROR(parse_int(&options->tau1, optarg)); break;
      case 'u': RETURN_ON_ERROR(parse_int(&options->tau2, optarg)); break;
      case 'v': options->verbose = true; break;
      case ':':
      case '?':
        RETURN_IF(true, HALFLOOP_FAILURE);
    }
  }

  if (!options->benchmark && !options->listdevs) {
    RETURN_IF(argc - optind < 2, HALFLOOP_FAILURE);
    options->filename = argv[optind];
    int chunklen = (int)strlen(argv[optind + 1]);
    RETURN_IF(chunklen == 0 || chunklen > 8, HALFLOOP_FAILURE);
    options->num_fixed = chunklen * 4;
    options->fixed_bits = (u32)strtoll(argv[optind + 1], NULL, 16);
  }

error:
  if (err != HALFLOOP_SUCCESS) {
    print_help(argv[0]);
  }
  return err;
}

/**
 * Program main function.
 * @param argc program argument count.
 * @param argv program argument vector.
 * @return program return value.
 */
int main(int argc, char *argv[]) {
  tuple_t *ct = NULL;
  tuple_pair_t *pairs = NULL;
  candidate_key_t *candidates = NULL;
  u32 num_candidates = 0;
  int num_ct = 0;
  int num_pairs = 0;
  int num_devices = 0;
  double p_tau1 = 0.0;
  double p_tau2 = 0.0;
  char *device_names = NULL;
  struct options options = {
    .benchmark = false,
    .listdevs = false,
    .verbose = false,
    .profile = false,
#ifdef CUDA_ENABLED
    .brute_force = true,
    .algorithm = GPU_ATTACK3,
#else
    .brute_force = false,
    .algorithm = CPU_ATTACK3,
#endif
    .filename = NULL,
    .fixed_bits = 0,
    .blockmul = 2,
    .devices = {0},
    .num_devices = 0,
    .num_fixed = 0,
    .p_ct = -10.22,
    .p_success = 0.5
  };
  halfloop_result_t err = HALFLOOP_SUCCESS;

  /* Parse command line arguments. */
  err = parse_options(argc, argv, &options);
  if (err != HALFLOOP_SUCCESS) {
    return err;
  }

  print_message("Initializing HALFLOOP-24 library.", WHITE);
  RETURN_ON_ERROR(init_halfloop());
  RETURN_ON_ERROR(test_halfloop());
#ifdef CUDA_ENABLED
  RETURN_ON_ERROR(halfloop_init_cuda());
  RETURN_ON_ERROR(halfloop_list_cuda_devices(&num_devices, &device_names));
#endif
  if (options.num_devices != 0) {
    print_message("Selected devices:", WHITE);
  }
  for (int i = 0; i < options.num_devices; i++) {
    if (options.devices[i] >= num_devices) {
      print_message("Invalid device id: %d", RED, options.devices[i]);
      RETURN_IF(true, HALFLOOP_FAILURE);
    }
    print_message("%d: %s", WHITE, options.devices[i],
        device_names + options.devices[i] * 256);
  }

  if (options.listdevs) {
    printf("Available CUDA devices:\n");
    printf("ID  Name\n");
    for (int i = 0; i < num_devices; i++) {
      printf("%2d  %s\n", i, device_names + i * 256);
    }
    RETURN_IF(true, HALFLOOP_SUCCESS);
  }

  setlocale(LC_NUMERIC, ""); /* For pretty-printing large numbers. */

  if (options.benchmark) {
    RETURN_ON_ERROR(test_halfloop_bitslice());
    RETURN_ON_ERROR(halfloop_benchmark_bitslice());
#ifdef CUDA_ENABLED
    RETURN_ON_ERROR(test_halfloop_cuda_bitslice());
#endif
    RETURN_IF(true, HALFLOOP_SUCCESS);
  }

  print_message("Loading tuples from %s.", WHITE, options.filename);
  RETURN_ON_ERROR(read_input_pairs(options.filename, &ct, &num_ct));
  print_message("Loaded %d tuples.", WHITE, num_ct);

  RETURN_ON_ERROR(generate_pairs(ct, num_ct, &pairs, &num_pairs));
  print_message("Identified %d viable tweak pairs.", GREEN, num_pairs);

  const char *algorithm_strings[] = {
    "CPU algorithm 1",
    "CPU algorithm 2",
    "CPU algorithm 3",
    "GPU algorithm 2",
    "GPU algorithm 3",
  };
  print_message("Using %s.", WHITE, algorithm_strings[options.algorithm]);

  print_message("p_ct = 2^%.2f", WHITE, options.p_ct);
  print_message("p_success = %.2f", WHITE, options.p_success);
  double p_ct = pow(2, options.p_ct);
  u32 num_counters = 0;
  double keys_per_match = 0;
  switch (options.algorithm) {
    case CPU_ATTACK1:
    case CPU_ATTACK2:
    case GPU_ATTACK2:
      num_counters = 0x1000000;
      keys_per_match = 126.0;
      break;
    case CPU_ATTACK3:
    case GPU_ATTACK3:
      num_counters = 0x10000;
      keys_per_match = 63.0;
      break;
  }
  if (options.tau1 == 0) {
    RETURN_ON_ERROR(get_tau(
        num_pairs,
        num_counters,
        keys_per_match,
        p_ct,
        options.p_success,
        &options.tau1,
        &p_tau1));
  } else {
    if (options.algorithm == GPU_ATTACK2 && options.tau1 >= 16) {
      options.tau1 = 15;
    }
    RETURN_ON_ERROR(calc_p_success(
        options.tau1,
        num_pairs,
        0x10000,
        63.0,
        p_ct,
        &p_tau1));
  }
  double p_ct2 = (num_pairs * p_ct)
      / (num_pairs * p_ct + 63.0 * num_pairs / 0x10000);
  if (options.tau2 == 0) {
    RETURN_ON_ERROR(get_tau(
        options.tau1,
        0x100,
        2.02,
        p_ct2,
        0.7,
        &options.tau2,
        &p_tau2));
  } else {
    RETURN_ON_ERROR(calc_p_success(
        options.tau2,
        options.tau1,
        0x100,
        2.02,
        p_ct2,
        &p_tau2));
  }
  print_message(
      "Setting tau1 = %2d. Success probability: %.3f",
      WHITE,
      options.tau1,
      p_tau1);
  print_message(
      "Setting tau2 = %2d. Success probability: %.3f",
      WHITE,
      options.tau2,
      p_tau2);

  print_message(
      "%d fixed bits: %0*x",
      WHITE,
      options.num_fixed,
      options.num_fixed / 4,
      options.fixed_bits);

  hltimer time;
  TIMER_START(&time);
  switch (options.algorithm) {
    case CPU_ATTACK1:
      RETURN_ON_ERROR(ct_attack1(
          ct,
          num_ct,
          pairs,
          num_pairs,
          options.tau1,
          options.tau2,
          options.fixed_bits,
          options.num_fixed,
          options.verbose,
          &candidates,
          &num_candidates));
      break;
    case CPU_ATTACK2:
      RETURN_ON_ERROR(ct_attack2(
          ct,
          num_ct,
          pairs,
          num_pairs,
          options.tau1,
          options.tau2,
          options.fixed_bits,
          options.num_fixed,
          options.verbose,
          &candidates,
          &num_candidates));
      break;
    case CPU_ATTACK3:
      RETURN_ON_ERROR(ct_attack3(
          ct,
          num_ct,
          pairs,
          num_pairs,
          options.tau1,
          options.tau2,
          options.fixed_bits,
          options.num_fixed,
          options.verbose,
          &candidates,
          &num_candidates));
      break;
    case GPU_ATTACK2:
    case GPU_ATTACK3:
#ifdef CUDA_ENABLED
      RETURN_ON_ERROR(cuda_ct_attack(
          options.algorithm,
          ct,
          num_ct,
          pairs,
          num_pairs,
          options.tau1,
          options.tau2,
          options.blockmul,
          options.fixed_bits,
          options.num_fixed,
          &candidates,
          &num_candidates,
          options.verbose,
          options.profile,
          options.num_devices == 0 ? NULL : options.devices,
          options.num_devices));
#endif
      break;
  }
  TIMER_STOP(&time);
  print_message("%f seconds", WHITE, timer_elapsed(time));
  if (num_candidates == 0) {
    print_message("No candidate keys found", RED);
    RETURN_IF(true, HALFLOOP_SUCCESS);
  }
  print_message(
      "%d candidate%s found",
      GREEN,
      num_candidates,
      num_candidates == 1 ? "" : "s");

  RETURN_IF(!options.brute_force, HALFLOOP_SUCCESS);
#ifdef CUDA_ENABLED
  u32 ct0 = 0;
  u32 ct1 = 0;
  u64 tw0 = 0;
  bool found_pair = false;
  for (u32 i = 0; i < num_ct && !found_pair; i++) {
    for (u32 j = i + 1; j < num_ct && !found_pair; j++) {
      if ((ct[i].tweak ^ (1 << 30)) == ct[j].tweak) {
        ct0 = ct[i].ct;
        ct1 = ct[j].ct;
        tw0 = ct[i].tweak;
        found_pair = true;
      }
    }
  }
  if (!found_pair) {
    print_message("Could not find suitable for brute force key search", RED);
    RETURN_IF(true, HALFLOOP_FAILURE);
  }
  print_message(
      "Brute force ct0: %06x ct1: %06x tw0: %016" PRIx64,
      WHITE,
      ct0,
      ct1,
      tw0);
  qsort(
      candidates,
      num_candidates,
      sizeof(candidate_key_t),
      compare_candidates);
  for (u32 i = 0; i < num_candidates; i++) {
    candidate_key_t *c = candidates + i;
    print_message(
        "Testing candidate key %02x %06x %06x %06x (%u matches)",
        WHITE,
        c->rk7,
        c->rk8,
        c->rk9,
        c->rk10,
        c->pairs);
    hlkey found = {0};
    err = halfloop_cuda_bitslice_all(
        ct0,
        ct1,
        tw0,
        ct,
        num_ct,
        c->rk7,
        c->rk8,
        c->rk9,
        c->rk10,
        options.devices,
        options.num_devices,
        options.verbose,
        &found);

    if (err == HALFLOOP_SUCCESS) {
      print_message(
          "Found key: %016" PRIx64 "%016" PRIx64,
          GREEN,
          found.hi,
          found.lo);
      RETURN_IF(true, HALFLOOP_SUCCESS);
    } else if (err == HALFLOOP_FAILURE) {
      err = HALFLOOP_SUCCESS;
    }
    RETURN_ON_ERROR(err);
  }

  print_message("No matching key found.", RED);
#endif /* CUDA_ENABLED */

error:
  free(device_names);
  free(candidates);
  free(ct);
  free(pairs);
  return err;
}
