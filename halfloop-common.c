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

#include <fcntl.h>
#include <immintrin.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "halfloop-common.h"

u8 ffmul_table_2[0x100] = {0};
u8 ffmul_table_6[0x100] = {0};
u8 ffmul_table_8[0x100] = {0};
u8 ffmul_table_9[0x100] = {0};
u8 ffmul_table_39[0x100] = {0};


#ifdef _WIN32

u64 timer_minutes(hltimer t) {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return t.QuadPart / (freq.QuadPart * 60);
}

u64 timer_seconds(hltimer t) {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return (t.QuadPart / freq.QuadPart) % 60;
}

double timer_elapsed(hltimer t) {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return t.QuadPart / (double)freq.QuadPart;
}

int ffsl(u64 v) {
  unsigned long idx;
  if (!_BitScanForward64(&idx, v)) {
    return 0;
  }
  return (int)idx + 1;
}

int clzl(u32 v) {
  u32 idx;
  if (!_BitScanReverse(&idx, v)) {
    return 32;
  }
  return 31 - idx;
}

#else /* _WIN32 */
u64 timer_minutes(hltimer t) {
  return t.tv_sec / 60;
}

u64 timer_seconds(hltimer t) {
  return t.tv_sec % 60;
}

double timer_elapsed(hltimer t) {
  return t.tv_sec + t.tv_nsec * 1E-9;
}
#endif /* _WIN32 */

/**
 * Performs finite field multiplication in F^8_2. Used by init_halfloop to
 * initialize the finite field multiplication lookup tables.
 * @param a term a.
 * @param b term b.
 * @return the product.
 */
static u8 ffmul(u8 a, u8 b) {
  u32 c = 0;

  for (int x = 0; x < 8; x++) {
    for (int y = 0; y < 8; y++) {
      if ((a >> x) & (b >> y) & 1) {
        c ^= (1 << (x + y));
      }
    }
  }
  while (c > 0xff) {
    c ^= 0x11b << (23 - CLZ(c));
  }
  return (u8)c;
}

const char* halfloop_get_result_text(halfloop_result_t result) {
  if (result < 0 || result > 10) {
    return NULL;
  }
  const char *result_strings[] = {
    "HALFLOOP_SUCCESS",
    "HALFLOOP_BAD_ARGUMENT",
    "HALFLOOP_FILE_ERROR",
    "HALFLOOP_END_OF_FILE",
    "HALFLOOP_FORMAT_ERROR",
    "HALFLOOP_NOT_IMPLEMENTED",
    "HALFLOOP_INTERNAL_ERROR",
    "HALFLOOP_MEMORY_ERROR",
    "HALFLOOP_FAILURE",
    "HALFLOOP_QUIT",
    "HALFLOOP_NETWORK_ERROR"
  };
  return result_strings[result];
}

halfloop_result_t init_halfloop(void) {
  /* Initialize the finite field multiplication lookup tables. */
  for (int i = 0; i < 0x100; i++) {
    ffmul_table_2[i]  = ffmul(2, (u8)i);
    ffmul_table_6[i]  = ffmul(6, (u8)i);
    ffmul_table_8[i]  = ffmul(8, (u8)i);
    ffmul_table_9[i]  = ffmul(9, (u8)i);
    ffmul_table_39[i] = ffmul(39, (u8)i);
  }
  return HALFLOOP_SUCCESS;
}

u32 sub_bytes(u32 state) {
  u8 a0 = (u8)(state >> 16);
  u8 a1 = (u8)(state >> 8);
  u8 a2 = (u8)state;
  return ((u32)SBOX[a0] << 16) | ((u32)SBOX[a1] << 8) | (u32)SBOX[a2];
}

u32 inv_sub_bytes(u32 state) {
  u8 a0 = (u8)(state >> 16);
  u8 a1 = (u8)(state >> 8);
  u8 a2 = (u8)state;
  return ((u32)inv_SBOX[a0] << 16) | ((u32)inv_SBOX[a1] << 8)
      | (u32)inv_SBOX[a2];
}

u32 rotate_rows(u32 state) {
  u32 a0 = state & 0xff0000;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  a1 = (a1 << 6) | (a1 >> 2);
  a2 = (a2 << 4) | (a2 >> 4);
  return a0 | (a1 << 8) | a2;
}

u32 inv_rotate_rows(u32 state) {
  u32 a0 = state & 0xff0000;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  a1 = (a1 >> 6) | (a1 << 2);
  a2 = (a2 >> 4) | (a2 << 4);
  return a0 | (a1 << 8) | a2;
}

u32 mix_columns(u32 in) {
  u32 a = in >> 16;
  u32 b = (in >> 8) & 0xff;
  u32 c = in & 0xff;
  u32 out = (ffmul_table_9[a] ^               b  ^ ffmul_table_2[c]) << 16;
  out    |= (ffmul_table_2[a] ^ ffmul_table_9[b] ^               c)  << 8;
  out    |= (              a  ^ ffmul_table_2[b] ^ ffmul_table_9[c]);
  return out;
}

u32 inv_mix_columns(u32 in) {
  u32 a = in >> 16;
  u32 b = (in >> 8) & 0xff;
  u32 c = in & 0xff;
  u32 out = (ffmul_table_6[a]  ^ ffmul_table_8[b]  ^ ffmul_table_39[c]) << 16;
  out    |= (ffmul_table_39[a] ^ ffmul_table_6[b]  ^ ffmul_table_8[c])  << 8;
  out    |= (ffmul_table_8[a]  ^ ffmul_table_39[b] ^ ffmul_table_6[c]);
  return out;
}

u32 key_schedule_g(u32 key_word, u32 rc) {
  u8 b0 =  key_word >> 24;
  u8 b1 = (key_word >> 16) & 0xFF;
  u8 b2 = (key_word >> 8)  & 0xFF;
  u8 b3 =  key_word        & 0xFF;
  return ((SBOX[b1] ^ rc) << 24)
        ^ (SBOX[b2]       << 16)
        ^ (SBOX[b3]       << 8)
        ^  SBOX[b0];
}

halfloop_result_t key_schedule(u32 *rk, hlkey key, u64 tweak) {
  if (rk == NULL) {
    return HALFLOOP_BAD_ARGUMENT;
  }
  key.hi ^= tweak;
  rk[0]   =  (key.hi >> 40) & 0xffffff;
  rk[1]   =  (key.hi >> 16) & 0xffffff;
  rk[2]   = ((key.hi << 8)  & 0xffff00) | (key.lo >> 56);
  rk[3]   =  (key.lo >> 32) & 0xffffff;
  rk[4]   =  (key.lo >> 8)  & 0xffffff;
  rk[5]   =  (key.lo & 0xff) << 16;
  key.hi ^= (u64)key_schedule_g(key.lo & 0xffffffff, 1) << 32;
  key.hi ^=   key.hi >> 32;
  key.lo ^=  (key.hi & 0xffffffff) << 32;
  key.lo ^=   key.lo >> 32;
  rk[5]  |=   key.hi >> 48;
  rk[6]   =  (key.hi >> 24) & 0xffffff;
  rk[7]   =   key.hi & 0xffffff;
  rk[8]   =   key.lo >> 40;
  rk[9]   =  (key.lo >> 16) & 0xffffff;
  rk[10]  =  (key.lo << 8)  & 0xffff00;
  key.hi ^= (u64)key_schedule_g(key.lo & 0xffffffff, 2) << 32;
  rk[10] |= key.hi >> 56;
  return HALFLOOP_SUCCESS;
}

halfloop_result_t halfloop_round_tweak(u64 tweak, u8 round, u32 *rtweak) {
  CHECK_BAD_ARGUMENT(round > 9);
  CHECK_BAD_ARGUMENT(rtweak == NULL);
  switch (round) {
    case 0: *rtweak = (tweak >> 40) & 0xffffff; break;
    case 1: *rtweak = (tweak >> 16) & 0xffffff; break;
    case 2: *rtweak = (tweak << 8)  & 0xffff00; break;
    case 3: *rtweak = 0; break;
    case 4: *rtweak = 0; break;
    case 5: *rtweak = (tweak >> 48); break;
    case 6: *rtweak = ((tweak >> 24) & 0xffffff) ^ (tweak >> 56); break;
    case 7: *rtweak = (tweak ^ (tweak >> 32)) & 0xffffff; break;
    case 8: *rtweak = ((tweak >> 8) ^ (tweak >> 40)) & 0xffffff; break;
    case 9: *rtweak = (((tweak << 16) & 0xff0000) | ((tweak >> 48) & 0x00ffff))
        ^ ((tweak >> 16) & 0xffffff); break;
  }
  return HALFLOOP_SUCCESS;
}

halfloop_result_t halfloop_round10_tweak(u64 tweak, u8 rk9lsb, u32 *rtweak) {
  CHECK_BAD_ARGUMENT(rtweak == NULL);
  *rtweak = (((tweak << 8) ^ (tweak >> 24)) & 0xffff00)
      | ((tweak >> 56) ^ SBOX[rk9lsb]
      ^ SBOX[(rk9lsb ^ (tweak >> 16) ^ (tweak >> 48)) & 0xff]);
  return HALFLOOP_SUCCESS;
}

static u32 halfloop_decrypt_round(u32 state, u32 round_key, bool last_round) {
  state ^= round_key;
  if (!last_round) {
    state = inv_mix_columns(state);
  }
  state = inv_rotate_rows(state);
  return inv_sub_bytes(state);
}

static u32 halfloop_encrypt_round(u32 state, u32 round_key, bool last_round) {
  state = sub_bytes(state);
  state = rotate_rows(state);
  if (!last_round) {
    state = mix_columns(state);
  }
  return state ^ round_key;
}

halfloop_result_t halfloop_encrypt(u32 pt, hlkey key, u64 tweak, u32 *ct) {
  if ((pt & 0xFF000000) != 0 || ct == NULL) {
    return HALFLOOP_BAD_ARGUMENT;
  }
  halfloop_result_t err = HALFLOOP_SUCCESS;
  u32 rk[11] = {0};
  RETURN_ON_ERROR(key_schedule(rk, key, tweak));
  *ct = pt ^ rk[0];
  for(int i = 1; i < 10; i++) {
    *ct = halfloop_encrypt_round(*ct, rk[i], false);
  }
  *ct = halfloop_encrypt_round(*ct, rk[10], true);
error:
  return err;
}

halfloop_result_t halfloop_decrypt(u32 ct, hlkey key, u64 tweak, u32 *pt) {
  if ((ct & 0xFF000000) != 0 || pt == NULL) {
    return HALFLOOP_BAD_ARGUMENT;
  }
  halfloop_result_t err = HALFLOOP_SUCCESS;
  u32 rk[11] = {0};
  RETURN_ON_ERROR(key_schedule(rk, key, tweak));
  *pt = halfloop_decrypt_round(ct, rk[10], true);
  for(int i = 9; i > 0; i--) {
    *pt = halfloop_decrypt_round(*pt, rk[i], false);
  }
  *pt ^= rk[0];
error:
  return err;
}

halfloop_result_t halfloop_decrypt_all(u32 ct, hlkey key, u64 tweak, u32 *pt) {
  if ((ct & 0xFF000000) != 0 || pt == NULL) {
    return HALFLOOP_BAD_ARGUMENT;
  }
  halfloop_result_t err = HALFLOOP_SUCCESS;
  u32 rk[11] = {0};
  RETURN_ON_ERROR(key_schedule(rk, key, tweak));
  pt[10] = halfloop_decrypt_round(ct, rk[10], true);
  for(int i = 9; i > 0; i--) {
    pt[i] = halfloop_decrypt_round(pt[i + 1], rk[i], false);
  }
  pt[0] = pt[1] ^ rk[0];
error:
  return err;
}

halfloop_result_t test_halfloop(void) {
  hlkey key = {
   .hi = 0x2b7e151628aed2a6ULL,
   .lo = 0xabf7158809cf4f3cULL
  };
  u64 tweak = 0x543bd88000017550;
  u32 pt = 0x010203;
  u32 ct = 0xf28c1e;
  u32 ct_test = 0;
  u32 pt_test = 0;
  u32 pt_array[11] = {0};
  halfloop_result_t err = HALFLOOP_SUCCESS;

  for (int i = 0; i < 0x100; i++) {
    RETURN_IF(inv_SBOX[SBOX[i]] != i, HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(FFMUL2(i) != ffmul(2, (u8)i), HALFLOOP_INTERNAL_ERROR);
    RETURN_IF(FFMUL9(i) != ffmul(9, (u8)i), HALFLOOP_INTERNAL_ERROR);
  }
  RETURN_ON_ERROR(halfloop_encrypt(pt, key, tweak, &ct_test));
  RETURN_IF(ct_test != ct, HALFLOOP_INTERNAL_ERROR);
  RETURN_ON_ERROR(halfloop_decrypt(ct, key, tweak, &pt_test));
  RETURN_IF(pt_test != pt, HALFLOOP_INTERNAL_ERROR);
  RETURN_ON_ERROR(halfloop_decrypt_all(ct, key, tweak, pt_array));
  RETURN_IF(pt_array[0] != pt, HALFLOOP_INTERNAL_ERROR);
error:
  return err;
}

bool halfloop_valid_plaintext(u32 pt) {
  pt &= 0xffffff;
  if ((pt >> 21) != 0x2) {
    return false;
  }
  for (int i = 0; i < 3; i++) {
    char c = pt & 0x7f;
    if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
      return false;
    }
    pt >>= 7;
  }
  return true;
}

halfloop_result_t print_message(const char *format, color_t color, ...) {
  if (format == NULL) {
    return HALFLOOP_BAD_ARGUMENT;
  }
  char *str = NULL;
  halfloop_result_t err = HALFLOOP_SUCCESS;

  va_list ap;
  va_start(ap, color);
  int len = vsnprintf(NULL, 0, format, ap) + 1;
  va_end(ap);
  str = malloc(len);
  RETURN_IF(str == NULL, HALFLOOP_MEMORY_ERROR);
  va_start(ap, color);
  vsnprintf(str, len, format, ap);
  va_end(ap);

  const char *normal_color = "\x1B[0m";
  const char *colorstring = normal_color;
  switch (color) {
    case RED:   colorstring = "\x1B[31m"; break;
    case GREEN: colorstring = "\x1B[32m"; break;
    case BLUE:  colorstring = "\x1B[34m"; break;
    default: break;
  }

#ifdef _WIN32
  SYSTEMTIME st;
  GetLocalTime(&st);
  printf(
      "[%02d:%02d:%02d] %s%s%s\n",
      st.wHour,
      st.wMinute,
      st.wSecond,
      colorstring,
      str,
      normal_color);
#else /* _WIN32 */
  struct timeval tv;
  struct tm tm;
  RETURN_IF(gettimeofday(&tv, NULL) != 0, HALFLOOP_INTERNAL_ERROR);
  localtime_r(&tv.tv_sec, &tm);
  printf(
      "[%02d:%02d:%02d] %s%s%s\n",
      tm.tm_hour,
      tm.tm_min,
      tm.tm_sec,
      colorstring,
      str,
      normal_color);
#endif

error:
  free(str);
  return err;
}

/**
 * @brief Checks a HALFLOOP tweak structure to ensure all values are compliant
 * with the specification.
 *
 * @param tweak a HALFLOOP tweak structure.
 * @return halfloop_result_t HALFLOOP_SUCCESS if the values in the structure are
 * compliant.
 */
static halfloop_result_t check_tweak(tweak_t tweak) {
  int days[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (   tweak.month < 1
      || tweak.month > 12
      || tweak.day < 1
      || tweak.day > days[tweak.month - 1]
      || tweak.coarse_time < 0
      || tweak.coarse_time >= 1440
      || tweak.fine_time < 0
      || tweak.fine_time >= 60
      || tweak.word < 0
      || tweak.word > 255
      || tweak.zero != 0
      || tweak.frequency <= 0
      || tweak.frequency >= 1000000000
      || tweak.frequency % 100 != 0) {
    return HALFLOOP_FORMAT_ERROR;
  }
  return HALFLOOP_SUCCESS;
}

halfloop_result_t parse_tweak(u64 tweak, tweak_t *parsed) {
  tweak_t p = {
    .month       =  tweak >> 60,
    .day         = (tweak >> 55) & 0x1f,
    .coarse_time = (tweak >> 44) & 0x3ff,
    .fine_time   = (tweak >> 38) & 0x3f,
    .word        = (tweak >> 30) & 0xff,
    .zero        = (tweak >> 28) & 0x3,
    .frequency   = 0
  };
  for (int i = 0; i < 7; i++) {
    p.frequency *= 10;
    int d = (tweak >> (24 - i * 4)) & 0xf;
    if (d >= 10) {
      return HALFLOOP_FORMAT_ERROR;
    }
    p.frequency += d;
  }
  p.frequency *= 100;
  if (check_tweak(p) != HALFLOOP_SUCCESS) {
    return HALFLOOP_FORMAT_ERROR;
  }
  memcpy(parsed, &p, sizeof(tweak_t));
  return HALFLOOP_SUCCESS;
}

halfloop_result_t create_tweak(tweak_t values, u64 *tweak) {
  if (tweak == NULL) {
    return HALFLOOP_BAD_ARGUMENT;
  }
  if (check_tweak(values) != HALFLOOP_SUCCESS) {
    return HALFLOOP_FORMAT_ERROR;
  }
  *tweak  = (u64)values.month << 60;
  *tweak |= (u64)values.day << 55;
  *tweak |= (u64)values.coarse_time << 44;
  *tweak |= (u64)values.fine_time << 38;
  *tweak |= (u64)values.word << 30;
  values.frequency /= 100;
  for (int i = 0; i < 7; i++) {
    *tweak |= (values.frequency % 10) << i * 4;
    values.frequency /= 10;
  }
  return HALFLOOP_SUCCESS;
}

#ifdef _WIN32
halfloop_result_t random_bytes(void* b, size_t num) {
  halfloop_result_t err = HALFLOOP_SUCCESS;
  RETURN_IF(BCryptGenRandom(NULL,
                            b,
                            (ULONG)num,
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG),
      HALFLOOP_INTERNAL_ERROR);
error:
  return err;
}
#else /* _WIN32 */
halfloop_result_t random_bytes(void *b, size_t num) {
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0) {
    return HALFLOOP_FILE_ERROR;
  }
  ssize_t r = read(fd, b, num);
  close(fd);
  if (r != num) {
    return HALFLOOP_FILE_ERROR;
  }
  return HALFLOOP_SUCCESS;
}
#endif

halfloop_result_t init_ddt(u16 *ddt) {
  CHECK_BAD_ARGUMENT(ddt == NULL);
  memset(ddt, 0xff, sizeof(u16) * 256 * 256);
  for (int i = 0; i < 256; i++) {
    for (int dout = 1; dout < 256; dout++) {
      u8 a = inv_SBOX[i];
      u8 b = inv_SBOX[i ^ dout];
      u8 din = a ^ b;
      int idx = dout * 256 | din;
      if (ddt[idx] == 0xffff) {
        ddt[idx] = (u16)i;
      } else if ((ddt[idx] & 0xff00) == 0 && (ddt[idx] ^ dout) != i) {
        ddt[idx] |= i << 8;
      }
    }
  }
  return HALFLOOP_SUCCESS;
}

halfloop_result_t init_y0_lut(u8 y0delta, __m256i *lut) {
  CHECK_BAD_ARGUMENT(lut == NULL);

  if (y0delta == 0) {
    *lut = _mm256_setzero_si256();
    return HALFLOOP_SUCCESS;
  }

  /* Temporary lookup table for converting x0 to x2. */
  u8 w[256] = {0};
  for (u32 gamma = 1; gamma < 256; gamma++) {
    u32 v = mix_columns(gamma << 16);
    w[v >> 16] = v & 0xff;
  }

  u64 l[4] = {0};
  for (u32 i = 0; i < 256; i++) {
    u32 v = w[inv_SBOX[i] ^ inv_SBOX[i ^ y0delta]];
    l[v >> 6] |= 1ULL << (v & 0x3f);
  }
  *lut = _mm256_set_epi64x(l[3], l[2], l[1], l[0]);

  return HALFLOOP_SUCCESS;
}

halfloop_result_t init_y1_lut(u8 y1delta, u8 delta, __m256i *lut) {
  CHECK_BAD_ARGUMENT(delta == 0);
  CHECK_BAD_ARGUMENT(lut == NULL);

  /* Temporary lookup table for converting x1 to x2. */
  u8 w[256] = {0};
  for (u32 gamma = 1; gamma < 256; gamma++) {
    u32 v = mix_columns(gamma << 16);
    w[(v >> 8) & 0xff] = v & 0xff;
  }

  u64 l[4] = {0};
  for (u32 i = 0; i < 256; i++) {
    u32 v = w[inv_SBOX[i] ^ inv_SBOX[i ^ y1delta] ^ delta];
    l[v >> 6] |= 1ULL << (v & 0x3f);
  }
  __m256i y1lut = _mm256_set_epi64x(l[3], l[2], l[1], l[0]);

  /* Calculate viable gamma values. */
  u64 g[4] = {0};
  for (u32 i = 0; i < 256; i++) {
    u32 gamma = SBOX[i] ^ SBOX[i ^ delta];
    g[gamma >> 6] |= 1ULL << (gamma & 0x3f);
  }
  *lut = _mm256_and_si256(y1lut, _mm256_set_epi64x(g[3], g[2], g[1], g[0]));

  return HALFLOOP_SUCCESS;
}

halfloop_result_t init_y2_lut(u8 y2delta, __m256i *lut) {
  CHECK_BAD_ARGUMENT(lut == NULL);

  u64 l[4] = {0};
  for (u32 i = 0; i < 256; i++) {
    u32 v = inv_SBOX[i] ^ inv_SBOX[i ^ y2delta];
    l[v >> 6] |= 1ULL << (v & 0x3f);
  }
  *lut = _mm256_set_epi64x(l[3], l[2], l[1], l[0]);

  return HALFLOOP_SUCCESS;
}

/**
 * Validates a candidate for round key 8 and calculates the MSB of LL^-1(rk7).
 * @param ct a list of ciphertext-tweak tuples.
 * @param v8 a list of v8 states in the same order as the ct list.
 * @param num_ct number of items in ct and v8.
 * @param pairs a list of pairs with the required tweak difference, where the
 * indexes correspond to indexes in ct and v8.
 * @param num_pairs number of items in the pairs list.
 * @param rk8 the candidate rk8.
 * @param key return variable for the candidate for the MSB of LL^-1(rk7).
 * @param num number of pairs that matched the required v7 and x6 differences.
 */
halfloop_result_t validate_rk8(
    const tuple_t *ct,
    const u32 *v8,
    int num_ct,
    const tuple_pair_t *pairs,
    int num_pairs,
    u32 rk8,
    u8 *key,
    u32 *num) {
  CHECK_BAD_ARGUMENT(ct == NULL);
  CHECK_BAD_ARGUMENT(v8 == NULL);
  CHECK_BAD_ARGUMENT(num_ct < 2);
  CHECK_BAD_ARGUMENT(pairs == NULL);
  CHECK_BAD_ARGUMENT(num_pairs < 1);
  CHECK_BAD_ARGUMENT(rk8 & 0xff000000);
  CHECK_BAD_ARGUMENT(key == NULL);
  CHECK_BAD_ARGUMENT(num == NULL);

  u32 keycount[256] = {0};
  u32 *v7 = NULL;
  u32 llrk8 = inv_rotate_rows(inv_mix_columns(rk8));
  halfloop_result_t err = HALFLOOP_SUCCESS;

  v7 = malloc(sizeof(u32) * num_ct);
  RETURN_IF(v7 == NULL, HALFLOOP_MEMORY_ERROR);

  /* Precalculate v7 states. */
  for (int i = 0; i < num_ct; i++) {
    u32 tw7 = (ct[i].tweak ^ (ct[i].tweak >> 32)) & 0xffffff;
    u32 v = inv_sub_bytes(v8[i] ^ llrk8) ^ tw7;
    v7[i] = inv_rotate_rows(inv_mix_columns(v));
  }

  for (const tuple_pair_t *p = pairs; p < pairs + num_pairs; p++) {
    u32 delta = ((ct[p->a].tweak ^ ct[p->b].tweak) >> 40) & 0xff;
    /* Check for required v7 difference. */
    u32 dv7 = v7[p->a] ^ v7[p->b];
    if ((dv7 & 0xffff) != 0 || (dv7 & 0xff0000) == 0) {
      continue;
    }
    /* Test keys. */
    for (int k = 0; k < 256; k++) {
      u32 a = v7[p->a] >> 16;
      u32 b = v7[p->b] >> 16;
      if ((inv_SBOX[a ^ k] ^ inv_SBOX[b ^ k]) == (u8)delta) {
        keycount[k] += 1;
        /* Select the most commonly indicated key byte. */
        if (keycount[k] > *num) {
          *num = keycount[k];
          *key = (u8)k;
        }
      }
    }
  }

error:
  free(v7);
  return err;
}
