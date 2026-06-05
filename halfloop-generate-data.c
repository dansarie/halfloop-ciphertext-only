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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "halfloop-common.h"

/**
 * Tests if a character is a valid callsign character.
 * @param c a character.
 * @return true of c is a valid callsign character.
 */
static bool is_valid_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

int main(int argc, char *argv[]) {
  int num_bins = 0;
  int num_calls = 0;
  hlkey key = {0};
  FILE *callsign_file = NULL;
  char *callsigns = NULL;
  int num_callsigns = 0;
  int alloc_callsigns = 0;
  tweak_t tweak = {0};
  u32 rk[11] = {0};
  halfloop_result_t err = HALFLOOP_SUCCESS;

  if (argc != 4) {
    fprintf(
        stderr,
        "Usage: %s <callsign file> <number of bins> <calls per bin>\n\n",
        argv[0]);
    return HALFLOOP_BAD_ARGUMENT;
  }

  num_bins = atoi(argv[2]);
  if (num_bins <= 0) {
    fprintf(
        stderr,
        "Bad number of bins: %d\n",
        num_bins);
    return HALFLOOP_BAD_ARGUMENT;
  }

  num_calls = atoi(argv[3]);
  if (num_calls <= 1 || num_calls > 1024) {
    fprintf(
        stderr,
        "Bad number of calls per bin: %d\n",
        num_calls);
    return HALFLOOP_BAD_ARGUMENT;
  }

  callsign_file = fopen(argv[1], "r");
  if (callsign_file == NULL) {
    fprintf(
        stderr,
        "Error when opening \"%s\": %s\n",
        argv[1],
        strerror(errno));
    return HALFLOOP_FILE_ERROR;
  }

  while (!feof(callsign_file)) {
    if (num_callsigns == alloc_callsigns) {
      alloc_callsigns += 100;
      char *tmp = realloc(callsigns, alloc_callsigns * 3);
      RETURN_IF(tmp == NULL, HALFLOOP_MEMORY_ERROR);
      callsigns = tmp;
    }
    char *cs = callsigns + num_callsigns * 3;
    if (fscanf(callsign_file, "%c%c%c\n", cs, cs + 1, cs + 2) == 3) {
      if (!(
          is_valid_char(cs[0])
          && is_valid_char(cs[1])
          && is_valid_char(cs[2]))) {
        fprintf(stderr, "Ignoring callsign with invalid character.\n");
      } else {
        num_callsigns += 1;
      }
    } else {
      int c;
      while ((c = fgetc(callsign_file)) != '\n' && c != EOF) {
        /* Empty. */
      }
    }
  }
  fclose(callsign_file);
  callsign_file = NULL;
  fprintf(stderr, "Loaded %d callsigns.\n", num_callsigns);

  RETURN_ON_ERROR(init_halfloop());
  RETURN_ON_ERROR(test_halfloop());

  RETURN_ON_ERROR(random_bytes(&key, sizeof(hlkey)));
  RETURN_ON_ERROR(random_bytes(&tweak, sizeof(tweak)));
  RETURN_ON_ERROR(key_schedule(rk, key, 0));
  tweak.month = (ABS(tweak.month) % 12) + 1;
  int days[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  tweak.day = (ABS(tweak.day) % days[tweak.month - 1]) + 1;
  tweak.coarse_time = (ABS(tweak.coarse_time) % 90) * 16;
  tweak.fine_time = 0;
  tweak.word = 0;
  tweak.zero = 0;
  tweak.frequency = (ABS(tweak.frequency) % 270000) * 100 + 3000000;

  fprintf(stderr, "       Key: %016" PRIx64 "%016" PRIx64 "\n", key.hi, key.lo);
  fprintf(stderr, "      rk 8: %06x\n", rk[8]);
  fprintf(stderr, "      rk 9: %06x\n", rk[9]);
  fprintf(stderr, "     rk 10: %06x\n", rk[10]);
  char *month[] = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
  };
  fprintf(
      stderr,
      "Start time: %d %s %02d:%02d:00\n",
      tweak.day,
      month[tweak.month - 1],
      tweak.coarse_time / 60,
      tweak.coarse_time % 60);
  fprintf(stderr, " Frequency: %d Hz\n", tweak.frequency);

  for (int bin = 0; bin < num_bins; bin++) {
    char slots[1024] = {0};
    for (int call = 0; call < num_calls; call++) {

      /* Randomly pick a second in the bin, without replacement. */
      u16 slot = 0;
      do {
        RETURN_ON_ERROR(random_bytes(&slot, sizeof(u16)));
        slot &= 0x3ff;
      } while (slots[slot] != 0 || (slot & 0x3f) >= 60);
      slots[slot] = 1;

      tweak.coarse_time &= ~0xf;
      tweak.coarse_time |= slot >> 6;
      tweak.fine_time = slot & 0x3f;
      tweak.word = 0;

      u32 randnum = 0;
      RETURN_ON_ERROR(random_bytes(&randnum, sizeof(u32)));
      char *cs = callsigns + (randnum % num_callsigns) * 3;
      u32 pt = 0x400000 | (((u32)cs[0]) << 14) | (((u32)cs[1]) << 7) | cs[2];
      u32 ct = 0;
      u64 tw = 0;
      RETURN_ON_ERROR(create_tweak(tweak, &tw));
      RETURN_ON_ERROR(halfloop_encrypt(pt, key, tw, &ct));
      printf("%06x %016" PRIx64 "\n", ct, tw);
      tweak.word = 1;
      RETURN_ON_ERROR(create_tweak(tweak, &tw));
      RETURN_ON_ERROR(halfloop_encrypt(pt, key, tw, &ct));
      printf("%06x %016" PRIx64 "\n", ct, tw);
    }
    tweak.coarse_time &= ~0xf;
    tweak.coarse_time += 16;
    if (tweak.coarse_time >= 1440) {
      tweak.coarse_time = 0;
      tweak.day += 1;
      if (tweak.day > days[tweak.month - 1]) {
        tweak.day = 1;
        tweak.month += 1;
        if (tweak.month > 12) {
          tweak.month = 1;
        }
      }
    }
  }

error:
  if (callsign_file != NULL) {
    fclose(callsign_file);
  }
  free(callsigns);
  return err;
}
