# HALFLOOP-24 ciphertext-only attack

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088257.svg)](https://doi.org/10.5281/zenodo.20088257)

HALFLOOP-24 is a cipher specified in MIL-STD-188-141 and used for encrypting
Automatic Link Etablishment (ALE) frames in the second and third generations of
the ALE standards. This repository contains implementations of the attacks on
HALFLOOP-24 described in **Ciphertext-Only Attack on HALFLOOP-24**.

The attack requires a large number of pairs of ciphertext and tweak. A utility,
`halfloop-generate-data`, that generates random good pairs is provided for
testing.

For details on the requirements of ciphertexts and tweaks, probabilities of
success, and attack times, the reader is referred to the aforementioned paper.

## System requirements

An x86-64 processor with the AVX instruction set. The CUDA implementation
requires a CUDA device with the Pascal, or later, architecture (compute
capability 6.1).

## Dependencies

* [CMake](https://cmake.org/) (build system)
* [CUDA toolkit](https://developer.nvidia.com/cuda/toolkit) (For the CUDA implementation)

## Build

```console
mkdir build
cd build
cmake ..
make
```

By default, CMake will attempt to build the CUDA features if CUDA is detected on
the build system. Use the cmake arguments `-DENABLE_CUDA=ON` and
`-DENABLE_CUDA=OFF` to manually turn on and off CUDA support during the build.

## Run

### halfloop-generate-data

Start by generating test data. `halfloop-generate-data` will generate a random
key and print it to `stderr`. It will then generate a random key and use it to
generate random ALE call messages, printing the corresponding ciphertexts and
tweaks to `stdout`. The generated key, along with some other information, is
printed to `stderr`.

The general usage format is
```console
./halfloop-generate-data <callsign file> <number of bins> <calls per bin>
```
The callsign file contains the callsigns that should be used, one per row. Each
callsign must be three characters long and must only contain the capital letters
A&ndash;Z or numbers 0&ndash;9. The number of bins argument specifies how many
16-minute bins with calls that should be simulated and the calls per bin
arguments specifies how many calls each bin should contain. The following
command generates 26 16-minute bins with 96 calls in each bin.
```console
./halfloop-generate-data ../sweden-callsigns.txt 26 96 > data.txt
Loaded 64 callsigns.
       Key: 790975cad321aacb375a9d28804c00a6
      rk 8: b51166
      rk 9: e4355d
     rk 10: 66421f
Start time: 6 January 07:28:00
 Frequency: 27760200 Hz
```

### halfloop-coa

The actual attacks are implemented by the halfloop-coa program. The general
usage format is
```console
./halfloop-coa [OPTIONS] <filename> <chunk>
./halfloop-coa [OPTIONS]
```
The first version of the command runs the attack. The file indicated by filename
contains a list of 24-bit ciphertexts and 64-bit tweaks. Both must be in
hexadecimal format and separated by a space, with one such pair per line.

The chunk argument is used to specify up to 32 known key bits from round keys 9
and 10 in hexadecimal format. It can be used to speed up the attack when
testing, or to divide the work between different computers. For example, a
single hexadecimal character, such as `f`, fixes only the last four bits of
round key 10, while an argument such as `66421f` will fix the entire round key
10. When specifying 28 or 32 bits, the least significant bits of round key 9
will also be fixed. The following command runs the attack with 32 fixed key
bits. The number of fixed key bits can be varied between 0 and 32.

The following command will run an attack using the data and key generated in the
example above.
```console
./halfloop-coa data.txt 5d66421f
```

#### Command-line arguments

The halfloop-coa command has numerous command line arguments for selecting
the attack algorithm, setting attack parameters, testing, and profiling. Running
halfloop-coa without arguments will show a list of available arguments. A
description of them is also provided here. Some commands, such as the selection
of which algorithm to use, are mutually exclusive. Others, such as listing the
available CUDA devices cannot be combined with an attack.

| Argument     | Description                                                   |
|--------------|---------------------------------------------------------------|
| `-1`         | Use CPU algorithm 1 for the attack.                           |
| `-2`         | Use CPU algorithm 2 for the attack.                           |
| `-3`         | Use CPU algorithm 3 for the attack.                           |
| `-4`         | Use GPU algorithm 2 for the attack. Requires CUDA.            |
| `-5`         | Use GPU algorithm 3 for the attack. Requires CUDA.            |
| `-b`         | Runs performance benchmarks for the bitslice implementations. |
| `-c`         | Sets the probability p<sub>ct</sub> that two randomly selected callsigns will differ only in the least significant byte. Entered as the exponential x in 2<sup>x</sup>, &minus;32 &#8805; x &lt; 0. The probability is used to calculate the τ values and the overall probability of success. |
| `-d`         | Specify CUDA device IDs to use. The provided device list must be a comma-separated list of device IDs. A list of available devices and their IDs can be found by using the `-l` argument. The default is to use all available devices. |
| `-f`         | Skip the brute-force search for the last 48 key bits. The brute force search is always skipped when not using CUDA. |
| `-l`         | Prints a list of available CUDA devices and their IDs.        |
| `-m`         | Set the number of CUDA blocks to use per multiprocessor. Can be used for performance tuning. Default value: 2. |
| `-p`         | Performs only a partial search. This option is used when running the program in a profiler. |
| `-s`         | Set the target probability of success, 0 &lt; p<sub>success</sub> &lt; 1. The default value is 0.5. |
| `-t`         | Set the τ<sub>1</sub> value. Overrides the value calculated from p<sub>ct</sub> and p</sub>success</sub>.|
| `-u`         | Set the τ<sub>2</sub> value. The default is set to achieve p<sub>success</sub> &#8805; 0.99 in the validation steps.|
| `-v`         | Enable verbose output. Prints more information.               |

## License

Copyright &#169; 2022 Marcus Dansarie, Patrick Derbez, Gregor Leander, and Lukas
Stennes.

Copyright &#169; 2025-2026 Marcus Dansarie, Gregor Leander, Shahram Rasoolzadeh,
Lukas Stennes, and Cihangir Tezcan.

This project is licensed under the GNU General Public License — see the
[LICENSE](LICENSE.txt) file for details.
