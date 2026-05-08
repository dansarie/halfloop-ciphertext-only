# HALFLOOP-24 ciphertext-only attack

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

HALFLOOP-24 is a cipher specified in MIL-STD-188-141 and used for encrypting
Automatic Link Etablishment (ALE) frames in the second and third generations of
the ALE standards. This repository contains implementations of the attacks on
HALFLOOP-24 described in **Ciphertext-Only Attack on HALFLOOP-24**.

The attack requires a large number of pairs of ciphertext and tweak. A utility,
halfloop-generate-data, that generates random good pairs is provided for
testing.

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

## Run

Start by generating test data. `halfloop-generate-data` will generate a random
key and print it to stderr. It will then generate a random key and use it to
generate random ALE call messages, printing the corresponding ciphertexts and
tweaks to stdout. The generated key, along with some other information, is
printed to stderr. The following command generates 26 16-minute bins with 96
calls in each bin.
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

The following command runs the attack with 32 fixed key bits. The number of
fixed key bits can be varied between 0 and 32.
```console
./halfloop-coa data.txt 5d66421f
```
The halfloop-coa command has a number of command line arguments for selecting
the attack algorithm, setting attack parameters, testing, and profiling. Run
halfloop-coa without arguments to show a list of available arguments.
