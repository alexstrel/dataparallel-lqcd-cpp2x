GCC_HOME=/opt/spack-0.20.0/opt/spack/linux-ubuntu22.04-x86_64_v4/gcc-11.3.0/gcc-12.3.0-6ottbe2opj4brgf6s7jz5ierwevo2f2a

nvc++ -O3 -std=c++20 --gcc-toolchain=${GCC_HOME} -stdpar=multicore -tp=skylake -mfma -mavx512f -I./ -I../include -I../../../include -o dslash_pmr_test.exe dslash_pmr_test.cpp

