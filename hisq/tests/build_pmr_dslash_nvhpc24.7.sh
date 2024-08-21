#GCC_HOME=/opt/spack-0.19.0/opt/spack/linux-ubuntu22.04-haswell/gcc-12.1.0/gcc-12.2.0-ogs53w6jn6kghfvjoaeqa3jmblrgrsne

GCC_HOME=/scratch/astrel/Install/spack-0.22.0/opt/spack/linux-ubuntu22.04-broadwell/gcc-12.3.0/gcc-13.2.0-lnntvms444y6qspof6tt63s7zibdocuw

nvc++ -O3 -std=c++20 --gcc-toolchain=${GCC_HOME} -stdpar=gpu -gpu=cc70 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll -I./ -I../include -o dslash_pmr_test.exe dslash_pmr_test.cpp

#nvc++ -O3 -std=c++20 --gcc-toolchain=${GCC_HOME} -stdpar=multicore -I./ -I../include -o dslash_cpu_pmr_test.exe dslash_pmr_test.cpp

