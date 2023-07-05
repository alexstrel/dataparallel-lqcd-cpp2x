GCC_HOME=/opt/spack-0.19.0/opt/spack/linux-ubuntu22.04-haswell/gcc-12.1.0/gcc-12.2.0-ogs53w6jn6kghfvjoaeqa3jmblrgrsne

#GCC_HOME=/opt/spack-0.19.0/opt/spack/linux-ubuntu22.04-haswell/gcc-12.1.0/gcc-13.1.0-zusgewjfb27js37miqopwadjzjke3rtp

nvc++ -O2 -std=c++20 --gcc-toolchain=${GCC_HOME}  -stdpar=gpu -gpu=cc75 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll -I./ -I../include -o block_spinor_test.exe block_spinor_test.cpp

