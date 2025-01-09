CXX=nvc++ 

GCC_HOME=/opt/spack-0.21.1/opt/spack/linux-ubuntu22.04-x86_64_v4/gcc-11.4.0/gcc-13.2.0-7zghxmvntize2wj4v6mujdv4s4huqcco
CUDA_CC=89

ifneq (,$(findstring nvc++,$(CXX)))
	CXXFLAGS=-O3 -fast -mp -std=c++20 --gcc-toolchain=${GCC_HOME} -I../include
	LDFLAGS=
	STDPARFLAGS_GPU=-mp -stdpar=gpu -gpu=cc${CUDA_CC} -gpu=mem:managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll
	STDPARFLAGS_CPU=-mp -stdpar=multicore
all: dslash_pmr_test_gpu dslash_pmr_test_x86
else
	echo "Unknown Compiler"
	exit
endif


dslash_pmr_test_gpu: dslash_pmr_test.o 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(STDPARFLAGS_GPU) 

dslash_pmr_test_x86: dslash_pmr_test_x86.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(STDPARFLAGS_CPU) 

dslash_pmr_test.o: dslash_pmr_test.cpp dslash_pmr_test.h
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c -o $@ $< 

dslash_pmr_test_x86.o: dslash_pmr_test.cpp dslash_pmr_test.h
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -c -o $@ $< 

.PHONY: clean
clean:
	-rm -f core *.o *_gpu *_x86
