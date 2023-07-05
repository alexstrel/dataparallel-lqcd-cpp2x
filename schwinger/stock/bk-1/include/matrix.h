#pragma once

#include <cstdio>
#include <iostream>
#include <complex>

//#include <register_traits.h>
//#include <float_vector.h>
//#include <complex_quda.h>
//#include <math_helper.cuh>

namespace hisq{

  template <typename T> constexpr bool is_nan(T x) { return x != x; }
  
  template <typename T> struct RealType { };
  //
  template <> struct RealType<double> { using type = double; };
  template <> struct RealType<float > { using type = float;  };
  template <> struct RealType<short > { using type = short;  };
  template <> struct RealType<int8_t> { using type = int8_t; };  
  //
  template <> struct RealType<std::complex<double>> { using type = double; };
  template <> struct RealType<std::complex<float> > { using type = float;  };
  template <> struct RealType<std::complex<short> > { using type = short;  };
  template <> struct RealType<std::complex<int8_t>> { using type = int8_t; };

  template<ArithmeticTp T, int N, int bSize = 1>
  class Matrix {
      using real = typename RealType<T>::type;

    private:
        inline int index(int i, int j, int b) const { return (i*N + j)*bSize + b; }
        inline int index(int l, int b)        const { return (l*bSize + b);         }

      public:
        std::array<std::complex<T>, N*N*bSize> data;

        constexpr int rows() const { return N; }
        constexpr int cols() const { return N; }
        constexpr int batch_size() const { return bSize; }
        constexpr int size()       const { return N*N;   }
        
        static constexpr int Rows() { return N; }
        static constexpr int Cols() { return N; }

        inline Matrix() = default;

        Matrix(const Matrix<T, N, bSize> &)            = default;
        Matrix(Matrix<T, N, bSize> &&)                 = default;
        Matrix &operator=(const Matrix<T, N, bSize> &) = default;
        Matrix &operator=(Matrix<T, N, bSize> &&)      = default;

        template <ArithmeticTp U> inline Matrix(const Matrix<U, N, bSize> &a)
        {
#pragma unroll
          for (int i = 0; i < data.size(); i++) data[i] = a.data[i];
        }

        inline Matrix(const std::complex<T> data_[])
        {
#pragma unroll
	  for (int i=0; i<data.size(); i++) data[i] = data_[i];
        }

        inline auto const & operator()(int i, int j, int b) const {
          return data[index(i,j,b)];
        }
        
        inline auto & operator()(int i, int j, int b) {
          return data[index(i,j,b)];
        }        
        
        inline auto const & operator()(int l, int b) const {
          int j = l % N;
          int k = l / N;
          return data[index(j,k,b)];
        }        

        inline auto& operator()(int l, int b) {
          int j = l % N;
          int k = l / N;
          return data[index(j,k,b)];
        }

	template<ArithmeticTp U>
	  inline void operator=(const Matrix<U,N,bSize> & b) {
#pragma unroll
	  for (int i=0; i<data.size(); i++) data[i] = b.data[i];
	}

  };

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize>
  inline Mat<T,N,bSize> operator+(const Mat<T,N,bSize> & a, const Mat<T,N,bSize> & b)
  {
      Mat<T,N,bSize> result;
#pragma unroll      
      for (int i = 0; i < N*N; i++) {
#pragma unroll
        for (int j = 0; j < bSize; j++) result.data[index(i,j)] = a.data[index(i,j)] + b.data[index(i,j)];
      }
      return result;
  }


  template< template< ArithmeticTp,int,int> typename Mat,  ArithmeticTp T, int N, int bSize>
  inline Mat<T,N,bSize> operator+=(Mat<T,N,bSize> & a, const Mat<T,N,bSize> & b)
  {
#pragma unroll      
      for (int i = 0; i < N*N; i++) {  
#pragma unroll
        for (int j = 0; j < bSize; j++) a.data[index(i,j)] += b.data[index(i,j)];
      }
      return a;
  }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize>//is it just arithmetic type?
  inline Mat<T,N,bSize> operator+=(Mat<T,N,bSize> & a, const std::complex<T> & b)
  {
#pragma unroll      
      for (int i = 0; i < a.rows(); i++) {   
#pragma unroll
        for (int j = 0; j < bSize; j++) a(i, i, j) += b;
      }
      return a;
  }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize>
    inline Mat<T,N,bSize> operator-=(Mat<T,N,bSize> & a, const Mat<T,N,bSize> & b)
    {
#pragma unroll      
      for (int i = 0; i < N*N; i++) {     
#pragma unroll
        for (int j = 0; j < bSize; j++) a.data[index(i,j)] -= b.data[index(i,j)];
      }
      return a;
    }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize>
    inline Mat<T,N,bSize> operator-(const Mat<T,N,bSize> & a, const Mat<T,N,bSize> & b)
    {
      Mat<T,N,bSize> result;
#pragma unroll      
      for (int i = 0; i < N*N; i++) {      
#pragma unroll
        for (int j = 0; j < bSize; j++) result.data[index(i,j)] = a.data[index(i,j)] - b.data[index(i,j)];
      }
      return result;
    }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize, typename S>
    inline Mat<T,N,bSize> operator*(const S & scalar, const Mat<T,N,bSize> & a){
      Mat<T,N,bSize> result;
#pragma unroll      
      for (int i = 0; i < N*N; i++) {      
#pragma unroll
        for (int j = 0; j < bSize; j++) result.data[index(i,j)] = scalar * a.data[index(i,j)];
      }
      return result;
    }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize, typename S>
    inline Mat<T,N,bSize> operator*(const Mat<T,N,bSize> & a, const S & scalar){
      return scalar*a;
    }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize, typename S>
    inline Mat<T,N,bSize> operator *=(Mat<T,N,bSize> & a, const S & scalar){
      a = scalar*a;
      return a;
    }

  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize>
    inline Mat<T,N,bSize> operator-(const Mat<T,N,bSize> & a){
      Mat<T,N,bSize> result;
#pragma unroll      
      for (int i = 0; i < N*N; i++) {      
#pragma unroll
        for (int j = 0; j < bSize; j++) result.data[index(i,j)] = -a.data[index(i,j)];
      }
      return result;
    }


  /**
     @brief Specialization of complex matrix multiplication that will issue optimal fma instructions
   */
  template< template<ArithmeticTp,int,int> typename Mat, ArithmeticTp T, int N, int bSize>
    inline Mat<T,N,bSize> operator*(const Mat<T,N,bSize> &a, const Mat<T,N,bSize> &b)
    {
      Mat<T,N,bSize> result;

#pragma unroll            
      for (int i=0; i<N; i++) {
#pragma unroll
	for (int k=0; k<N; k++) {
#pragma unroll
          for (int l = 0; l < bSize; l++) {
            result(i, k, l).real(a(i, 0, l).real() * b(0, k, l).real());
            result(i, k, l).real(result(i, k, l).real() - a(i, 0, l).imag() * b(0, k, l).imag());
            result(i, k, l).imag(a(i, 0, l).real() * b(0, k, l).imag());
            result(i, k, l).imag(result(i, k, l).imag() + a(i, 0, l).imag() * b(0, k, l).real());
          }
#pragma unroll
	  for (int j=1; j<N; j++) {
#pragma unroll
            for (int l = 0; l < bSize; l++) {
              result(i, k, l).real(result(i, k, l).real() + a(i, j, l).real() * b(j, k, l).real());
              result(i, k. l).real(result(i, k, l).real() - a(i, j, l).imag() * b(j, k, l).imag());
              result(i, k, l).imag(result(i, k, l).imag() + a(i, j, l).real() * b(j, k, l).imag());
              result(i, k, l).imag(result(i, k, l).imag() + a(i, j, l).imag() * b(j, k, l).real());
            }
          }
	}
      }
      return result;
    }




  template<ArithmeticTp T, int N, int bSize>
  inline void setIdentity(Matrix<T,N,bSize>* m){
#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int l = 0; l < bSize; l++) {  
          (*m)(i,i,l) = 1;
        }
#pragma unroll
        for (int j=i+1; j<N; ++j){
#pragma unroll
          for (int l = 0; l < bSize; l++) {
            (*m)(i,j,l) = (*m)(j,i,l) = 0;
          }
        }
      }
  }


  // Need to write more generic code for this!
  template<ArithmeticTp T, int N, int bSize>
  inline void setZero(Matrix<T,N,bSize>* m){
#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int j=0; j<N; ++j){
#pragma unroll
          for (int l = 0; l < bSize; l++) {
            (*m)(i,j,l) = 0;
          }
        }
      }
    }


  // Matrix and array are very similar
  // Maybe I should factor out the similar
  // code. However, I want to make sure that
  // the compiler knows to store the
  // data elements in registers, so I won't do
  // it right now.
  template<ArithmeticTp T, int N, int bSize = 1>
    class Array
    {
      private:
        std::array<std::complex<T>, N*bSize> data;

      public:
      
        static constexpr int size() { return N; }
        // access function
        inline auto const & operator[](int i) const{ return data[i]; }
        inline auto const & operator()(int i, int j) const{ return data[i*bSize+j]; }

        // assignment function
        inline auto & operator[](int i){ return data[i]; }
        inline auto & operator()(int i, int j) { return data[i*bSize+j]; }        
    };


  template<ArithmeticTp T, int N, int bSize>
    inline void copyColumn(const Matrix<T,N, bSize>& m, int c, Array<T,N, bSize>& a){   
#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int l = 0; l < bSize; l++) { 
          a(i,l) = m(i,c,l); // c is the column index
        }
      }
    }
   //internal data format 
   template <ArithmeticTp Float> using su3_matrix_  = Matrix<Float, 3>;
   template <ArithmeticTp Float> using su3_vector_  = Array<Float, 3>;      
   //external data format
   //
   template <ArithmeticTp Float, int N, int bSize> using sun_matrix  = Matrix<Float, N, bSize>;
   template <ArithmeticTp Float, int N, int bSize> using sun_vector  = Array<Float, N, bSize>;   
   
} // end namespace sun
