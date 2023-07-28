#pragma once

#include <cstdio>
#include <iostream>
#include <complex>

#include <core/common.h>


namespace impl{

  template <typename T> constexpr bool is_nan(T x) { return x != x; }
  //
  template<ComplexTp T, int Nc, int bSize = 1>
  class ColorMatrix {
    public:  
      using real      = typename T::value_type;
      using indx_type = std::size_t;
      //
      consteval int color_size()  const { return Nc*Nc; }//note  
      static constexpr int N = Nc * Nc;    
      //
      using View3D   =  stdex::mdspan<T,       stdex::extents<indx_type, Nc, Nc, bSize>, stdex::layout_right, stdex::default_accessor<T>>;
      using CView3D  =  stdex::mdspan<const T, stdex::extents<indx_type, Nc, Nc, bSize>, stdex::layout_right, stdex::default_accessor<const T>>;

      using View2D   =  stdex::mdspan<T,       stdex::extents<indx_type, N, bSize>, stdex::layout_right, stdex::default_accessor<T>>;
      using CView2D  =  stdex::mdspan<const T, stdex::extents<indx_type, N, bSize>, stdex::layout_right, stdex::default_accessor<const T>>;
      
      std::array<T, N*bSize> m;
        
      static constexpr int Rows() { return Nc; }
      static constexpr int Cols() { return Nc; }

      inline ColorMatrix() = default;

      ColorMatrix(const ColorMatrix<T, Nc, bSize> &)            = default;
      ColorMatrix(ColorMatrix<T, Nc, bSize> &&)                 = default;
        
      ColorMatrix &operator=(const ColorMatrix<T, Nc, bSize> &) = default;
      ColorMatrix &operator=(ColorMatrix<T, Nc, bSize> &&)      = default;

      template <ComplexTp U> inline ColorMatrix(const ColorMatrix<U, Nc, bSize> &a) : m{a.m} { }

      inline ColorMatrix(const std::array<T, Nc*Nc*bSize> &m_) : m{m_} { }
        
      inline decltype(auto) cview() const {
        return CView3D(m.data());
      }

      inline decltype(auto) view() {
        return View3D(m.data());
      }

      inline decltype(auto) flat_cview() const {
        return CView2D(m.data());
      }

      inline decltype(auto) flat_view() {
        return View2D(m.data());
      }

      template<ComplexTp U> inline void operator=(const ColorMatrix<U,Nc,bSize> & b) {
#pragma unroll
	for (int i = 0; i < m.size(); i++) m[i] = b.m[i];
      }

  };


//////////////////////////////////////////////////////////////////////////////////////////////

  template< template<ComplexTp, int, int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
  inline Mat<T,Nc,bSize> operator+(const Mat<T,Nc,bSize> & a, const Mat<T,Nc,bSize> & b) {
      Mat<T,Nc,bSize> result;
      //
      const auto a_ = a.flat_cview();
      const auto b_ = b.flat_cview();      
      auto       c_ = result.flat_view();      
      //
#pragma unroll      
      for (int i = 0; i < c_.extent(0); i++) {
#pragma unroll
        for (int j = 0; j < c_.extent(1); j++) c_(i,j) = a_(i,j) + b_(i,j);
      }
      return result;
  }


  template< template< ComplexTp, int, int> typename Mat,  ComplexTp T, int Nc, int bSize = 1>
  inline Mat<T,Nc,bSize> operator+=(Mat<T,Nc,bSize> & a, const Mat<T,Nc,bSize> & b)
  {
      auto       a_ = a.flat_view();
      const auto b_ = b.flat_cview();      
  
#pragma unroll      
      for (int i = 0; i < a_.extent(0); i++) {  
#pragma unroll
        for (int j = 0; j < a_.extent(1); j++) a_(i,j) = a_(i,j) + b_(i,j);
      }
      
      return a;
  }

  template< template<ComplexTp, int, int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
  inline Mat<T,Nc,bSize> operator+=(Mat<T,Nc,bSize> & a, const T & b)
  {
      auto a_ = a.view();  
#pragma unroll      
      for (int i = 0; i < a_.extent(0); i++) {   
#pragma unroll
        for (int j = 0; j < a_.extent(2); j++) a_(i, i, j) = a_(i, i, j) + b;
      }
      //
      return a;
  }

  template< template<ComplexTp,int,int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator-=(Mat<T,Nc,bSize> & a, const Mat<T,Nc,bSize> & b)
    {
      auto       a_ = a.flat_view();
      const auto b_ = b.flat_cview();      
  
#pragma unroll      
      for (int i = 0; i < a_.extent(0); i++) {  
#pragma unroll
        for (int j = 0; j < a_.extent(1); j++) a_(i,j) = a_(i,j) - b_(i,j);
      }
    
      return a;
    }

  template< template<ComplexTp,int,int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator-(const Mat<T,Nc,bSize> & a, const Mat<T,Nc,bSize> & b)
    {
      Mat<T,Nc,bSize> result;
      //
      const auto a_ = a.flat_cview();
      const auto b_ = b.flat_cview();      
      auto       c_ = result.flat_view();      
      //
#pragma unroll      
      for (int i = 0; i < c_.extent(0); i++) {
#pragma unroll
        for (int j = 0; j < c_.extent(1); j++) c_(i,j)= a_(i,j) - b_(i,j);
      }
      return result;
    }

  template<ArithmeticTp S, template<ComplexTp, int, int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator*(const S & scalar, const Mat<T,Nc,bSize> & a){
      Mat<T,Nc,bSize> result;
      //
      const auto a_ = a.flat_cview();
      auto       c_ = result.flat_view();      
      //
#pragma unroll      
      for (int i = 0; i < c_.extent(0); i++) {
#pragma unroll
        for (int j = 0; j < c_.extent(1); j++) c_(i,j)= scalar*a_(i,j);
      }
      return result;      
    }

  template<ArithmeticTp S, template<ComplexTp,int,int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator*(const Mat<T,Nc,bSize> & a, const S & scalar){
      return scalar*a;
    }

  template<ArithmeticTp S, template<ComplexTp, int, int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator *=(Mat<T,Nc,bSize> & a, const S & scalar){
      a = scalar*a;
      return a;
    }

  template< template<ComplexTp,int,int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator-(const Mat<T,Nc,bSize> & a){      
      Mat<T,Nc,bSize> result;
      //
      const auto a_ = a.flat_cview();
      auto       c_ = result.flat_view();      
      //
#pragma unroll      
      for (int i = 0; i < c_.extent(0); i++) {
#pragma unroll
        for (int j = 0; j < c_.extent(1); j++) c_(i,j)= -a_(i,j);
      }
      return result;
      
    }


  /**
     @brief Specialization of complex matrix multiplication that will issue optimal fma instructions
   */
  template< template<ComplexTp, int, int> typename Mat, ComplexTp T, int Nc, int bSize = 1>
    inline Mat<T,Nc,bSize> operator*(const Mat<T,Nc,bSize> &a, const Mat<T,Nc,bSize> &b) {
      Mat<T,Nc,bSize> result;

      const auto a_ = a.cview();
      const auto b_ = b.cview();      
      auto       c_ = result.view();      

#pragma unroll            
      for (int i=0; i<Nc; i++) {
#pragma unroll
	for (int k=0; k<Nc; k++) {
#pragma unroll
          for (int l = 0; l < bSize; l++) {
            c_(i, k, l).real(a_(i, 0, l).real() * b_(0, k, l).real());
            c_(i, k, l).real(c_(i, k, l).real() - a_(i, 0, l).imag() * b_(0, k, l).imag());
            c_(i, k, l).imag(a_(i, 0, l).real() * b_(0, k, l).imag());
            c_(i, k, l).imag(c_(i, k, l).imag() + a_(i, 0, l).imag() * b_(0, k, l).real());
          }
#pragma unroll
	  for (int j=1; j<Nc; j++) {
#pragma unroll
            for (int l = 0; l < bSize; l++) {
              c_(i, k, l).real(c_(i, k, l).real() + a_(i, j, l).real() * b_(j, k, l).real());
              c_(i, k. l).real(c_(i, k, l).real() - a_(i, j, l).imag() * b_(j, k, l).imag());
              c_(i, k, l).imag(c_(i, k, l).imag() + a_(i, j, l).real() * b_(j, k, l).imag());
              c_(i, k, l).imag(c_(i, k, l).imag() + a_(i, j, l).imag() * b_(j, k, l).real());
            }
          }
	}
      }
      return result;
    }

  template<ComplexTp T, int Nc, int bSize = 1> 
  inline ColorMatrix<T,Nc,bSize> conj(const ColorMatrix<T,Nc,bSize> & other){
      ColorMatrix<T,Nc,bSize> result;

      const auto other_  = other.cview();      
      auto       result_ = result.view();      
      
#pragma unroll
      for (int i=0; i<Nc; ++i){
#pragma unroll
        for (int j=0; j<Nc; ++j){
#pragma unroll
          for (int b=0; b<bSize; ++b){
            result_(i,j,b) = conj( other_(j,i,b) );
          }
        }
      }
      return result;
    }


  template<ComplexTp T, int Nc, int bSize = 1>
  inline void setIdentity(ColorMatrix<T,Nc,bSize> &m){
      auto m_ = m.view();        
#pragma unroll
      for (int i=0; i<Nc; ++i){
#pragma unroll
        for (int l = 0; l < bSize; l++) {  
          m_(i,i,l) = 1;
        }
#pragma unroll
        for (int j=i+1; j<Nc; ++j){
#pragma unroll
          for (int l = 0; l < bSize; l++) {
            m_(i,j,l) = m_(j,i,l) = 0;
          }
        }
      }
    }


  // Need to write more generic code for this!
  template<ComplexTp T, int Nc, int bSize = 1>
  inline void setZero(ColorMatrix<T,Nc,bSize> &m){
      auto m_ = m.view();          
#pragma unroll
      for (int i=0; i<Nc; ++i){
#pragma unroll
        for (int j=0; j<Nc; ++j){
#pragma unroll
          for (int l = 0; l < bSize; l++) {
            m_(i,j,l) = 0;
          }
        }
      }
    }

  template<ComplexTp T, int N, int bSize = 1>
  class Vector{
  
      using indx_type = std::size_t;
      
      using View2D   =  stdex::mdspan<T,       stdex::extents<indx_type, N, bSize>, stdex::layout_right, stdex::default_accessor<T>>;
      using CView2D  =  stdex::mdspan<const T, stdex::extents<indx_type, N, bSize>, stdex::layout_right, stdex::default_accessor<const T>>;
  
      private:
        std::array<T, N*bSize> v;

      public:
        // access function
        inline auto const & operator[](int i) const{ return v[i]; }
        // assignment function
        inline auto & operator[](int i){ return v[i]; }
        
        inline decltype(auto) cview() const {
          return CView2D(v.data());
        }

        inline decltype(auto) view() {
          return View2D(v.data());
        }        
    };


  template<ComplexTp T, int Nc, int bSize = 1>
    inline void copyColumn(const ColorMatrix<T,Nc, bSize>& m, int c, Vector<T,Nc, bSize>& a){   
      const auto m_ = m.cview();
      auto a_       = a.view();      
#pragma unroll
      for (int i=0; i<Nc; ++i){
#pragma unroll
        for (int l = 0; l < bSize; l++) { 
          a_(i,l) = m_(i,c,l); // c is the column index
        }
      }
    } 
   
} // 
