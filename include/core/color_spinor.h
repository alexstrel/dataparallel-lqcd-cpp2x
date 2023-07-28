#pragma once

#include <cstdio>
#include <iostream>
#include <complex>
//
#include <core/color_matrix.h>


namespace impl{

  template<int nSpin> concept StaggeredSpinor     = (nSpin == 1);// or nSpin == invalid_spin);

  template<int nSpin> concept TwoComponentSpinor  = (nSpin == 2);
  //
  template<int nSpin> concept FourComponentSpinor = (nSpin == 4);  

  template <ComplexTp T, int Nc, int Ns, int bSize>
    class ColorSpinor {
      public:        
        //
        static constexpr int nColor = Nc;        
        static constexpr int nSpin  = Ns == invalid_spin ? 1: Ns;                
        static constexpr int bsize  = bSize;                        
        static constexpr int N      = nColor*nSpin;        
             
        using indx_type = std::size_t;      
        //              
        using View3D   =  stdex::mdspan<T,       stdex::extents<indx_type, nColor, nSpin, bSize>, stdex::layout_right, stdex::default_accessor<T>>;
        using CView3D  =  stdex::mdspan<const T, stdex::extents<indx_type, nColor, nSpin, bSize>, stdex::layout_right, stdex::default_accessor<const T>>;

        using View2D   =  stdex::mdspan<T,       stdex::extents<indx_type, N, bSize>, stdex::layout_right, stdex::default_accessor<T>>;
        using CView2D  =  stdex::mdspan<const T, stdex::extents<indx_type, N, bSize>, stdex::layout_right, stdex::default_accessor<const T>>;      
        //
        std::array<T, N*bSize> data = {};//void

        inline ColorSpinor() = default;
        
        inline ColorSpinor(const std::array<T, N*bSize> &data_) : data{data_} { } 
        
        inline const T& operator()(int idx) const { return data[idx]; } //FIXME : flat accessor should not be used 
        inline       T& operator()(int idx)       { return data[idx]; } //FIXME : flat accessor should not be used                              

        inline ColorSpinor operator-() const {
          ColorSpinor<T, Nc, Ns, bSize> a;
#pragma unroll
          for (int i = 0; i < data.size(); i++) { a.data[i] = -data[i]; }
          return a;
        }

        inline ColorSpinor& operator+=(const ColorSpinor<T, Nc, Ns, bSize> &a) {
#pragma unroll
          for (int i = 0; i < data.size(); i++) { data[i] = data[i] + a.data[i]; }
          return *this;
        }

        inline ColorSpinor &operator*=(const ComplexTp auto &a) {
#pragma unroll
          for (int i = 0; i < data.size(); i++) { data[i] = data[i]*a; }
          return *this;
        }

        inline ColorSpinor &operator*=(const FloatTp auto &a) {
#pragma unroll
          for (int i = 0; i < data.size(); i++) { data[i] = T{data[i].real()*a, data[i].imag()*a}; }
          return *this;
        }


        inline ColorSpinor &operator-=(const ColorSpinor<T, Nc, Ns, bSize> &a) {
          if (this != &a) {
#pragma unroll
            for (int i = 0; i < data.size(); i++) { data[i] = data[i] - a.data[i]; }
          }
          return *this;
        }
            
        inline decltype(auto) spincolor_cview() const {
          return CView3D(data.data());
        }

        inline decltype(auto) spincolor_view() {
          return View3D(data.data());
        }

        inline decltype(auto) flat_cview() const {
          return CView2D(data.data());
        }

        inline decltype(auto) flat_view() {
          return View2D(data.data());
        }

        template <int Ns_ = Ns> requires StaggeredSpinor<Ns_>
        inline decltype(auto) color_cview() const {
          return CView2D(data.data());
        }

        template <int Ns_ = Ns> requires StaggeredSpinor<Ns_>
        inline decltype(auto) color_view() {
          return View2D(data.data());
        }        
        
        template<int sign, int Ns_ = Ns> requires TwoComponentSpinor<Ns_>
        inline decltype(auto) project(int dir) const {

          using data_tp = typename T::value_type; 
      
          ColorSpinor<T, Nc, 2, bSize> proj;	    

          auto I = [](auto a){ return T{-a.imag(), a.real()};};

          const auto in = CView3D{data.data()}; 
          auto proj_    = proj.spincolor_view();           

          if constexpr (sign == +1) {
            switch (dir) {
              case 0 :
#pragma unroll              
              for(int c = 0; c < Nc; c++ ) {
#pragma unroll              
                for(int b = 0; b < bSize; b++) {
                  proj_(c, 0, b) = in(c, 0, b) - in(c, 1, b);
                  proj_(c, 1, b) = in(c, 1, b) - in(c, 0, b);
                }
              }

              break;

              case 1 :
#pragma unroll              
              for(int c = 0; c < Nc; c++ ) {
#pragma unroll              
                for(int b = 0; b < bSize; b++) {
                  proj_(c, 0, b) = in(c, 0, b) + I(in(c, 1, b));
                  proj_(c, 1, b) = in(c, 1, b) - I(in(c, 0, b));
                }
              }              
              break;
             }
           } else if constexpr (sign == -1) {	      
            switch (dir) {
              case 0 :
#pragma unroll              
              for(int c = 0; c < Nc; c++ ) {
#pragma unroll              
                for(int b = 0; b < bSize; b++) {
                  proj_(c, 0, b) = in(c, 0, b) + in(c, 1, b);
                  proj_(c, 1, b) = in(c, 1, b) + in(c, 0, b);
                }
              }

              break;

              case 1 :
#pragma unroll              
              for(int c = 0; c < Nc; c++ ) {
#pragma unroll              
                for(int b = 0; b < bSize; b++) {
                  proj_(c, 0, b) = in(c, 0, b) - I(in(c, 1, b));
                  proj_(c, 1, b) = in(c, 1, b) + I(in(c, 0, b));
                }
              }              
              break;
             }	      
           }
           //
           return proj;
         }        
        
    };
  
  /**
     @brief Compute the matrix-vector product y = A * x
     @param[in] A Input matrix
     @param[in] x Input vector
     @return The vector A * x
  */
  template<ComplexTp T, int Nc, int Ns, int bSize = 1>
    inline ColorSpinor<T,Nc,Ns,bSize> operator*(const ColorMatrix<T,Nc,bSize> &A, const ColorSpinor<T,Nc,Ns,bSize> &x) {
      //
      using data_tp = T::value_type;
      
      constexpr int nColor = ColorSpinor<T,Nc,Ns,bSize>::nColor;
      constexpr int nSpin  = ColorSpinor<T,Nc,Ns,bSize>::nSpin;      
      
      ColorSpinor<T,Nc,Ns,bSize> y;
      //
      auto y_ = y.spincolor_view();
      //
      const auto A_ = A.cview();
      const auto x_ = x.spincolor_cview();    
      //
#pragma unroll
      for (int c = 0; c < nColor; c++) {
#pragma unroll
        for (int s = 0; s < nSpin; s++) { //
          data_tp re;
          data_tp im;
#pragma unroll        
          for (int b = 0; b < bSize; b++) {
	    re  = A_(c,0,b).real() * x_(0, s, b).real();
	    re -= A_(c,0,b).imag() * x_(0, s, b).imag();
	    im  = A_(c,0,b).real() * x_(0, s, b).imag();
	    im += A_(c,0,b).imag() * x_(0, s, b).real();
	    y_(c,s,b)   = T{re, im};
	  }
#pragma unroll
          for (int cc = 1; cc < nColor; cc++) {
#pragma unroll        
            for (int b = 0; b < bSize; b++) {
	      re += A_(c,cc,b).real() * x_(cc, s, b).real();
	      re -= A_(c,cc,b).imag() * x_(cc, s, b).imag();
	      im += A_(c,cc,b).real() * x_(cc, s, b).imag();
	      im += A_(c,cc,b).imag() * x_(cc, s, b).real();
	      y_(c,s,b) += T{re, im};
	    }	  
	  }	  
        }
      }

      return y;
    }
  
   /**
     @brief caxpy operation on ColorSpinor objects
     @param[in] a complex scalar
     @param[in] x Vector that is scaled
     @param[in,out] y Accumulation vector
  */
  template <ComplexTp T, int Nc, int Ns, int bSize = 1>
  inline void caxpy(const T &a, const ColorSpinor<T, Nc, Ns, bSize> &x, ColorSpinor<T, Nc, Ns, bSize> &y)
  {
    const auto x_ = x.flat_cview();      
    auto       y_ = y.flat_view();
    //
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int b = 0; b < bSize; b++) {    
        y_(i,b).real( a.real() * x_(i,b).real() + y_(i,b).real());
        y_(i,b).real(-a.imag() * x_(i,b).imag() + y_(i,b).real());
        y_(i,b).imag( a.imag() * x_(i,b).real() + y_(i,b).imag());
        y_(i,b).imag( a.real() * x_(i,b).imag() + y_(i,b).imag());
      }
    }
  }
  
  /**
     @brief ColorSpinor addition operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<ComplexTp T, int Nc, int Ns, int bSize = 1>
  inline ColorSpinor<T,Nc,Ns,bSize> operator+(const ColorSpinor<T,Nc,Ns,bSize> &x, const ColorSpinor<T,Nc,Ns,bSize> &y) {

    ColorSpinor<T,Nc,Ns,bSize> z;

          auto z_ = z.flat_view();    

    const auto x_ = x.flat_cview();    
    const auto y_ = y.flat_cview();

#pragma unroll
    for (int i=0; i<N; i++) {
#pragma unroll      
      for (int b=0; b<bSize; b++) {
	z_(i,b) = x_(i,b) + y_(i,b);
      }
    }

    return z;
  }

  /**
     @brief ColorSpinor subtraction operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<ComplexTp T, int Nc, int Ns, int bSize = 1>
    inline ColorSpinor<T,Nc,Ns,bSize> operator-(const ColorSpinor<T,Nc,Ns,bSize> &x, const ColorSpinor<T,Nc,Ns,bSize> &y) {

    ColorSpinor<T,N,bSize> z;

    const auto y_ = y.flat_cview();
    const auto x_ = x.flat_cview();    

    auto z_ = z.flat_view();    

#pragma unroll
    for (int i=0; i<N; i++) {
#pragma unroll      
      for (int b=0; b<bSize; b++) {
	z_(i,b) = x_(i,b) - y_(i,b);
      }
    }

    return z;
  }

  /**
     @brief Compute the scalar-vector product y = a * x
     @param[in] a Input scalar
     @param[in] x Input vector
     @return The vector a * x
  */
  template<ArithmeticTp S, ComplexTp T, int Nc, int Ns, int bSize = 1> 
    inline ColorSpinor<T,Nc,Ns,bSize> operator*(const S &a, const ColorSpinor<T,Nc,Ns,bSize> &x) {

    ColorSpinor<T,Nc,Ns,bSize> y;

    const auto x_ = x.flat_cview();    
          auto y_ = y.flat_view();    

#pragma unroll
    for (int i=0; i<N; i++) {
#pragma unroll      
      for (int b=0; b<bSize; b++) {
	y_(i,b) = a*x_(i,b);
      }
    }

    return y;
  }
  
   
} // 
