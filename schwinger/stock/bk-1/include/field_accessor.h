#pragma once

#include <common.h>

template <int spin_dof, int mu>
class FieldAccessorArgs{
  public:
    static constexpr int spin_dof_ = spin_dof;

    const std::tuple<std::array<int, D>, std::array<int, D>> params;

    FieldArgs(const std::array<int, D> &dims) :
	params([d=dims]()->decltype(auto) {
                std::array<int, D> nm1{d[0]-1};		
                std::array<int, D> offsets{d[0]};//Nx, NxNy, NxNyNz, ... etc
                //
                for(int i = 1; i < D; i++) {
		  nm1[i]     = d[i] - 1;
		  offsets[i] = offsets[i-1]*d[i];

                } return std::tie(nm1, offsets);} ()) {}

    inline decltype(auto) get_rangem1() const { return std::get<0>(params);}
    inline decltype(auto) get_offsets() const { return std::get<1>(params);}
    
};

template <ArithmeticTp T, typename Arg>
class FieldAccessor{
  public:
    static constexpr int D = Arg::Dims;
  private :
    T *v;//no allocation

    const Arg& args;

  public :
    FieldAccessor(std::vector<T> &latt, const Arg &args) :
        v(latt.data()), 
        args(args)
	{ }
        
    void Set(T* in) {v = in;}  
    T* Get() {return v;} 
    
    template <Shift shift>
    inline int get_shift_dir() const {

      if constexpr        (shift == Shift::ShiftXp1) {
         return 0;
      } else if constexpr (shift == Shift::ShiftXm1) {
         return 1;
      } else if constexpr (shift == Shift::ShiftYp1) {
         return 2;
      } else if constexpr (shift == Shift::ShiftYm1) {
         return 3;
      } else if constexpr (shift == Shift::ShiftZp1) {
         return 4;
      } else if constexpr (shift == Shift::ShiftZm1) {
         return 5;
      }

      return -1;
    }	    

    template<int dir>
    inline constexpr int check_face_type(const std::array<int,D> &x) const {

      const auto& Nm1 = args.get_rangem1();

      int face_idx = 0;
      
      if        constexpr (dir == 0) {
        if (x[0] == Nm1[0]) face_idx = 1; 
      } else if constexpr (dir == 1) {
        if (x[0] == 0     ) face_idx = 2;
      } else if constexpr (dir == 2) {
        if (x[1] == Nm1[1]) face_idx = 4;
      } else if constexpr (dir == 3) {
        if (x[1] == 0     ) face_idx = 8;
      } else if constexpr (dir == 4) {
        if (x[2] == Nm1[2]) face_idx =16;
      } else if constexpr (dir == 5) {
        if (x[2] == 0     ) face_idx =32;
      }

      return face_idx;
    }

    template<int dir>
    inline T get_bndry_term(const std::array<int, D> &x, const int i) const {
      return static_cast<T>(0.0);//Trivial BC
    }

    //works for both nvc++ and g++
    //template recursion (still ugly ...)
    template<Shift shift, Shift... other_shifts>
    inline constexpr int GetNeighborIdx(int i) const {
      //currently unsafe: no check on dimensionality (should be done during stencil inst.)
      
      const auto& offsets = args.get_offsets();
            
      if constexpr        (shift == Shift::ShiftXp1) {
         i += 1;
      } else if constexpr (shift == Shift::ShiftXm1) {
         i -= 1;
      } else if constexpr (shift == Shift::ShiftYp1) {
         i += offsets[0];
      } else if constexpr (shift == Shift::ShiftYm1) {
         i -= offsets[0];
      } else if constexpr (shift == Shift::ShiftZp1) {
         i += offsets[1];
      } else if constexpr (shift == Shift::ShiftZm1) {
         i -= offsets[1];
      }

      if constexpr (sizeof...(other_shifts) != 0) {
        return GetNeighborIdx<other_shifts...>(i);
      } 

      return i;//stop recursion and return the index of the neighbor
    }

    T& operator[](const int i) const { return v[i];}

    template<Shift face_shift, Shift... other_shifts>
    T operator()(const int i) const {
      //	      
      if constexpr ((sizeof...( other_shifts)  > 3)) {
        printf("Number of shifts is not supported.\n"); 
        exit(-1);
      }
      //
      if constexpr (face_shift == Shift::NoShift) return v[i]; //no shifts, direct access

      const auto j = GetNeighborIdx<face_shift, other_shifts...>(i); //shifted access

      return v[j];
    }

    //
    inline decltype(auto) Indx2Coord(const int &i) const {
       //
       const auto& offsets = args.get_offsets();
       //
       std::array<int, D> x{i};// return it for 1D domain, otherwise use it also as temp.

       if constexpr (D > 1) {
#pragma unroll
         for (int j = D - 1; j >= 1; j--) {
           x[j] = x[0] / offsets[j-1];
           x[0] = (x[0] - x[j]*offsets[j-1]);
         }      
       }
              
       return x;
    }
};

template <typename T, typename Arg> struct field_mapper {
};

template <typename Arg> struct field_mapper<double, Arg> {
  using type = FieldAccessor<double, Arg>;
};

template <typename Arg> struct field_mapper<float, Arg> {
  using type = FieldAccessor<float, Arg>;
};
