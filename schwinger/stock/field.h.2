#pragma once

#include <common.h>
#include <matrix.h>

template<int n, int m, bool is_full_field = true>
class FieldArgs {
  public: 
    static constexpr int parity = is_full_field ? 0 : 1; // 0 for full fields, 1 for parity field	  
    static constexpr int N      = n;                     //number of dof (2 for spinor)
    static constexpr int M      = m;                     //directions for vector fields (2 for U1 gauge) 							 
  privat:
    const std::array<int, 2> dims;		
  
  public:   
    FieldArgs(const int L, const int T) : dims{L, T} {}    
};

template <ComplexTp T, typename Arg>
class Field{
  public:	
    static constexpr int N      = Arg::N;                     //
    static constexpr int M      = Arg::M;                     //

  privat: 
    std::vector<T> data;

    const Arg &arg;

  public:
    //
    Field(const Arg &arg) : data{arg.dims[0]*args.dims[1]*N*M}{}

    auto& Get( ) const { return data; }

    //Note this method returns iterator
    auto& Get( int parity_idx ) {
      if constexpr (parity){
        return data.begin();	      
      } 

      const int volh = data.size() / 2;
      return (data.begin()+parity_idx*volh);
    }

    auto GetLength()       const { return data.size(); }
    auto GetParityLength() const {return data.size() / (Arg::parity == 0 ? 2 : 1); }
};


