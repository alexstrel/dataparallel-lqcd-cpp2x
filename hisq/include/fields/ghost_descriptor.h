#pragma once

#include <memory>
#include <memory_resource>
#include <functional>

#include <fields/field_concepts.h>
#include <core/enums.h>
#include <core/memory.h>

template<FieldType type, std::size_t nDim, std::size_t nSpin, std::size_t nColor, std::size_t nParity>
class GhostDescriptor {
  private:
    static constexpr int nFace       = 1;//always 1 for schwinger model
    static constexpr int CommDir     = type == FieldType::VectorFieldType ? 1 : 2;
    static constexpr int nGhostExtra = type == FieldType::VectorFieldType ? (nParity == 2 ? 1 : 0) : (nParity == 2 ? 3 : 2); 
    
    const std::array<int, nDim> comm_dir;
        
    const std::array<std::size_t, (nDim-1)+nGhostExtra> mdStrides; //for mdspan views only 
  public: 
    GhostDescriptor(const std::array<int, nDim> &dir, const std::array<int, nDim> &comm_dir) :
                    comm_dir{comm_dir}, 
                    mdStrides([&d = dir]()->std::array<std::size_t, nDim-1+nGhostExtra> {
                        std::array<std::size_t, nDim-1+nGhostExtra> strides{1};
                        if constexpr (nParity == 2) {
                          const int d0 = d[0] / 2;                        
                          if constexpr (type == FieldType::VectorFieldType) {
                            strides[nDim+nGhostExtra-2] = d0;
                          } else { 
                            strides[nDim+nGhostExtra-4] = d0;
                            strides[nDim+nGhostExtra-3] = d0*nSpin;
                            strides[nDim+nGhostExtra-2] = d0*nSpin*nParity;
                          }
                        } else {
                          if constexpr (type == FieldType::SpinorFieldType) {
                            strides[nDim+nGhostExtra-3] = d[0];
                            strides[nDim+nGhostExtra-2] = d[0]*nSpin;
                          }
                        } return strides;                        
                   }()) { }

    GhostDescriptor(const std::array<int, nDim> &dir, const FieldParity parity) : 
	            comm_dir([&dir_= dir]()->std::array<int, nDim> {
                        std::array<int, nDim> comm_dir{};

                        for( int d = 0; d < nDim; d++){
                          const int div = (d == 0 and nParity == 1) ? 2 : 1;
                          
                          int vol = 1;
                          for( int d2 = 0; d2 < nDim; d2++ ) vol *= (d2 == d ? 1 : dir_[d2] / div);
                          
                          comm_dir[d] = vol;
                        } return comm_dir;
                      }()),
                      mdStrides([&d = dir]()->std::array<std::size_t, nDim-1+nGhostExtra> {
                        std::array<std::size_t, nDim-1+nGhostExtra> strides{1};
                        if constexpr (nParity == 2) {
                          const int d0 = d[0] / 2;                        
                          if constexpr (type == FieldType::VectorFieldType) {
                            strides[nDim+nGhostExtra-2] = d0;
                          } else { 
                            strides[nDim+nGhostExtra-4] = d0;
                            strides[nDim+nGhostExtra-3] = d0*nSpin;
                            strides[nDim+nGhostExtra-2] = d0*nSpin*nParity;
                          }
                        } else {
                          if constexpr (type == FieldType::SpinorFieldType) {
                            strides[nDim+nGhostExtra-3] = d[0];
                            strides[nDim+nGhostExtra-2] = d[0]*nSpin;
                          }
                        } return strides;                        
                   }()) { }                    
                    
    static constexpr int getNFace() { return nFace; }                    
    
    inline auto& GetGhostMDStrides() const { return mdStrides; } 
    
    inline decltype(auto) GetGhostZoneSize() const {
      int vol = 1; 
#pragma unroll      
      for(int i = 0; i < nDim; i++) vol += comm_dir[i];
      
      if  constexpr (type == FieldType::ScalarFieldType) {
        return vol*nFace;
      } else if constexpr (type == FieldType::VectorFieldType) {
        return vol*nColor*nColor*nFace;
      } else if constexpr (type == FieldType::SpinorFieldType) {
        return vol*nSpin*nColor*nFace;
      }
      return static_cast<std::size_t>(0);
    }

    inline auto GetFaceSize(const int i) const {
      if  constexpr (type == FieldType::ScalarFieldType) {
        return comm_dir[i];
      } else if constexpr (type == FieldType::VectorFieldType) {
	return comm_dir[i]*nColor*nColor;
      } else if constexpr (type == FieldType::SpinorFieldType) {
	return comm_dir[i]*nSpin*nColor;
      }
      //
      return static_cast<std::size_t>(0);
    } 
    
    inline int  GetCommDims(const int i) const { return comm_dir[i]; }    
    
    inline auto GetCommDims()            const { return comm_dir;    }                          
                   
    GhostDescriptor()                        = default;
    GhostDescriptor(const GhostDescriptor& ) = default;
    GhostDescriptor(GhostDescriptor&& )      = default;                           
};

