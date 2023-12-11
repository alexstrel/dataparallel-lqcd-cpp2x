#pragma once

#include <assert.h>
#include <algorithm>
#include <ranges>
#include <execution>
#include <numeric>
#include <memory>
#include <memory_resource>
#include <functional>

#include <fields/field_concepts.h>
#include <core/enums.h>
#include <core/memory.h>

#include <fields/ghost_descriptor.h>

template<std::size_t nD, std::size_t nS, std::size_t nC>
consteval FieldType get_field_type() {

  if constexpr (nD != invalid_dir and nS == invalid_spin  and nC != invalid_color){
    return FieldType::VectorFieldType;	  
  } else if constexpr (nD == invalid_dir and nS != invalid_spin  and nC != invalid_color) {
    return FieldType::SpinorFieldType;	      
  } else if constexpr (nD == invalid_dir and nS == invalid_spin  and nC != invalid_color) {
    return FieldType::StaggeredSpinorFieldType;	  
  }

  return FieldType::InvalidFieldType;
}

template<std::size_t nDim, std::size_t nDir = invalid_dir, std::size_t nSpin = invalid_spin, std::size_t nColor = invalid_color, std::size_t nParity = invalid_parity>
class FieldDescriptor {
  private:
    template <std::size_t src_nParity>
    static auto get_dims(const auto &src_dir){
      constexpr std::size_t dst_nParity = nParity; 
      
      std::array dir_{src_dir};
      
      if constexpr (dst_nParity == 1 and src_nParity == 2) dir_[0] /= 2;
      
      return dir_;
    }
    
  public: 
    static constexpr std::size_t ndim   = nDim;                    // FIXME
    //
    static constexpr std::size_t ndir   = nDir;                    //vector field dim   (2 for U1 gauge)	  
    static constexpr std::size_t nspin  = nSpin;                   //number of spin dof (2 for spinor)
    static constexpr std::size_t ncolor = nColor;                  //for all fields
    static constexpr std::size_t nparity= nParity;                 //for all fields    

    static constexpr FieldType  type = get_field_type<ndir, nspin, ncolor>();

    //gauge  field extra dimensions: color + dirs[ + parity, if par = 2 ] 
    //spinor field extra dimensions: color [ + parity, if par = 2 ]     
    static constexpr int nExtra      = type == FieldType::VectorFieldType ? (nparity == 2 ? 4 : 3) : (nparity == 2 ? 2 : 1);       
    
    using ghost = GhostDescriptor<type, ndim, nspin, ncolor, nparity>;
    
    static constexpr int nFace() { return ghost::getNFace(); }    

    const std::array<int, ndim> dir;    

    const FieldOrder         order  = FieldOrder::InvalidFieldOrder;        		
    const FieldParity        parity = FieldParity::InvalidFieldParity;//this is optional param
    const FieldBoundary      bc     = FieldBoundary::InvalidBC;

    std::shared_ptr<PMRBuffer> pmr_buffer;

    const std::array<std::size_t, ndim+nExtra> mdStrides;            //for mdspan views only

    const ghost ghost_descriptor;

    FieldDescriptor()                        = default;
    FieldDescriptor(const FieldDescriptor& ) = default;
    FieldDescriptor(FieldDescriptor&& )      = default;

    FieldDescriptor(const std::array<int, ndim> dir, 
                    const std::array<int, ndim> comm_dir,  
	            const FieldParity     parity   = FieldParity::InvalidFieldParity,
	            const FieldOrder      order    = FieldOrder::EOFieldOrder,
                    const FieldBoundary   bc       = FieldBoundary::InvalidBC,		    
	            const bool is_exclusive        = true) : 
	            dir{dir},
	            order(order),
	            parity(parity), 
		    bc(bc),
                    pmr_buffer(nullptr), 
                    mdStrides([&d = dir]()->std::array<std::size_t, ndim+nExtra> {
                        const int d0 = nparity == 2 ? d[0] / 2 : d[0];
                        std::array<std::size_t, ndim+nExtra> strides{1, d0, d0*d[1], d0*d[1]*d[2], d0*d[1]*d[2]*d[3]};

                        if constexpr (nparity == 2) {
                          if constexpr (type == FieldType::VectorFieldType) {
                            strides[ndim+1] =  strides[ndim+0]*ncolor;                                                  
                            strides[ndim+2] =  strides[ndim+1]*ncolor; 
                            strides[ndim+3] =  strides[ndim+2]*ndir;                             
                          } else if constexpr (type == FieldType::StaggeredSpinorFieldType){ //spinor
                            strides[ndim+1] =  strides[ndim+0]*ncolor;                                                                            
                          }
                        } else if constexpr (nparity == 1) {
                          if constexpr (type == FieldType::VectorFieldType) {
                            strides[ndim+1] =  strides[ndim+0]*ncolor;                                                  
                            strides[ndim+2] =  strides[ndim+1]*ncolor;                                                                              
                          }
                        } 
                        return strides;
                      }()),
                   ghost_descriptor(dir, comm_dir) { 
                      if (parity != FieldParity::InvalidFieldParity and nparity != 1) {
                        std::cerr << "Incorrect number of parities " << std::endl;
                        std::quick_exit( EXIT_FAILURE );
                      }
                    } 
                    
    template<typename Args>
    FieldDescriptor(const Args &args, const FieldParity parity) : 
                    dir(get_dims<std::remove_cvref_t<decltype(args)>::nparity>(args.dir)),
	            order(args.order),
	            parity(parity),
		    bc(args.bc),
                    pmr_buffer(args.pmr_buffer), 
                    mdStrides([&d = this->dir]()->std::array<std::size_t, ndim+nExtra> {
                        const int d0 = nparity == 2 ? d[0] / 2 : d[0];
                        std::array<std::size_t, ndim+nExtra> strides{1, d0, d0*d[1], d0*d[1]*d[2], d0*d[1]*d[2]*d[3]};

                        if constexpr (nparity == 2) {
                          if constexpr (type == FieldType::VectorFieldType) {
                            strides[ndim+1] =  strides[ndim+0]*ncolor;                                                  
                            strides[ndim+2] =  strides[ndim+1]*ncolor; 
                            strides[ndim+3] =  strides[ndim+2]*ndir;                               
                          } else if constexpr (type == FieldType::StaggeredSpinorFieldType){ //spinor
                            strides[ndim+1] =  strides[ndim+0]*ncolor;                                                                            
                          }
                        } else if constexpr (nparity == 1) {
                          if constexpr (type == FieldType::VectorFieldType) {
                            strides[ndim+1] =  strides[ndim+0]*ncolor;                                                  
                            strides[ndim+2] =  strides[ndim+1]*ncolor;                                                                              
                          }
                        } 
                        return strides;
                      }()),
                      ghost_descriptor(args.dir, parity) { 
                      if (parity != FieldParity::InvalidFieldParity and nparity != 1) {
                        std::cerr << "Incorrect number of parities " << std::endl;
                        std::quick_exit( EXIT_FAILURE );
                      }                    
                    } 

    //Use it for block fields only:
    FieldDescriptor(const FieldDescriptor &args, 
                    const std::shared_ptr<PMRBuffer> extern_pmr_buffer) :
                    dir(args.dir),
                    order(args.order),
                    parity(args.parity),
		    bc(args.bc),
                    pmr_buffer(extern_pmr_buffer), 
                    mdStrides(args.mdStrides), 
                    ghost_descriptor(args.ghost_descriptor) { }
    
    decltype(auto) GetFieldSize() const {
      int vol = 1; 
#pragma unroll      
      for(int i = 0; i < ndim; i++) vol *= dir[i];
      
      if  constexpr (type == FieldType::ScalarFieldType) {
        return vol;
      } else if constexpr (type == FieldType::VectorFieldType) {
	return vol*nDir*nColor*nColor;
      } else if constexpr (type == FieldType::SpinorFieldType) {
	return vol*nSpin*nColor;
      } else if constexpr (type == FieldType::StaggeredSpinorFieldType) {
	return vol*nColor;	
      }
      //
      return static_cast<std::size_t>(0);
    } 

    decltype(auto) GetGhostZoneSize() const { return ghost_descriptor.GetGhostZoneSize(); }

    auto GetFaceSize(const int i) const { return ghost_descriptor.GetFaceSize(i); }

    auto GetLatticeDims() const { return dir; }

    auto GetParityLatticeDims() const {
      std::array xcb{dir};
      xcb[0] = nParity == 2 ? xcb[0] / nParity : xcb[0];	    
      return xcb;
    }
    
    auto GetParity() const { return parity; }
    
    auto GetFieldSubset() const { return (nParity == 2 ? FieldSiteSubset::FullSiteSubset : (nParity == 1 ? FieldSiteSubset::ParitySiteSubset : FieldSiteSubset::InvalidSiteSubset)); }

    inline int  X(const int i) const { return dir[i]; }
    
    inline auto X() const { return dir; }    
    
    inline int  GetCommDims(const int i) const { return ghost_descriptor.GetCommDims(i); }    
    
    inline auto GetCommDims()            const { return ghost_descriptor.GetCommDims(); }        
 
    inline auto& GetMDStrides()      const { return mdStrides; }

    inline auto& GetGhostMDStrides() const { return ghost_descriptor.GetGhostMDStrides(); }     
  
    template<ArithmeticTp T, bool is_exclusive = true>
    void RegisterPMRBuffer(const bool is_reserved = false) {  
      // 
      const std::size_t nbytes = (GetFieldSize()+GetGhostZoneSize())*sizeof(T);
      //
      if (pmr_buffer != nullptr) pmr_buffer.reset(); 
      //
      pmr_buffer = pmr_pool::pmr_malloc<is_exclusive>(nbytes, is_reserved);
    }    

    void UnregisterPMRBuffer() {
      //
      if(pmr_buffer != nullptr) { 
        pmr_buffer->Release();
        //
        pmr_buffer.reset(); 
      }
    }
    
    void ResetPMRBuffer() {
      //
      if(pmr_buffer != nullptr) { 
        pmr_buffer.reset(); 
      }
    }

    void ReleasePMRBuffer() const { if(pmr_buffer != nullptr) pmr_buffer->Release(); }     
    
    template<ArithmeticTp T>    
    bool IsReservedPMR() const {
      //
      const std::size_t nbytes = (GetFieldSize()+GetGhostZoneSize())*sizeof(T);    
      //
      return pmr_buffer->IsReserved(nbytes);
    } 

    bool IsExclusive() const { 
      if (pmr_buffer != nullptr) return pmr_buffer->IsExclusive(); 
      //
      return false;
    }

    auto GetState() const { 
      if (pmr_buffer != nullptr) {
        return pmr_buffer->State(); 
      }

      return PMRState::InvalidState;
    } 

    auto SetState(PMRState state) { 
      if (pmr_buffer != nullptr) { pmr_buffer->SetState(state); }
    }
    
    void UpdatedReservedPMR() const { pmr_buffer->UpdateReservedState(); }

    auto operator=(const FieldDescriptor&) -> FieldDescriptor& = default;
    auto operator=(FieldDescriptor&&     ) -> FieldDescriptor& = default;
};

constexpr std::size_t ndims   = 4;
constexpr std::size_t ncolors = 3;
constexpr std::size_t ndirs   = 4;

template<int nParity = invalid_parity> using GaugeFieldArgs           = FieldDescriptor<ndims, ndirs, invalid_spin, ncolors, nParity>;

template<int nParity = invalid_parity> using StaggeredSpinorFieldArgs = FieldDescriptor<ndims, invalid_dir, invalid_spin, ncolors, nParity>;

