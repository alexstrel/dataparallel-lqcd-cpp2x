#pragma once

#include <core/common.h>

enum class PMRState {
  Vacant            =  0,
  //    
  ReservedExclusive =  1, 
  LockedExclusive   =  2,          
  //
  Shared                =  3, 
  ReservedShared        =  4,
  ReservedNonExclusive  =  5,       
  LockedNonExclusive    =  6, 
  //     
  InvalidState      = min_int  
};

//using a-la QUDA terminology 
enum class FieldSiteSubset {
  FullSiteSubset         = 2,  
  ParitySiteSubset       = 1,  
  InvalidSiteSubset      = min_int  
};

enum class FieldParity {
  EvenFieldParity        =  0, 
  OddFieldParity         =  1, 
  InvalidFieldParity     =  min_int  
};

enum class FieldType {
  ScalarFieldType    =  1,
  VectorFieldType    =  2,
  SpinorFieldType    =  3,
  InvalidFieldType   =  min_int
};

enum class FieldOrder {
  LexFieldOrder     =  1,
  EOFieldOrder      =  2,
  InvalidFieldOrder =  min_int
};


