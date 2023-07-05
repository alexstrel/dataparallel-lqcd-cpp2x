#pragma once

//using a-la QUDA terminology 

enum class FieldSiteSubset {
  FullSiteSubset         = 2,  
  ParitySiteSubset       = 1,  
  InvalidSiteSubset      = 0  
};

enum class FieldParity {
  EvenFieldParity        =  0, 
  OddFieldParity         =  1, 
  InvalidFieldParity     = -1   
};


