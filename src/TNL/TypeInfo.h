/***************************************************************************
                          TypeInfo.h  -  description
                             -------------------
    begin                : Nov 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cfloat>
#include <TNL/Devices/Cuda.h>

namespace TNL 
{

template< typename Type >
class TypeInfo
{
};

template<>
class TypeInfo< float >
{
   public:
      
      static __cuda_callable__ float getMaxValue() { return FLT_MAX; }
      
      static __cuda_callable__ float getMinValue() { return FLT_MIN; }   
};

template<>
class TypeInfo< double >
{
   public:
      
      static __cuda_callable__ double getMaxValue() { return DBL_MAX; }
      
      static __cuda_callable__ double getMinValue() { return DBL_MIN; }
   
   
};

} // namespace TNL