/* 
 * File:   tnlTypeInfo.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:46 PM
 */

#pragma once

#include <limits>

template< typename Type >
class tnlTypeInfo
{   
};

template<>
class tnlTypeInfo< double >
{
   public:
      
      typedef double Type;
      
      static __cuda_callable__
      Type getMaxValue() { return DBL_MAX; };
};

template<>
class tnlTypeInfo< float >
{
   public:
      
      typedef float Type;
      
      static __cuda_callable__
      Type getMaxValue() { return FLT_MAX; };
};


