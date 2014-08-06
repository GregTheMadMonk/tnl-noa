/***************************************************************************
                          tnlTestFunction.h  -  description
                             -------------------
    begin                : Aug 2, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLTESTFUNCTION_H_
#define TNLTESTFUNCTION_H_


template< int FunctionDimensions,
          typename Real,
          typename Device >
class tnlTestFunction
{
   protected:

   enum TestFunctions{ none,
                       constant,
                       expBump,
                       sinBumps,
                       sinWave };

   public:

   enum{ Dimensions = FunctionDimensions };
   typedef Real RealType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;

   tnlTestFunction();

   bool init( const tnlParameterContainer& parameters );

   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getValue( const Vertex& vertex ) const;

   ~tnlTestFunction();

   protected:

   template< typename FunctionType >
   bool initFunction( const tnlParameterContainer& parameters );

   template< typename FunctionType >
   void deleteFunction();

   void* function;

   TestFunctions functionType;

};

#include <implementation/functions/tnlTestFunction_impl.h>

#endif /* TNLTESTFUNCTION_H_ */
