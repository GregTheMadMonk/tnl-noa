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

#include <core/tnlHost.h>

template< int FunctionDimensions,
          typename Real = double,
          typename Device = tnlHost >
class tnlTestFunction
{
   protected:

   enum TestFunctions{ constant,
                       expBump,
                       sinBumps,
                       sinWave };

   enum TimeDependence { none,
                         linear,
                         quadratic,
                         sine };

   public:

   enum{ Dimensions = FunctionDimensions };
   typedef Real RealType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;

   tnlTestFunction();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool init( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getValue( const Vertex& vertex,
                  const Real& time = 0 ) const;

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getTimeDerivative( const Vertex& vertex,
                           const Real& time = 0 ) const;

   ~tnlTestFunction();

   protected:

   template< typename FunctionType >
   bool initFunction( const tnlParameterContainer& parameters,
                      const tnlString& prefix = "" );

   template< typename FunctionType >
   void deleteFunction();

   void* function;

   TestFunctions functionType;

   TimeDependence timeDependence;

   Real timeScale;

};

#include <implementation/functions/tnlTestFunction_impl.h>

#endif /* TNLTESTFUNCTION_H_ */
