/***************************************************************************
                          tnlExactLinearDiffusion.h  -  description
                             -------------------
    begin                : Aug 8, 2014
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

#ifndef TNLEXACTNONLINEARDIFFUSION_H_
#define TNLEXACTNONLINEARDIFFUSION_H_

#include <functions/tnlFunction.h>

template< typename OperatorQ, int Dimensions >
class tnlExactNonlinearDiffusion
{};

template< typename OperatorQ >
class tnlExactNonlinearDiffusion< OperatorQ, 1 >
{
   public:

      enum { Dimensions = 1 };

      static tnlString getType();
   
#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif      
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0 );
};

template< typename OperatorQ >
class tnlExactNonlinearDiffusion< OperatorQ, 2 >
{
   public:

      enum { Dimensions = 2 };

      static tnlString getType();

#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif 
#ifdef HAVE_CUDA
      __device__ __host__
#endif      
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0 );
};

template< typename OperatorQ >
class tnlExactNonlinearDiffusion< OperatorQ, 3 >
{
   public:

      enum { Dimensions = 3 };

      static tnlString getType();

#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif 
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0 );
};

template< typename OperatorQ, int Dimensions >
class tnlFunctionType< tnlExactNonlinearDiffusion< OperatorQ, Dimensions > >
{
   public:
      enum { Type = tnlAnalyticFunction };
};

#include "tnlExactNonlinearDiffusion_impl.h"

#endif /* TNLEXACTNONLINEARDIFFUSION_H_ */
