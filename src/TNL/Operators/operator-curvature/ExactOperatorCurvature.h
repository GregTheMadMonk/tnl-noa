/***************************************************************************
                          ExactOperatorCurvature.h  -  description
                             -------------------
    begin                : Jan 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/tnlFunction.h>

namespace TNL {
namespace Operators {   

template< typename ExactOperatorQ, int Dimensions >
class ExactOperatorCurvature
{};

template< typename ExactOperatorQ >
class ExactOperatorCurvature< OperatorQ, 1 >
{
   public:

      enum { Dimensions = 1 };

      static String getType();

#ifdef HAVE_NOT_CXX11      
      template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0, typename Function, typename Vertex, typename Real >
#else   
      template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0, typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0, const Real& eps = 1.0 );
      
};

template< typename ExactOperatorQ >
class ExactOperatorCurvature< ExactOperatorQ, 2 >
{
   public:

      enum { Dimensions = 2 };

      static String getType();
         
#ifdef HAVE_NOT_CXX11      
      template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0, typename Function, typename Vertex, typename Real >
#else   
      template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0, typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif
#ifdef HAVE_CUDA
      __device__ __host__
#endif      
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0, const Real& eps = 1.0 );
};

template< typename ExactOperatorQ >
class ExactOperatorCurvature< ExactOperatorQ, 3 >
{
   public:

      enum { Dimensions = 3 };

      static String getType();
   
#ifdef HAVE_NOT_CXX11      
      template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0, typename Function, typename Vertex, typename Real >
#else   
      template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0, typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0, const Real& eps = 1.0 )
      {
         return 0;
      }
};

template< typename ExactOperatorQ, int Dimensions >
class tnlFunctionType< ExactOperatorCurvature< ExactOperatorQ, Dimensions > >
{
   public:
      enum { Type = tnlSpaceDomain };
};

} // namespace Operators
} // namespace TNL

#include <TNL/Operators/operator-curvature/ExactOperatorCurvature_impl.h>
