#ifndef TNLEXACTOPERATORQ_H
#define	TNLEXACTOPERATORQ_H

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <mesh/tnlGrid.h>
#include <functors/tnlFunctionType.h>

template< int Dimensions >
class tnlExactOperatorQ
{};

template<>
class tnlExactOperatorQ< 1 >
{
   public:

      enum { Dimensions = 1 };

      static tnlString getType();

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

template<>
class tnlExactOperatorQ< 2 >
{
   public:

      enum { Dimensions = 2 };

      static tnlString getType();
         
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

template<>
class tnlExactOperatorQ< 3 >
{
   public:

      enum { Dimensions = 3 };

      static tnlString getType();
   
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

template< int Dimensions >
class tnlFunctionType< tnlExactOperatorQ< Dimensions > >
{
   public:
      enum { Type = tnlAnalyticFunction };
};

#include <operators/operator-Q/tnlExactOperatorQ_impl.h>


#endif	/* TNLEXACTOPERATORQ_H */
