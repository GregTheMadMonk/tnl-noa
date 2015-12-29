/***************************************************************************
                          tnlExactLinearDiffusion_impl.h  -  description
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

#ifndef TNLEXACTOPERATORQ_IMPL_H_
#define TNLEXACTOPERATORQ_IMPL_H_

#include <operators/operator-Q/tnlExactOperatorQ.h>

tnlString
tnlExactOperatorQ< 1 >::
getType()
{
   return "tnlExactOperatorQ< 1 >";
}

template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Function, typename Vertex, typename Real >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real
tnlExactOperatorQ< 1 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time, const Real& eps )
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
        return 0.0;
   if (XDiffOrder == 0)
        return sqrt(eps * eps + function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 1, 0, 0 >( v, time ) );
   if (XDiffOrder == 1)
        return (function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 2, 0, 0 >( v, time ) ) / getValue( function, v, time, eps);
   return 0;
}

tnlString
tnlExactOperatorQ< 2 >::
getType()
{
   return "tnlExactOperatorQ< 2 >";
}

template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Function, typename Vertex, typename Real >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real
tnlExactOperatorQ< 2 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time, const Real& eps )
{
   if( ZDiffOrder != 0 )
        return 0.0;
   if (XDiffOrder == 0 && YDiffOrder == 0 )
        return sqrt(eps * eps + function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 1, 0, 0 >( v, time ) 
                + function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 0, 1, 0 >( v, time ) );
   if (XDiffOrder == 1 && YDiffOrder == 0 )
        return (function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 2, 0, 0 >( v, time ) + 
                function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 1, 1, 0 >( v, time )) / getValue( function, v, time, eps);
   if (XDiffOrder == 0 && YDiffOrder == 1 )
        return (function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 0, 2, 0 >( v, time ) + 
                function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 1, 1, 0 >( v, time )) / getValue( function, v, time, eps);
   return 0;
}

tnlString
tnlExactOperatorQ< 3 >::
getType()
{
   return "tnlExactOperatorQ< 3 >";
}

template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Function, typename Vertex, typename Real >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real
tnlExactOperatorQ< 3 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time, const Real& eps )
{
   if ( XDiffOrder == 0 && YDiffOrder == 0  && ZDiffOrder == 0 )
        return sqrt(eps * eps + function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 1, 0, 0 >( v, time ) 
                + function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 0, 1, 0 >( v, time )
                + function.template getValue< 0, 0, 1 >( v, time ) * function.template getValue< 0, 0, 1 >( v, time ) );
   if (XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
        return (function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 2, 0, 0 >( v, time ) + 
                function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 1, 1, 0 >( v, time ) + 
                function.template getValue< 0, 0, 1 >( v, time ) * function.template getValue< 1, 0, 1 >( v, time )) / getValue( function, v, time, eps);
   if (XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
        return (function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 1, 1, 0 >( v, time ) + 
                function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 0, 2, 0 >( v, time ) + 
                function.template getValue< 0, 0, 1 >( v, time ) * function.template getValue< 0, 1, 1 >( v, time )) / getValue( function, v, time, eps);
   if (XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
        return (function.template getValue< 1, 0, 0 >( v, time ) * function.template getValue< 1, 0, 1 >( v, time ) + 
                function.template getValue< 0, 1, 0 >( v, time ) * function.template getValue< 0, 1, 1 >( v, time ) + 
                function.template getValue< 0, 0, 1 >( v, time ) * function.template getValue< 0, 0, 2 >( v, time )) / getValue( function, v, time, eps);
   return 0;
}

#endif /* TNLEXACTOPERATORQ_IMPL_H_ */
