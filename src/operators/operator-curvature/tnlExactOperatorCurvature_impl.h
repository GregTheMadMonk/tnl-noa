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

#ifndef TNLEXACTOPERATORCURVATURE_IMPL_H_
#define TNLEXACTOPERATORCURVATURE_IMPL_H_

#include <operators/operator-curvature/tnlExactOperatorCurvature.h>

template< typename ExactOperatorQ >
tnlString
tnlExactOperatorCurvature< ExactOperatorQ, 1 >::
getType()
{
   return "tnlExactOperatorCurvature< " + ExactOperatorQ::getType() + ",1 >";
}

template< typename OperatorQ >
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
        return function.template getValue< 2, 0, 0, Vertex >( v, time )/ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) -
               ( function.template getValue< 1, 0, 0, Vertex >( v, time ) * ExactOperatorQ::template getValue< 1, 0, 0 >( function, v, time, eps ) )
                / ( ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) * ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) );
   return 0;
}

template< typename ExactOperatorQ >
tnlString
tnlExactOperatorCurvature< ExactOperatorQ, 2 >::
getType()
{
   return "tnlExactOperatorCurvature< " + ExactOperatorQ::getType() + ",2 >";
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
        return ( function.template getValue< 2, 0, 0, Vertex >( v, time ) * function.template getValue< 0, 2, 0, Vertex >( v, time ) )
               /ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) - ( function.template getValue< 1, 0, 0, Vertex >( v, time ) * 
               ExactOperatorQ::template getValue< 1, 0, 0 >( function, v, time, eps ) + function.template getValue< 0, 1, 0, Vertex >( v, time ) * 
               ExactOperatorQ::template getValue< 0, 1, 0 >( function, v, time, eps ) )
                / ( ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) * ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) );
   return 0;
}

template< typename ExactOperatorQ >
tnlString
tnlExactOperatorCurvature< ExactOperatorQ, 3 >::
getType()
{
   return "tnlExactOperatorCurvature< " + ExactOperatorQ::getType() + ",3 >";
}

#endif /* TNLEXACTOPERATORCURVATURE_IMPL_H_ */
