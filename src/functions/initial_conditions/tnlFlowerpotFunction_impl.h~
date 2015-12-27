/***************************************************************************
                          tnlExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLFLOWERPOTFUNCTION_IMPL_H_
#define TNLFLOWERPOTFUNCTION_IMPL_H_

#include <functions/initial_conditions/tnlFlowerpotFunction.h>

template< typename Real >
bool
tnlFlowerpotFunctionBase< Real >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->diameter = parameters.getParameter< double >( prefix + "diameter" );
   return true;
}

template< typename Real >
void tnlFlowerpotFunctionBase< Real >::setDiameter( const Real& sigma )
{
   this->diameter = diameter;
}

template< typename Real >
const Real& tnlFlowerpotFunctionBase< Real >::getDiameter() const
{
   return this->diameter;
}

/***
 * 1D
 */

template< typename Real >
tnlString
tnlFlowerpotFunction< 1, Real >::getType()
{
   return "tnlFlowerpotFunction< 1, " + ::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlFlowerpotFunction< 1, Real >::tnlFlowerpotFunction()
{
}

template< typename Real >
   template< int XDiffOrder, 
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlFlowerpotFunction< 1, Real >::getValue( const Vertex& v,
                                         const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return sin( M_PI * tanh( 5 * ( x * x - this->diameter ) ) );
   return 0.0;
}

/****
 * 2D
 */

template< typename Real >
tnlString
tnlFlowerpotFunction< 2, Real >::getType()
{
   return tnlString( "tnlFlowerpotFunction< 2, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlFlowerpotFunction< 2, Real >::tnlFlowerpotFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlFlowerpotFunction< 2, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return sin( M_PI * tanh( 5 * ( x * x + y * y - this->diameter ) ) );
   return 0.0;
}

/****
 * 3D
 */

template< typename Real >
tnlString
tnlFlowerpotFunction< 3, Real >::getType()
{
   return tnlString( "tnlFlowerpotFunction< 3, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlFlowerpotFunction< 3, Real >::tnlFlowerpotFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlFlowerpotFunction< 3, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return sin( M_PI * tanh( 5 * ( x * x + y * y + z * z - 0.25 ) ) );
   return 0.0;
}


#endif /* TNLFLOWERPOTFUNCTION_IMPL_H_ */
