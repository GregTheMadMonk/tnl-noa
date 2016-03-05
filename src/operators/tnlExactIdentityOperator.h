/***************************************************************************
                          tnlExactIdentityOperator.h  -  description
                             -------------------
    begin                : Feb 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLEXACTIDENTITYOPERATOR_H
#define TNLEXACTIDENTITYOPERATOR_H

#include <core/tnlString.h>
#include <core/tnlCuda.h>
#include <operators/tnlOperator.h>

template< int Dimensions >
class tnlExactIdentityOperator
   : public tnlDomain< Dimensions, SpaceDomain >
{
   public:
     
      static tnlString getType()
      {
         return tnlString( "tnlExactIdentityOperator< " ) + 
                tnlString( Dimensions) + " >";         
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
         return function( v, time );
      }
      
      template< typename Function, 
                int XDerivative = 0,
                int YDerivative = 0,
                int ZDerivative = 0 >
      __cuda_callable__
      typename Function::RealType
         getPartialDerivative( const Function& function,
                               const typename Function::VertexType& v,
                               const typename Function::RealType& time = 0.0 ) const
      {
         static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
            "Partial derivative must be non-negative integer." );
         
         return function.template getPartialDerivative< XDerivative, YDerivative, ZDerivative >( v, time );         
      }
};



#endif /* TNLEXACTIDENTITYOPERATOR_H */

