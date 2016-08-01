/***************************************************************************
                          tnlExactFunctionInverseOperator.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/operators/tnlOperator.h>
#include <TNL/operators/tnlExactIdentityOperator.h>

namespace TNL {

template< int Dimensions,
          typename InnerOperator= tnlExactIdentityOperator< Dimensions > >
class tnlExactFunctionInverseOperator
   : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
 
      static String getType()
      {
         return String( "tnlExactFunctionInverseOperator< " ) +
                String( Dimensions) + " >";
      }
 
      InnerOperator& getInnerOperator()
      {
         return this->innerOperator;
      }
 
      const InnerOperator& getInnerOperator() const
      {
         return this->innerOperator;
      }

      template< typename Function >
      __cuda_callable__
      typename Function::RealType
         operator()( const Function& function,
                     const typename Function::VertexType& v,
                     const typename Function::RealType& time = 0.0 ) const
      {
         typedef typename Function::RealType RealType;
         return 1.0 / innerOperator( function, v, time );
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
         static_assert( XDerivative + YDerivative + ZDerivative < 2, "Partial derivative of higher order then 1 are not implemented yet." );
         typedef typename Function::RealType RealType;
 
         if( XDerivative == 1 )
         {
            const RealType f = innerOperator( function, v, time );
            const RealType f_x = innerOperator.template getPartialDerivative< Function, 1, 0, 0 >( function, v, time );
            return -f_x / ( f * f );
         }
         if( YDerivative == 1 )
         {
            const RealType f = innerOperator( function, v, time );
            const RealType f_y = innerOperator.template getPartialDerivative< Function, 0, 1, 0 >( function, v, time );
            return -f_y / ( f * f );
         }
         if( ZDerivative == 1 )
         {
            const RealType f = innerOperator( function, v, time );
            const RealType f_z = innerOperator.template getPartialDerivative< Function, 0, 0, 1 >( function, v, time );
            return -f_z / ( f * f );
         }
      }
 
   protected:
 
      InnerOperator innerOperator;
};

} // namespace TNL

