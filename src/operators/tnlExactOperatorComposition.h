/***************************************************************************
                          tnlExactOperatorComposition.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename OuterOperator,
          typename InnerOperator >
class tnlExactOperatorComposition
{
   public:
 
      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::VertexType& v,
                                              const typename Function::RealType& time = 0.0 ) const
      {
         return OuterOperator( innerOperator( function, v, time), v, time );
      }
 
   protected:
 
      InnerOperator innerOperator;
 
      OuterOperator outerOperator;
};

} // namespace TNL

