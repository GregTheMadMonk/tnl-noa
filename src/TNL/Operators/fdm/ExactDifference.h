/***************************************************************************
                          ExactDifference.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Operators {   

template< int Dimensions,
          int XDerivative,
          int YDerivative,
          int ZDerivative >
class ExactDifference
   : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
 
      static String getType()
      {
         return String( "ExactDifference< " ) +
            String( Dimensions ) + ", " +
            String( XDerivative ) + ", " +
            String( YDerivative ) + ", " +
            String( ZDerivative ) + " >";
      }
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType operator()(
         const Function& function,
         const typename Function::VertexType& vertex,
         const typename Function::RealType& time = 0 ) const
      {
         return function.template getPartialDerivative<
            XDerivative,
            YDerivative,
            ZDerivative >(
            vertex,
            time );
      }
};

} // namespace Operators
} // namespace TNL
