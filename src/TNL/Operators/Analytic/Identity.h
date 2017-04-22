/***************************************************************************
                          Identity.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Operators {
namespace Analytic {   
   
   
template< int Dimensions, typename Real >
class Identity : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
      
      typedef Real RealType;
      typedef Containers::StaticVector< Dimensions, RealType > PointType;
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true;
      };
      
      
      template< typename Function >
      __cuda_callable__
      RealType operator()( const Function& function,
                           const PointType& vertex,
                           const RealType& time = 0 ) const
      {
         return function( vertex, time );
      }
      
      template< typename Function,
                int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const Function& function,
                                     const PointType& vertex,
                                     const RealType& time = 0 ) const
      {
         return function.getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL