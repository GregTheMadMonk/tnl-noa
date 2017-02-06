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

namespace TNL {
namespace Operators {
namespace Analytic {   
   
   
template< typename Function >
class Identity : public Functions::Domain< Function::getDomainDimensions(), 
                                           Function::getDomainType() >
{
   public:
      
      typedef typename Function::RealType RealType;
      typedef Containers::StaticVector< Function::getDomainDimensions(), 
                                        RealType > VertexType;
      
      __cuda_callable__
      RealType operator()( const Function& function,
                           const VertexType& vertex,
                           const RealType& time = 0 ) const
      {
         return function( vertex, time );
      }
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL