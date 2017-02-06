/***************************************************************************
                          Heaviside.h  -  description
                             -------------------
    begin                : Feb 6, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
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
class Heaviside : public Functions::Domain< Function::getDomainDimensions(), 
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
         const RealType aux = function( vertex, time );
         if( aux > 0.0 )
            return 1.0;
         return 0.0;
      }
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL