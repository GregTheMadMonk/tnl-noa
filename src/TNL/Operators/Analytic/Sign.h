/***************************************************************************
                          ExpBump.h  -  description
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
class Sign : public Functions::Domain< Function::getDomainDimenions(), 
                                       Function::getDomainTyep() >
{
   public:
      
      typedef typename Function::RealType RealType;
      typedef Containers::StaticVector< Function::getDomainDimenions(), 
                                        RealType > VertexType;
      
      __cuda_callable__
      RealType operator()( const Function& function,
                           const VertexType& vertex,
                           const RealType& time = 0 ) const
      {
         const RealType aux = function( vertex );
         if( aux > 0.0 )
            return 1.0;
         else
            if( aux < 0.0 )
               return -1.0;
         return 0.0;
      }
};
