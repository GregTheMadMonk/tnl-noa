/***************************************************************************
                          SmoothHeaviside.h  -  description
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
class SmoothHeaviside : public Functions::Domain< Function::getDomainDimenions(), 
                                                  Function::getDomainTyep() >
{
   public:
      
      typedef typename Function::RealType RealType;
      typedef Containers::StaticVector< Function::getDomainDimenions(), 
                                        RealType > VertexType;
      
      SmoothHeaviside()
      : sharpness( 1.0 )
      
      void setSharpness( const RealType& sharpness )
      {
         this->sharpness = sharpness;
      }
      
      __cuda_callable__
      const RealType getShaprness() const
      {
         return this->sharpness;
      }
      
      __cuda_callable__
      RealType operator()( const Function& function,
                           const VertexType& vertex,
                           const RealType& time = 0 ) const
      {
         const RealType aux = function( vertex, time );
         return 1.0 / ( 1.0 + exp( -2.0 * sharpness * aux ) );
      }
      
   protected:

      RealType sharpness;
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL
