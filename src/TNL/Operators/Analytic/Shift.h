/***************************************************************************
                          Shift.h  -  description
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
class Shift : public Functions::Domain< Function::getDomainDimenions(), 
                                        Function::getDomainTyep() >
{
   public:
      
      typedef typename Function::RealType RealType;
      typedef Containers::StaticVector< Function::getDomainDimenions(), 
                                        RealType > VertexType;
      
      
      Shift() : shift( 0.0 ) {};
      
      void setShift( const VertexType& vertex )
      {
         this->shift = shift;
      }
      
      __cuda_callable__
      const VertexType& getShift() const
      {
         return this->shift;
      }
      
      __cuda_callable__
      RealType operator()( const Function& function,
                           const VertexType& vertex,
                           const RealType& time = 0 ) const
      {
         return function( vertex + shift, time );
      }
      
   protected:
      
      VerexType shift;
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL
