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
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Operators {
namespace Analytic {   
   
   
template< typename Function >
class Shift : public Functions::Domain< Function::getDomainDimensions(), 
                                        Function::getDomainType() >
{
   public:
      
      typedef typename Function::RealType RealType;
      typedef Containers::StaticVector< Function::getDomainDimensions(), 
                                        RealType > VertexType;
      
      
      Shift() : shift( 0.0 ) {};
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return shift.setup( parameters, prefix + "shift-" );
      };
      
      
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
      
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const Function& function,
                                     const VertexType& vertex,
                                     const RealType& time = 0 ) const
      {
         if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
            return this->operator()( function, vertex, time );
         // TODO: implement the rest
      }
      
      
   protected:
      
      VertexType shift;
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL
