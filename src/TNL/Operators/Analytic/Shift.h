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
   
   
template< int Dimensions, typename Real >
class Shift : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
      
      typedef Real RealType;
      typedef Containers::StaticVector< Dimensions, RealType > VertexType;
      
      
      Shift() : shift( 0.0 ) {};
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return shift.setup( parameters, prefix + "shift-" );
      };
      
      
      void setShift( const VertexType& shift )
      {
         this->shift = shift;
      }
      
      __cuda_callable__
      const VertexType& getShift() const
      {
         return this->shift;
      }
      
      template< typename Function >
      __cuda_callable__
      RealType operator()( const Function& function,
                           const VertexType& vertex,
                           const RealType& time = 0 ) const
      {         
         return function( vertex - this->shift, time );
      }
      
      template< typename Function, 
                int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const Function& function,
                                     const VertexType& vertex,
                                     const RealType& time = 0 ) const
      {
         if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
            return this->operator()( function, vertex - this->shift, time );
         // TODO: implement the rest
      }
      
      
   protected:
      
      VertexType shift;
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL
