/***************************************************************************
                          tnlConstantFunction.h  -  description
                             -------------------
    begin                : Aug 2, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <iostream>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Functions {   

template< int dimensions,
          typename Real = double >
class tnlConstantFunction : public Domain< dimensions, NonspaceDomain >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< dimensions, RealType > VertexType;
 
      tnlConstantFunction();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setConstant( const RealType& constant );

      const RealType& getConstant() const;

   #ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
   #else
      template< int XDiffOrder,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
   #endif
      __cuda_callable__ inline
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;

      __cuda_callable__ inline
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const
      {
         return constant;
      }
 
       __cuda_callable__ inline
      RealType getValue( const Real& time = 0.0 ) const
      {
          return constant;
      }

   protected:

      RealType constant;
};

template< int dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const tnlConstantFunction< dimensions, Real >& f )
{
   str << "Constant function: constant = " << f.getConstant();
   return str;
}

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/ConstantFunction_impl.h>

