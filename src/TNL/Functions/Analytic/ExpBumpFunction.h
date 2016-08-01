/***************************************************************************
                          ExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Functions {   

template< int dimensions,
          typename Real >
class ExpBumpFunctionBase : public Domain< dimensions, SpaceDomain >
{
   public:
 
      typedef Real RealType;
 
      ExpBumpFunctionBase();
 
      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setAmplitude( const RealType& amplitude );

      const RealType& getAmplitude() const;

      void setSigma( const RealType& sigma );

      const RealType& getSigma() const;

   protected:

      RealType amplitude, sigma;
};

template< int Dimensions,
          typename Real >
class ExpBumpFunction
{
};

template< typename Real >
class ExpBumpFunction< 1, Real > : public ExpBumpFunctionBase< 1, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 1, RealType > VertexType;

      static String getType();

      ExpBumpFunction();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
   __cuda_callable__
   RealType getPartialDerivative( const VertexType& v,
                                  const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const VertexType& v,
                        const RealType& time = 0.0 ) const;
};

template< typename Real >
class ExpBumpFunction< 2, Real > : public ExpBumpFunctionBase< 2, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 2, RealType > VertexType;

      static String getType();

      ExpBumpFunction();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
   __cuda_callable__ inline
   RealType getPartialDerivative( const VertexType& v,
                                  const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const VertexType& v,
                        const Real& time = 0.0 ) const;
};

template< typename Real >
class ExpBumpFunction< 3, Real > : public ExpBumpFunctionBase< 3, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 3, RealType > VertexType;

 
      static String getType();

      ExpBumpFunction();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
   __cuda_callable__
   RealType getPartialDerivative( const VertexType& v,
                                  const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const VertexType& v,
                        const Real& time = 0.0 ) const;
 
};

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const ExpBumpFunction< Dimensions, Real >& f )
{
   str << "ExpBump. function: amplitude = " << f.getAmplitude() << " sigma = " << f.getSigma();
   return str;
}

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/ExpBumpFunction_impl.h>


