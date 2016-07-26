/***************************************************************************
                          tnlExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/functions/tnlDomain.h>

namespace TNL {

template< int dimensions,
          typename Real >
class tnlExpBumpFunctionBase : public tnlDomain< dimensions, SpaceDomain >
{
   public:
 
      typedef Real RealType;
 
      tnlExpBumpFunctionBase();
 
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
class tnlExpBumpFunction
{
};

template< typename Real >
class tnlExpBumpFunction< 1, Real > : public tnlExpBumpFunctionBase< 1, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 1, RealType > VertexType;

      static String getType();

      tnlExpBumpFunction();

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
class tnlExpBumpFunction< 2, Real > : public tnlExpBumpFunctionBase< 2, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 2, RealType > VertexType;

      static String getType();

      tnlExpBumpFunction();

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
class tnlExpBumpFunction< 3, Real > : public tnlExpBumpFunctionBase< 3, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 3, RealType > VertexType;

 
      static String getType();

      tnlExpBumpFunction();

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
std::ostream& operator << ( std::ostream& str, const tnlExpBumpFunction< Dimensions, Real >& f )
{
   str << "ExpBump. function: amplitude = " << f.getAmplitude() << " sigma = " << f.getSigma();
   return str;
}

} // namespace TNL

#include <TNL/functions/tnlExpBumpFunction_impl.h>


