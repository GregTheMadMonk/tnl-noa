/***************************************************************************
                          SinWaveFunction.h  -  description
                             -------------------
    begin                : Nov 19, 2013
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
          typename Real = double >
class SinWaveFunctionBase : public Domain< dimensions, SpaceDomain >
{
   public:
 
   SinWaveFunctionBase();

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setWaveLength( const Real& waveLength );

   Real getWaveLength() const;

   void setAmplitude( const Real& amplitude );

   Real getAmplitude() const;

   void setPhase( const Real& phase );

   Real getPhase() const;

   protected:

   Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class SinWaveFunction
{
};

template< typename Real >
class SinWaveFunction< 1, Real > : public SinWaveFunctionBase< 1, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 1, RealType > VertexType;

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

template< typename Real >
class SinWaveFunction< 2, Real > : public SinWaveFunctionBase< 2, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 2, RealType > VertexType;
 
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

template< typename Real >
class SinWaveFunction< 3, Real > : public SinWaveFunctionBase< 3, Real >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 3, RealType > VertexType;


 
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
std::ostream& operator << ( std::ostream& str, const SinWaveFunction< Dimensions, Real >& f )
{
   str << "Sin Wave. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/SinWaveFunction_impl.h>

