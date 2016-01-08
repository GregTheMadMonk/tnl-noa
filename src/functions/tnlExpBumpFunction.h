/***************************************************************************
                          tnlExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLEXPBUMPFUNCTION_H_
#define TNLEXPBUMPFUNCTION_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlDomain.h>

template< int dimensions,
          typename Real >
class tnlExpBumpFunctionBase : public tnlDomain< dimensions, SpaceDomain >
{
   public:
     
      typedef Real RealType;
      
      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

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
      typedef tnlStaticVector< 1, RealType > VertexType;      

      static tnlString getType();

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
      typedef tnlStaticVector< 2, RealType > VertexType;      

      static tnlString getType();

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
      typedef tnlStaticVector< 3, RealType > VertexType;      

    
      static tnlString getType();

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
ostream& operator << ( ostream& str, const tnlExpBumpFunction< Dimensions, Real >& f )
{
   str << "ExpBump. function: amplitude = " << f.getAmplitude() << " sigma = " << f.getSigma();
   return str;
}

#include <functions/tnlExpBumpFunction_impl.h>


#endif /* TNLEXPBUMPFUNCTION_H_ */
