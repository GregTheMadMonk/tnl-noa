/***************************************************************************
                          tnlTestFunction.h  -  description
                             -------------------
    begin                : Aug 2, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLTESTFUNCTION_H_
#define TNLTESTFUNCTION_H_

#include <core/tnlHost.h>
#include <core/vectors/tnlStaticVector.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

template< int FunctionDimensions,
          typename Real = double,
          typename Device = tnlHost >
class tnlTestFunction
{
   protected:

   enum TestFunctions{ constant,
                       expBump,
                       sinBumps,
                       sinWave };

   enum TimeDependence { none,
                         linear,
                         quadratic,
                         cosine };

   public:

   enum{ Dimensions = FunctionDimensions };
   typedef Real RealType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;

   tnlTestFunction();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   const tnlTestFunction& operator = ( const tnlTestFunction& function );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getValue( const Vertex& vertex,
                  const Real& time = 0 ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
   Real getValue( const Vertex& vertex,
                  const Real& time = 0 ) const
   {
      return this->getValue< 0, 0, 0, Vertex >( vertex, time );
   }
#endif                  

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getTimeDerivative( const Vertex& vertex,
                           const Real& time = 0 ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
   Real getTimeDerivative( const Vertex& vertex,
                           const Real& time = 0 ) const
   {
      return this->getTimeDerivative< 0, 0, 0, Vertex >( vertex, time );
   }   
#endif                              

   ostream& print( ostream& str ) const;

   ~tnlTestFunction();

   protected:

   template< typename FunctionType >
   bool setupFunction( const tnlParameterContainer& parameters,
                      const tnlString& prefix = "" );

   template< typename FunctionType >
   void deleteFunction();

   void deleteFunctions();

   template< typename FunctionType >
   void copyFunction( const void* function );

   template< typename FunctionType >
   ostream& printFunction( ostream& str ) const;

   void* function;

   TestFunctions functionType;

   TimeDependence timeDependence;

   Real timeScale;

};

template< int FunctionDimensions,
          typename Real,
          typename Device >
ostream& operator << ( ostream& str, const tnlTestFunction< FunctionDimensions, Real, Device >& f )
{
   str << "Test function: ";
   return f.print( str );
}

#include <functions/tnlTestFunction_impl.h>

#endif /* TNLTESTFUNCTION_H_ */
