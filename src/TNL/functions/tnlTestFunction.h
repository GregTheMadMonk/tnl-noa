/***************************************************************************
                          tnlTestFunction.h  -  description
                             -------------------
    begin                : Aug 2, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlHost.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/functions/tnlDomain.h>

namespace TNL {

template< int FunctionDimensions,
          typename Real = double,
          typename Device = tnlHost >
class tnlTestFunction : public tnlDomain< FunctionDimensions, SpaceDomain >
{
   protected:

   enum TestFunctions{ constant,
                       expBump,
                       sinBumps,
                       sinWave,
		       cylinder,
		       flowerpot,
		       twins,
           pseudoSquare,
           blob };

   enum TimeDependence { none,
                         linear,
                         quadratic,
                         cosine };

   public:

   enum{ Dimensions = FunctionDimensions };
   typedef Real RealType;
   typedef Vectors::StaticVector< Dimensions, Real > VertexType;

   tnlTestFunction();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   const tnlTestFunction& operator = ( const tnlTestFunction& function );

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
   Real getPartialDerivative( const VertexType& vertex,
                              const Real& time = 0 ) const;

   __cuda_callable__
   Real operator()( const VertexType& vertex,
                  const Real& time = 0 ) const
   {
      return this->getPartialDerivative< 0, 0, 0 >( vertex, time );
   }


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
   Real getTimeDerivative( const VertexType& vertex,
                           const Real& time = 0 ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
   __cuda_callable__
   Real getTimeDerivative( const Vertex& vertex,
                           const Real& time = 0 ) const
   {
      return this->getTimeDerivative< 0, 0, 0, Vertex >( vertex, time );
   }
#endif

   std::ostream& print( std::ostream& str ) const;

   ~tnlTestFunction();

   protected:

   template< typename FunctionType >
   bool setupFunction( const Config::ParameterContainer& parameters,
                      const String& prefix = "" );

   template< typename FunctionType >
   void deleteFunction();

   void deleteFunctions();

   template< typename FunctionType >
   void copyFunction( const void* function );

   template< typename FunctionType >
   std::ostream& printFunction( std::ostream& str ) const;

   void* function;

   TestFunctions functionType;

   TimeDependence timeDependence;

   Real timeScale;

};

template< int FunctionDimensions,
          typename Real,
          typename Device >
std::ostream& operator << ( std::ostream& str, const tnlTestFunction< FunctionDimensions, Real, Device >& f )
{
   str << "Test function: ";
   return f.print( str );
}

} // namespace TNL

#include <TNL/functions/tnlTestFunction_impl.h>

