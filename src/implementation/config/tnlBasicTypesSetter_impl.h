/***************************************************************************
                          tnlBasicTypesSetter_impl.h  -  description
                             -------------------
    begin                : Feb 23, 2013
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
 
#include <core/tnlHost.h>
#include <core/tnlCuda.h>
#include <solvers/ode/tnlEulerSolver.h>

template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: run( const tnlParameterContainer& parameters )
{
   return this -> setRealType( parameters );
};

template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: checkSupportedRealTypes( const tnlString& realType,
                                                             const tnlParameterContainer& parameters ) const
{
   return true;
}

template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: checkSupportedIndexTypes( const tnlString& indexType,
                                                              const tnlParameterContainer& parameters ) const
{
   return true;
}

/*template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: checkSupportedDimensions( const int dimensions,
                                                              const tnlParameterContainer& parameters ) const
{
   return true;
}


template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: checkSupportedDiscreteSolvers( const tnlString& timeDiscretisation,
                                                                   const tnlParameterContainer& parameters ) const
{
   return true;
}

template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: checkSupportedTimeDiscretisations( const tnlString& timeDiscretisation,
                                                                       const tnlParameterContainer& parameters ) const
{
   return true;
}*/


template< typename ProblemTypesSetter >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: setRealType( const tnlParameterContainer& parameters ) const
{
   const tnlString& realType = parameters. GetParameter< tnlString >( "real-type" );
   if( ! checkSupportedRealTypes( realType, parameters ) )
   {
      cerr << "The real type '" << realType << "' is not supported." << endl;
      return false;
   }
   if( realType == "float" )
      return setIndexType< float >( parameters );
   if( realType == "double" )
      return setIndexType< double >( parameters );
   if( realType == "long-double" )
      return setIndexType< long double >( parameters );
   cerr << "The real type '" << realType << "' is not defined. " << endl;
   return false;
}

template< typename ProblemTypesSetter >
   template< typename RealType >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: setIndexType( const tnlParameterContainer& parameters ) const
{
   const tnlString& indexType = parameters. GetParameter< tnlString >( "index-type" );
   if( ! checkSupportedIndexTypes( indexType, parameters ) )
   {
      cerr << "The index type '" << indexType << "' is not supported." << endl;
      return false;
   }
   if( indexType == "int" )
      return setDeviceType< RealType, int >( parameters );
   if( indexType == "long int" )
      return setDeviceType< RealType, long int >( parameters );
   cerr << "The index type '" << indexType << "' is not defined. " << endl;
   return false;
}

template< typename ProblemTypesSetter >
   template< typename RealType,
             typename IndexType >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: setDeviceType( const tnlParameterContainer& parameters ) const
{
   const tnlString& device = parameters. GetParameter< tnlString >( "device" );
   if( ! checkSupportedDevices( device, parameters ) )
   {
      cerr << "The device '" << device << "' is not supported." << endl;
      return false;
   }
   ProblemTypesSetter problemTypesSetter;
   if( device == "host" )
      return problemTypesSetter. run< RealType, tnlHost, IndexType >( parameters );
   if( device == "cuda" )
      return problemTypesSetter. run( parameters );
   cerr << "The device '" << device << "' is not defined. " << endl;
   return false;
}

/*
template< typename ProblemTypesSetter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: setDimensions( const tnlParameterContainer& parameters ) const
{
   const int dimensions = parameters. GetParameter< int >( "dimensions" );
   if( dimensions < 1 || dimensions > 4 || ! checkSupportedDimensions( dimensions, parameters ) )
   {
      cerr << dimensions << " dimensions are not supported." << endl;
      return false;
   }

   if( dimensions == 1 )
      return ProblemTemplateParametersSetter :: setParameters< 1, RealType, DeviceType, IndexType >( parameters, this );
}

template< typename ProblemTypesSetter >
          typename ProblemTemplateParametersSetter  >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: setDiscreteSolver( const tnlParameterContainer& parameters ) const
{
   const tnlString& timeDiscretisation = parameters. GetParameter< tnlString >( "time-discretisation" );
   if( ! checkSupportedTimeDiscretisations( timeDiscretisation ) )
   {
      cerr << timeDiscretisation << " time discretisation is not supported." << endl;
      return false;
   }
   tnlString discreteSolver = parameters. GetParameter< tnlString >( "discrete-solver" );
   if( ! checkSupportedDiscreteSolvers( discreteSolver ) )
   {
      cerr << discreteSolver << " solver is not supported." << endl;
      return false;
   }

   if( ( ( discreteSolver == "euler" ||
           discreteSolver == "merson" ) &&
              timeDiscretisation != "explicit" ) ||
       ( ( discreteSolver == "sor" ||
           discreteSolver == "cg" ||
           discreteSolver == "bicg-stab" ||
           discreteSolver == "gmres" ||
           discreteSolver == "tfqmr" ) &&
              timeDiscretisation != "semi-implicit" ) )
   {
      cerr << "The solver " << discreteSolver << " is not compatible with the "
           << timeDiscretisation << " time discretisation." << endl;
      return false;
   }
   if( discreteSolver == "euler" )
   {

   }
}


template< typename ProblemTypesSetter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool tnlBasicTypesSetter< ProblemTypesSetter > :: setTimeDiscretisation( const tnlParameterContainer& parameters ) const
{
   tnlString device = parameters. GetParameter< tnlString >( "device" );
   if( ! checkSupportedDevices( device, parameters ) )
   {
      cerr << "The device '" << device << "' is not supported." << endl;
      return false;
   }
   if( device == "host" )
      return setTimeDiscretisation< RealType, tnlHost, IndexType >( parameters );
   if( indexType == "cuda" )
      return setTimeDiscretisation< RealType, tnlCuda, IndexType >( parameters );
   cerr << "The device '" << device << "' is not defined. " << endl;
   return false;
}
*/
