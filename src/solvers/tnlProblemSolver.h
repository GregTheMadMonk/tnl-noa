/***************************************************************************
                          tnlProblemSolver.h  -  description
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

#ifndef TNLPROBLEMSOLVER_H_
#define TNLPROBLEMSOLVER_H_

#include <core/tnlObject.h>

template< typename ProblemSetter >
class tnlProblemSolver : public tnlObject
{
   public:

   bool run( const char* configFileName, int argc, char* argv[] );

   /*virtual bool checkSupportedRealTypes( const tnlString& realType,
                                         const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedIndexTypes( const tnlString& indexType,
                                          const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedDevices( const tnlString& device,
                                       const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedDimensions( const int dimensions,
                                          const tnlParameterContainer& parameters ) const;


   virtual bool checkSupportedDiscreteSolvers( const tnlString& timeDiscretisation,
                                              const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedTimeDiscretisations( const tnlString& timeDiscretisation,
                                                   const tnlParameterContainer& parameters ) const;

   protected:

   bool setRealType( const tnlParameterContainer& parameters ) const;

   template< typename RealType >
   bool setIndexType( const tnlParameterContainer& parameters ) const;

   template< typename RealType,
             typename IndexType >
   bool setDeviceType( const tnlParameterContainer& parameters ) const;

   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   bool setDimensions( const tnlParameterContainer& parameters ) const;


   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   bool setupProblem( const tnlParameterContainer& parameters ) const;


   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   bool setDiscreteSolver( const tnlParameterContainer& parameters ) const;


   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   bool setTimeDiscretisation( const tnlParameterContainer& parameters ) const;*/
};

#include <implementation/solvers/tnlProblemSolver_impl.h>

#endif /* TNLPROBLEMSOLVER_H_ */
