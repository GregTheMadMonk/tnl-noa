/***************************************************************************
                          tnlSolverInitiator.h  -  description
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

#ifndef TNLSOLVERINITIATOR_H_
#define TNLSOLVERINITIATOR_H_

#include <core/tnlObject.h>

template< typename ProblemSetter >
class tnlSolverInitiator : public tnlObject
{
   public:

   bool run( const char* configFileName, int argc, char* argv[] );

   virtual bool checkSupportedRealTypes( const tnlString& realType,
                                         const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedIndexTypes( const tnlString& indexType,
                                          const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedDevices( const tnlString& device,
                                       const tnlParameterContainer& parameters ) const;

   protected:

   bool setRealType( const tnlParameterContainer& parameters ) const;

   template< typename RealType >
   bool setIndexType( const tnlParameterContainer& parameters ) const;

   template< typename RealType,
             typename IndexType >
   bool setDeviceType( const tnlParameterContainer& parameters ) const;

   int verbose;
};

#include <implementation/solvers/tnlSolverInitiator_impl.h>

#endif /* TNLSOLVERINITIATOR_H_ */
