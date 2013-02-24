/***************************************************************************
                          tnlBasicTypesSetter.h  -  description
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

#ifndef TNLBASICTYPESSETTERR_H_
#define TNLBASICTYPESSETTERR_H_

#include <core/tnlObject.h>
#include <config/tnlParameterContainer.h>
#include <config/tnlDefaultBasicTypesChecker.h>

template< typename ProblemTypesSetter,
          typename ProblemTypesChecker = tnlDefaultBasicTypesChecker >
class tnlBasicTypesSetter : public tnlObject
{
   public:

   bool run( const tnlParameterContainer& parameters );

   /*virtual bool checkSupportedDimensions( const int dimensions,
                                          const tnlParameterContainer& parameters ) const;


   virtual bool checkSupportedDiscreteSolvers( const tnlString& timeDiscretisation,
                                              const tnlParameterContainer& parameters ) const;

   virtual bool checkSupportedTimeDiscretisations( const tnlString& timeDiscretisation,
                                                   const tnlParameterContainer& parameters ) const;*/

   protected:

   bool setRealType( const tnlParameterContainer& parameters ) const;

   template< typename RealType >
   bool setIndexType( const tnlParameterContainer& parameters ) const;

   template< typename RealType,
             typename IndexType >
   bool setDeviceType( const tnlParameterContainer& parameters ) const;

   /*template< typename RealType,
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

#include <implementation/config/tnlBasicTypesSetter_impl.h>

#endif /* TNLBASICTYPESSETTER_H_ */
