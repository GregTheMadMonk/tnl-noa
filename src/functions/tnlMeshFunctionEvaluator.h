/***************************************************************************
                          tnlMeshFunctionEvaluator.h  -  description
                             -------------------
    begin                : Jan 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLMESHFUNCTIONEVALUATOR_H
#define	TNLMESHFUNCTIONEVALUATOR_H

#include <mesh/tnlGrid.h>
#include <functions/tnlMeshFunction.h>

template< typename OutMeshFunction,
          typename InFunction >
class tnlMeshFunctionEvaluator
{
   public:
      typedef typename OutMeshFunction::MeshType MeshType;
      typedef typename MeshType::RealType MeshRealType;
      typedef typename MeshType::DeviceType MeshDeviceType;
      typedef typename MeshType::IndexType MeshIndexType;
      typedef typename OutMeshFunction::Real RealType;
      
      const static int meshEntityDimensions = OutMeshFunction::entityDimensions;
      
      static_assert( MeshType::meshDimensions == InFunction::Dimensions, 
         "Input function and the mesh of the mesh function have both different number of dimensions." );
      
      static void assign( OutMeshFunction& meshFunction,
                          const InFunction& function,                          
                          const RealType& time = 0.0,
                          const RealType& outFunctionMultiplicator = 0.0,
                          const RealType& inFunctionMultiplicator = 1.0 );
      
      class TraverserUserData
      {
         public:
            TraverserUserData( const InFunction* function,
                               const RealType* time,
                               OutMeshFunction* meshFunction,
                               const RealType* outFunctionMultiplicator,
                               const RealType* inFunctionMultiplicator )
            : meshFunction( meshFunction ), function( function ), time( time ), 
              outFunctionMultiplicator( outFunctionMultiplicator ),
              inFunctionMultiplicator( inFunctionMultiplicator ){}

         protected:
            OutMeshFunction* meshFunction;            
            const InFunction* function;
            const RealType *time, *outFunctionMultiplicator, *inFunctionMultiplicator;
            
      };
}; 

template< int Dimensions,
          typename MeshReal1,
          typename MeshReal2,
          typename MeshDevice1,
          typename MeshDevice2,
          typename MeshIndex1,
          typename MeshIndex2,
          int MeshEntityDimensions,
          typename Real >
class tnlMeshFunctionEvaluator< tnlMeshFunction< tnlGrid< Dimensions, MeshReal1, MeshDevice1, MeshIndex1 >, MeshEntityDimensions, Real >,
                                tnlMeshFunction< tnlGrid< Dimensions, MeshReal2, MeshDevice2, MeshIndex2 >, MeshEntityDimensions, Real > >
{
   public:
      
      typedef tnlGrid< Dimensions, MeshReal1, MeshDevice1, MeshIndex1 > Mesh1;
      typedef tnlGrid< Dimensions, MeshReal2, MeshDevice2, MeshIndex2 > Mesh2;
      typedef tnlMeshFunction< Mesh1, MeshEntityDimensions, Real > OutMeshFunction;
      typedef tnlMeshFunction< Mesh2, MeshEntityDimensions, Real > InFunction;
      
      static void assign( OutMeshFunction& f1,
                          const InFunction& f2 )
      {
         if( f1.getMesh().getDimensions() == f2.getMesh().getDimensions() )
            f1.getData() = f2.getData();
         else
         {
            //TODO: Interpolace
         }
      };
};

#endif	/* TNLMESHFUNCTIONEVALUATOR_H */

