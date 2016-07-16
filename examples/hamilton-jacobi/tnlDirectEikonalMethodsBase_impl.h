/* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <core/tnlTypeInfo.h>
#include <functions/tnlFunctions.h>

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > >::
initInterface( const MeshFunctionType& input,
               MeshFunctionType& output,
               InterfaceMapType& interfaceMap  )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().x() = 1;
        cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
        cell.getCoordinates().x() ++ )
   {
      cell.refresh();
      const RealType& c = input( cell );      
      if( ! cell.isBoundaryEntity()  )
      {
         const auto& neighbours = cell.getNeighbourEntities();
         //const IndexType& c = cell.getIndex();
         const IndexType e = neighbours.template getEntityIndex<  1 >();
         const IndexType w = neighbours.template getEntityIndex< -1 >();

         if( c * input[ e ] <= 0 || c * input[ w ] <= 0 )
         {
            output[ cell.getIndex() ] = c;
            interfaceMap[ cell.getIndex() ] = true;
            continue;
         }
      }
      output[ cell.getIndex() ] =
      c > 0 ? tnlTypeInfo< RealType >::getMaxValue() :
             -tnlTypeInfo< RealType >::getMaxValue();
      interfaceMap[ cell.getIndex() ] = false;
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
}


template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > >::
initInterface( const MeshFunctionType& input,
               MeshFunctionType& output,
               InterfaceMapType& interfaceMap  )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < mesh.getDimensions().y();
        cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < mesh.getDimensions().x();
           cell.getCoordinates().x() ++ )
      {
         cell.refresh();
         const RealType& c = input( cell );
         if( ! cell.isBoundaryEntity()  )
         {
            auto neighbours = cell.getNeighbourEntities();
            const IndexType e = neighbours.template getEntityIndex<  1,  0 >();
            const IndexType w = neighbours.template getEntityIndex< -1,  0 >();
            const IndexType n = neighbours.template getEntityIndex<  0,  1 >();
            const IndexType s = neighbours.template getEntityIndex<  0, -1 >();            
            if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                c * input[ n ] <= 0 || c * input[ s ] <= 0 )
            {
               output[ cell.getIndex() ] = c;
               interfaceMap[ cell.getIndex() ] = true;
               continue;
            }
         }
         output[ cell.getIndex() ] =
            c > 0 ? tnlTypeInfo< RealType >::getMaxValue() :
                   -tnlTypeInfo< RealType >::getMaxValue();  
         interfaceMap[ cell.getIndex() ] = false;
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
   const auto& neighbourEntities = cell.template getNeighbourEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& h = mesh.getSpaceSteps().x(); 
   const RealType value = u( cell );
   Real a, b, tmp;

   if( cell.getCoordinates().x() == 0 )
      a = u[ neighbourEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighbourEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = ArgAbsMin( u[ neighbourEntities.template getEntityIndex< -1,  0 >() ],
                     u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] );
   }

   if( cell.getCoordinates().y() == 0 )
      b = u[ neighbourEntities.template getEntityIndex< 0,  1 >()];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighbourEntities.template getEntityIndex< 0,  -1 >() ];
   else
   {
      b = ArgAbsMin( u[ neighbourEntities.template getEntityIndex< 0,  -1 >() ],
                     u[ neighbourEntities.template getEntityIndex< 0,   1 >() ] );
   }

   if( fabs( a ) == tnlTypeInfo< Real >::getMaxValue() && 
       fabs( b ) == tnlTypeInfo< Real >::getMaxValue() )
      return;
   if( fabs( a ) == tnlTypeInfo< Real >::getMaxValue() ||
       fabs( b ) == tnlTypeInfo< Real >::getMaxValue() ||
       fabs( a - b ) >= h )
   {
      tmp = ArgAbsMin( a, b ) + Sign( value ) * h;
      /*   std::cerr << "a = " << a << " b = " << b << " h = " << h 
             << " ArgAbsMin( a, b ) = " << ArgAbsMin( a, b ) << " Sign( value ) = " << Sign( value )
             << " Sign( value ) * h = " << Sign( value ) * h
             << " ArgAbsMin( a, b ) + Sign( value ) * h = " << ArgAbsMin( a, b ) + Sign( value ) * h           
             << " tmp = " << tmp << std::endl;
      tmp = ArgAbsMin( a, b ) + Sign( value ) * h;
      tmp = ArgAbsMin( a, b ) + Sign( value ) * h;
      tmp = ArgAbsMin( a, b ) + Sign( value ) * h;
      res = ArgAbsMin( a, b ) + Sign( value ) * h;
      std::cerr << " tmp = " << tmp << std::endl;
      std::cerr << " res = " << res << std::endl;*/

   }
   else
      tmp = 0.5 * ( a + b + Sign( value ) * sqrt( 2.0 * h * h - ( a - b ) * ( a - b ) ) );

   u[ cell.getIndex() ] = ArgAbsMin( value, tmp );
   //std::cerr << ArgAbsMin( value, tmp ) << " ";   
}


template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > >::
initInterface( const MeshFunctionType& input,
               MeshFunctionType& output,
               InterfaceMapType& interfaceMap  )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().z() = 1;
        cell.getCoordinates().z() < mesh.getDimensions().z() - 1;
        cell.getCoordinates().z() ++ )   
      for( cell.getCoordinates().y() = 1;
           cell.getCoordinates().y() < mesh.getDimensions().y() - 1;
           cell.getCoordinates().y() ++ )
         for( cell.getCoordinates().x() = 1;
              cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
              cell.getCoordinates().x() ++ )
         {
            cell.refresh();
            const RealType& c = input( cell );
            if( ! cell.isBoundaryEntity() )
            {
               auto neighbours = cell.getNeighbourEntities();
               //const IndexType& c = cell.getIndex();
               const IndexType e = neighbours.template getEntityIndex<  1,  0,  0 >();
               const IndexType w = neighbours.template getEntityIndex< -1,  0,  0 >();
               const IndexType n = neighbours.template getEntityIndex<  0,  1,  0 >();
               const IndexType s = neighbours.template getEntityIndex<  0, -1,  0 >();
               const IndexType t = neighbours.template getEntityIndex<  0,  0,  1 >();
               const IndexType b = neighbours.template getEntityIndex<  0,  0, -1 >();

               if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                   c * input[ n ] <= 0 || c * input[ s ] <= 0 ||
                   c * input[ t ] <= 0 || c * input[ b ] <= 0 )
               {
                  output[ cell.getIndex() ] = c;
                  interfaceMap[ cell.getIndex() ] = true;
                  continue;
               }
            }
            output[ cell.getIndex() ] =
               c > 0 ? tnlTypeInfo< RealType >::getMaxValue() :
                      -tnlTypeInfo< RealType >::getMaxValue();
            interfaceMap[ cell.getIndex() ] = false;
         }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
   
}
