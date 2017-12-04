/* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <TNL/TypeInfo.h>

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
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
         const auto& neighbors = cell.getNeighborEntities();
         const RealType& h = mesh.getSpaceSteps().x();
         //const IndexType& c = cell.getIndex();
         const IndexType e = neighbors.template getEntityIndex<  1 >();
         const IndexType w = neighbors.template getEntityIndex< -1 >();

         /*if( c * input[ e ] <= 0 || c * input[ w ] <= 0 )
         {
            output[ cell.getIndex() ] = c;
            interfaceMap[ cell.getIndex() ] = true;
            continue;
         }*/
         if( c * input[ e ] <=0 )
         {
             /*if( c >= 0 )
             {
                output[ cell.getIndex() ] = ( h * c )/( c - input[ e ] );
             }
             else
             {
                 output[ cell.getIndex() ] = - ( h * c )/( c - input[ e ] );
             }*/
             output[ cell.getIndex() ] =
             c >= 0 ? ( h * c )/( c - input[ e ] ) :
                      - ( h * c )/( c - input[ e ] );
             interfaceMap[ cell.getIndex() ] = true;
             continue;
         }
         if( c * input[ w ] <=0 )
         {
             output[ cell.getIndex() ] =
             c >= 0 ? ( h * c )/( c - input[ w ] ) :
                      - ( h * c )/( c - input[ w ] );
             interfaceMap[ cell.getIndex() ] = true;
             continue;
         }
      }
      output[ cell.getIndex() ] =
      c > 0 ? TNL::TypeInfo< RealType >::getMaxValue() :
             -TypeInfo< RealType >::getMaxValue();
      interfaceMap[ cell.getIndex() ] = false;
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
}


template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
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
          output[ cell.getIndex() ] =
              input( cell ) >= 0 ? TypeInfo< RealType >::getMaxValue() :
                                   - TypeInfo< RealType >::getMaxValue();
   
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
            auto neighbors = cell.getNeighborEntities();
            const RealType& hx = mesh.getSpaceSteps().x();
            const RealType& hy = mesh.getSpaceSteps().y();
            Real pom = 0;
            const IndexType e = neighbors.template getEntityIndex<  1,  0 >();
            //const IndexType w = neighbors.template getEntityIndex< -1,  0 >();
            const IndexType n = neighbors.template getEntityIndex<  0,  1 >();
            //const IndexType s = neighbors.template getEntityIndex<  0, -1 >();            
            /*if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                c * input[ n ] <= 0 || c * input[ s ] <= 0 )
            {
               output[ cell.getIndex() ] = TypeInfo< RealType >::getMaxValue();
               interfaceMap[ cell.getIndex() ] = true;
               continue;
            }*/
            if( c * input[ n ] <= 0 )
            {
                if( c >= 0 )
                {
                    pom = ( hy * c )/( c - input[ n ]);
                    if( output[ cell.getIndex() ] > pom ) 
                        output[ cell.getIndex() ] = pom;
                    
                    output[ n ] = ( hy * c )/( c - input[ n ]) - hy;
                }else
                {
                    pom = - ( hy * c )/( c - input[ n ]);
                    if( output[ cell.getIndex() ] < pom )
                        output[ cell.getIndex() ] = pom;
                    
                    output[ n ] = hy - ( hy * c )/( c - input[ n ]);
                }
                interfaceMap[ cell.getIndex() ] = true;
                interfaceMap[ n ] = true;
                continue;
            }
            if( c * input[ e ] <= 0 )
            {
                if( c >= 0 )
                {
                    pom = ( hx * c )/( c - input[ e ]);
                    if( output[ cell.getIndex() ] > pom )
                        output[ cell.getIndex() ] = pom;
                   
                    pom = pom - hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                    if( output[ e ] != 0 && output[ e ] >= pom )
                        output[ e ] = pom;                         
                }else
                {
                    pom = - (hx * c)/( c - input[ e ]);
                    if( output[ cell.getIndex() ] < pom )
                        output[ cell.getIndex() ] = pom;
                    
                    pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                    if( output[ e ] < pom )
                        output[ e ] = pom;
                }
                interfaceMap[ cell.getIndex() ] = true;
                interfaceMap[ e ] = true;
                continue;
            }
         }
         output[ cell.getIndex() ] =
            c > 0 ? TypeInfo< RealType >::getMaxValue() :
                   -TypeInfo< RealType >::getMaxValue();  
         interfaceMap[ cell.getIndex() ] = false;
      }
}

/*template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
   const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& h = mesh.getSpaceSteps().x(); 
   const RealType value = u( cell );
   Real a, b, tmp;

   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
                     u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
   }

   if( cell.getCoordinates().y() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
   else
   {
      b = argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
                     u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
   }

   if( fabs( a ) == TypeInfo< Real >::getMaxValue() && 
       fabs( b ) == TypeInfo< Real >::getMaxValue() )
      return;
   if( fabs( a ) == TypeInfo< Real >::getMaxValue() ||
       fabs( b ) == TypeInfo< Real >::getMaxValue() ||
       fabs( a - b ) >= h )
   {
      tmp = argAbsMin( a, b ) + sign( value ) * h;*/
      /*   std::cerr << "a = " << a << " b = " << b << " h = " << h 
             << " ArgAbsMin( a, b ) = " << ArgAbsMin( a, b ) << " sign( value ) = " << sign( value )
             << " sign( value ) * h = " << sign( value ) * h
             << " ArgAbsMin( a, b ) + sign( value ) * h = " << ArgAbsMin( a, b ) + sign( value ) * h           
             << " tmp = " << tmp << std::endl;
      tmp = ArgAbsMin( a, b ) + sign( value ) * h;
      tmp = ArgAbsMin( a, b ) + sign( value ) * h;
      tmp = ArgAbsMin( a, b ) + sign( value ) * h;
      res = ArgAbsMin( a, b ) + sign( value ) * h;
      std::cerr << " tmp = " << tmp << std::endl;
      std::cerr << " res = " << res << std::endl;*/

   /*}
   else
      tmp = 0.5 * ( a + b + sign( value ) * sqrt( 2.0 * h * h - ( a - b ) * ( a - b ) ) );

   u[ cell.getIndex() ] = argAbsMin( value, tmp );
   //std::cerr << ArgAbsMin( value, tmp ) << " ";   
}*/

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell)
            //MeshFunctionType& v
{
   const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();
   const RealType value = u( cell );
   //const RealType permeability = v( cell );
   Real a, b, tmp;
   
   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
                     u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
   }

   if( cell.getCoordinates().y() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
   else
   {
      b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
                     u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
   }
   if( fabs( a ) == TypeInfo< Real >::getMaxValue() && 
       fabs( b ) == TypeInfo< Real >::getMaxValue() )
      return;
   if( fabs( a ) == TypeInfo< Real >::getMaxValue() ||
       fabs( b ) == TypeInfo< Real >::getMaxValue() ||
       fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/1 ) ) //permeability ) )
   {
      tmp = 
        fabs( a ) >= fabs( b ) ? b + TNL::sign( value ) * hy :
                                 a + TNL::sign( value ) * hx;
   }
   else
      tmp = ( hx * hx * a + hy * hy * b + 
            sign( value ) * hx * hy * sqrt( ( hx * hx + hy * hy )/1 - 
            ( a - b ) * ( a - b ) ) )/( hx * hx + hy * hy ); //permeability
            
   
   u[ cell.getIndex() ] = argAbsMin( value, tmp );
}

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
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
               auto neighbors = cell.getNeighborEntities();
               //const IndexType& c = cell.getIndex();
               const IndexType e = neighbors.template getEntityIndex<  1,  0,  0 >();
               const IndexType w = neighbors.template getEntityIndex< -1,  0,  0 >();
               const IndexType n = neighbors.template getEntityIndex<  0,  1,  0 >();
               const IndexType s = neighbors.template getEntityIndex<  0, -1,  0 >();
               const IndexType t = neighbors.template getEntityIndex<  0,  0,  1 >();
               const IndexType b = neighbors.template getEntityIndex<  0,  0, -1 >();

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
               c > 0 ? TypeInfo< RealType >::getMaxValue() :
                      -TypeInfo< RealType >::getMaxValue();
            interfaceMap[ cell.getIndex() ] = false;
         }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
   
}
