/***************************************************************************
                          MeshFunctionVTKWriter_impl.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunctionVTKWriter.h>
#include <TNL/Meshes/Readers/VTKEntityType.h>

namespace TNL {
namespace Functions {   

template< typename MeshFunction >
void
MeshFunctionVTKWriter< MeshFunction >::
writeHeader( const MeshFunction& function,
             std::ostream& str )
{
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshFunction >
bool
MeshFunctionVTKWriter< MeshFunction >::
write( const MeshFunction& function,
       std::ostream& str )
{
   writeHeader( function, str );

   using MeshType = typename MeshFunction::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimensions() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;
   using LocalIndex = typename MeshType::LocalIndexType;

   const MeshType& mesh = function.getMesh();
   const GlobalIndex verticesCount = mesh.template getEntitiesCount< 0 >();
   const GlobalIndex entitiesCount = mesh.template getEntitiesCount< MeshFunction::getEntitiesDimensions() >();
   const LocalIndex verticesPerEntity = EntityType::getVerticesCount();

   str << "POINTS " << verticesCount << " float" << std::endl;
   for( GlobalIndex i = 0; i < verticesCount; i++ ) {
      const auto& vertex = mesh.template getEntity< 0 >( i );
      const auto& point = vertex.getPoint();
      for( LocalIndex j = 0; j < point.size; j++ ) {
         str << point[ j ];
         if( j < point.size - 1 )
            str << " ";
      }
      // VTK needs zeros for unused dimensions
      for( LocalIndex j = 0; j < 3 - point.size; j++ )
         str << " 0";
      str << std::endl;
   }
 
   const GlobalIndex cellsListSize = entitiesCount * ( verticesPerEntity + 1 );
   str << std::endl << "CELLS " << entitiesCount << " " << cellsListSize << std::endl;
   for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
      const auto& entity = mesh.template getEntity< MeshFunction::getEntitiesDimensions() >( i );
      str << verticesPerEntity;
      for( LocalIndex j = 0; j < verticesPerEntity; j++ )
         str << " " << entity.template getSubentityIndex< 0 >( j );
      str << std::endl;
   }
 
   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
      const int type = (int) Meshes::Readers::TopologyToVTKMap< typename EntityType::EntityTopology >::type;
      str << type << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS cellFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;
   for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
      str << function.getData().getElement( i ) << std::endl;
   }
 
   return true;
}

template< typename Mesh,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Mesh, 0, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename Mesh,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Mesh, 0, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str )
{
   writeHeader( function, str );

   using MeshType = typename MeshFunctionType::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunctionType::getEntitiesDimensions() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;
   using LocalIndex = typename MeshType::LocalIndexType;

   const MeshType& mesh = function.getMesh();
   const GlobalIndex verticesCount = mesh.template getEntitiesCount< 0 >();
   const GlobalIndex entitiesCount = mesh.template getEntitiesCount< MeshFunctionType::getEntitiesDimensions() >();
   const LocalIndex verticesPerEntity = 1;

   str << "POINTS " << verticesCount << " float" << std::endl;
   for( GlobalIndex i = 0; i < verticesCount; i++ ) {
      const auto& vertex = mesh.template getEntity< 0 >( i );
      const auto& point = vertex.getPoint();
      for( LocalIndex j = 0; j < point.size; j++ ) {
         str << point[ j ];
         if( j < point.size - 1 )
            str << " ";
      }
      // VTK needs zeros for unused dimensions
      for( LocalIndex j = 0; j < 3 - point.size; j++ )
         str << " 0";
      str << std::endl;
   }
 
   const GlobalIndex cellsListSize = entitiesCount * ( verticesPerEntity + 1 );
   str << std::endl << "CELLS " << entitiesCount << " " << cellsListSize << std::endl;
   for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
      const auto& entity = mesh.template getEntity< MeshFunctionType::getEntitiesDimensions() >( i );
      str << verticesPerEntity << i << std::endl;
   }
 
   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
      const int type = (int) Meshes::Readers::TopologyToVTKMap< typename EntityType::EntityTopology >::type;
      str << type << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS cellFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;
   for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
      str << function.getData().getElement( i ) << std::endl;
   }
 
   return true;
}


/****
 * 1D grid, cells
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType origin = mesh.getOrigin().x();
   const RealType spaceStep = mesh.getSpaceSteps().x();
 
   str << "POINTS " << mesh.getDimensions().x() + 1 << " float" << std::endl;
   for (int i = 0; i <= mesh.getDimensions().x(); i++)
   {
       str << origin + i * spaceStep << " 0 0" << std::endl;
   }
 
   str << std::endl << "CELLS " << mesh.getDimensions().x() << " " << mesh.getDimensions().x() * 3 << std::endl;
   for (int i = 0; i < mesh.getDimensions().x(); i++)
   {
       str << "2 " << i << " " << i+1 << std::endl;
   }
 
   str << std::endl << "CELL_TYPES " << mesh.getDimensions().x() << std::endl;
   for (int i = 0; i < mesh.getDimensions().x(); i++)
   {
       str << "3 " << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << mesh.getDimensions().x() << std::endl;
   str << "SCALARS cellFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < mesh.template getEntitiesCount< typename MeshType::Cell >(); i++ )
   {
      typename MeshType::Cell entity = mesh.template getEntity< typename MeshType::Cell >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }
 
   return true;
}

/****
 * 1D grid, vertices
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType origin = mesh.getOrigin().x();
   const RealType spaceStep = mesh.getSpaceSteps().x();
 
   str << "POINTS " << mesh.getDimensions().x() + 1 << " float" << std::endl;
   for (int i = 0; i < mesh.getDimensions().x() + 1; i++)
   {
       str << origin + i * spaceStep << " 0 0" << std::endl;
   }
 
   str << std::endl << "CELLS " << mesh.getDimensions().x() + 1 << " " << ( mesh.getDimensions().x() + 1 ) * 2 << std::endl;
   for (int i = 0; i < mesh.getDimensions().x() + 1; i++)
   {
       str << "1 " << i << std::endl;
   }
 
   str << std::endl << "CELL_TYPES " << mesh.getDimensions().x() + 1 << std::endl;
   for (int i = 0; i < mesh.getDimensions().x() + 1; i++)
   {
       str << "1 " << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << mesh.getDimensions().x() + 1 << std::endl;
   str << "SCALARS VerticesFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < mesh.template getEntitiesCount< typename MeshType::Vertex >(); i++ )
   {
      typename MeshType::Vertex entity = mesh.template getEntity< typename MeshType::Vertex >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }
 
   return true;
}

/****
 * 2D grid, cells
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
   const MeshIndex entitiesCount = mesh.template getEntitiesCount< typename MeshType::Cell >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int j = 0; j < mesh.getDimensions().y() + 1; j++)
   {
        for (int i = 0; i < mesh.getDimensions().x() + 1; i++)
        {
             str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " 0" << std::endl;
        }
   }
 
   str << std::endl << "CELLS " << entitiesCount << " " << entitiesCount * 5 << std::endl;
   for (int j = 0; j < mesh.getDimensions().y(); j++)
   {
        for (int i = 0; i < mesh.getDimensions().x(); i++)
        {
            str << "4 " << j * ( mesh.getDimensions().x() + 1 ) + i << " " << j * ( mesh.getDimensions().x() + 1 )+ i + 1 <<
                   " " << (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " " << (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << std::endl;
        }
   }
 
   str << std::endl << "CELL_TYPES " << mesh.getDimensions().x() * mesh.getDimensions().y() << std::endl;
   for (int i = 0; i < mesh.getDimensions().x()*mesh.getDimensions().y(); i++)
   {
       str << "8 " << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS cellFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < entitiesCount; i++ )
   {
      typename MeshType::Cell entity = mesh.template getEntity< typename MeshType::Cell >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

/****
 * 2D grid, faces
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   typedef typename MeshType::template EntityType< 0 > Vertex;
   typedef typename MeshType::template EntityType< 1 > Face;
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
   const MeshIndex entitiesCount = mesh.template getEntitiesCount< typename MeshType::Face >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int j = 0; j < ( mesh.getDimensions().y() + 1); j++)
   {
        for (int i = 0; i < ( mesh.getDimensions().x() + 1 ); i++)
        {
             str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " 0" << std::endl;
        }
   }
 
   str << std::endl << "CELLS " << entitiesCount << " " << entitiesCount * 3 << std::endl;
   for (int j = 0; j < mesh.getDimensions().y(); j++)
   {
        for (int i = 0; i < ( mesh.getDimensions().x() + 1 ); i++)
        {
            str << "2 " << j * ( mesh.getDimensions().x() + 1 ) + i << " " << (j+1) * ( mesh.getDimensions().x() + 1 ) + i << std::endl;
        }
   }
 
   for (int j = 0; j < (mesh.getDimensions().y()+1); j++)
   {
        for (int i = 0; i < mesh.getDimensions().x(); i++)
        {
            str << "2 " << j * ( mesh.getDimensions().x() + 1 ) + i << " " <<j * ( mesh.getDimensions().x() + 1 ) + i + 1<< std::endl;
        }
   }
 
   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   for (int i = 0; i < entitiesCount; i++)
   {
       str << "3" << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS FaceslFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < entitiesCount; i++ )
   {
      typename MeshType::Face entity = mesh.template getEntity< typename MeshType::Face >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

/****
 * 2D grid, vertices
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   typedef typename MeshType::template EntityType< 0 > Vertex;
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int j = 0; j < ( mesh.getDimensions().y() + 1); j++)
   {
        for (int i = 0; i < ( mesh.getDimensions().x() + 1 ); i++)
        {
             str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " 0" << std::endl;
        }
   }
 
   str << std::endl << "CELLS " << verticesCount << " " << verticesCount * 2 << std::endl;
   for (int j = 0; j < ( mesh.getDimensions().y() + 1 ); j++)
   {
        for (int i = 0; i < ( mesh.getDimensions().x() + 1 ); i++)
        {
            str << "1 " << j * mesh.getDimensions().x() + i  << std::endl;
        }
   }
 
   str << std::endl << "CELL_TYPES " << verticesCount << std::endl;
   for (int i = 0; i < verticesCount; i++)
   {
       str << "1" << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << verticesCount << std::endl;
   str << "SCALARS VerticesFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < verticesCount; i++ )
   {
      typename MeshType::Vertex entity = mesh.template getEntity< typename MeshType::Vertex >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

/****
 * 3D grid, cells
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const RealType originZ = mesh.getOrigin().z();
   const RealType spaceStepZ = mesh.getSpaceSteps().z();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
   const MeshIndex entitiesCount = mesh.template getEntitiesCount< typename MeshType::Cell >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int k = 0; k <= mesh.getDimensions().y(); k++)
   {
       for (int j = 0; j <= mesh.getDimensions().y(); j++)
       {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                 str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " " <<
                        originZ + k * spaceStepZ << std::endl;
            }
       }
   }
 
   str << std::endl << "CELLS " << entitiesCount << " " <<
          entitiesCount * 9 << std::endl;
   for (int k = 0; k < mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j < mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i < mesh.getDimensions().x(); i++)
            {
                str << "8 " <<  k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << std::endl;
            }
        }
   }
 
   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   for (int i = 0; i < entitiesCount; i++)
   {
       str << "11" << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS cellFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < entitiesCount; i++ )
   {
      typename MeshType::Cell entity = mesh.template getEntity< typename MeshType::Cell >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

/****
 * 3D grid, faces
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const RealType originZ = mesh.getOrigin().z();
   const RealType spaceStepZ = mesh.getSpaceSteps().z();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
   const MeshIndex entitiesCount = mesh.template getEntitiesCount< typename MeshType::Face >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int k = 0; k <= mesh.getDimensions().y(); k++)
   {
       for (int j = 0; j <= mesh.getDimensions().y(); j++)
       {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                 str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " " <<
                        originZ + k * spaceStepZ << std::endl;
            }
       }
   }
 
   str << std::endl << "CELLS " << entitiesCount << " " << entitiesCount * 5 << std::endl;
   for (int k = 0; k < mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j < mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                str << "4 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << std::endl;
            }
        }
   }
 
   for (int k = 0; k < mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j <= mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i < mesh.getDimensions().x(); i++)
            {
                str << "4 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << std::endl;
            }
        }
   }
 
   for (int k = 0; k <= mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j < mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i < mesh.getDimensions().x(); i++)
            {
                str << "4 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1<< std::endl;
            }
        }
   }
 
   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   for (int i = 0; i < entitiesCount; i++)
   {
       str << "8" << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS facesFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < entitiesCount; i++ )
   {
      typename MeshType::Face entity = mesh.template getEntity< typename MeshType::Face >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

/****
 * 3D grid, edges
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const RealType originZ = mesh.getOrigin().z();
   const RealType spaceStepZ = mesh.getSpaceSteps().z();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
   const MeshIndex entitiesCount = mesh.template getEntitiesCount< typename MeshType::Edge >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int k = 0; k <= mesh.getDimensions().y(); k++)
   {
       for (int j = 0; j <= mesh.getDimensions().y(); j++)
       {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                 str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " " <<
                        originZ + k * spaceStepZ << std::endl;
            }
       }
   }
 
   str << std::endl << "CELLS " << entitiesCount << " " << entitiesCount * 3 << std::endl;
   for (int k = 0; k <= mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j <= mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i < mesh.getDimensions().x(); i++)
            {
                str << "3 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << std::endl;
            }
        }
   }
 
   for (int k = 0; k <= mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j < mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                str << "3 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << std::endl;
            }
        }
   }
 
   for (int k = 0; k < mesh.getDimensions().z(); k++)
   {
        for (int j = 0; j <= mesh.getDimensions().y(); j++)
        {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                str << "3 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                    << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << std::endl;
            }
        }
   }
 
   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   for (int i = 0; i < entitiesCount; i++)
   {
       str << "3" << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
   str << "SCALARS edgesFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < entitiesCount; i++ )
   {
      typename MeshType::Edge entity = mesh.template getEntity< typename MeshType::Edge >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

/****
 * 3D grid, vertices
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
void
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > >::
writeHeader( const MeshFunctionType& function,
             std::ostream& str )
{
    const MeshType& mesh = function.getMesh();
    const typename MeshType::PointType& origin = mesh.getOrigin();
    const typename MeshType::PointType& proportions = mesh.getProportions();
    str << "# vtk DataFile Version 2.0" << std::endl;
    str << "TNL DATA" << std::endl;
    str << "ASCII" << std::endl;
    str << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str,
       const double& scale )
{
   writeHeader(function, str);
 
   const MeshType& mesh = function.getMesh();
   const RealType originX = mesh.getOrigin().x();
   const RealType spaceStepX = mesh.getSpaceSteps().x();
   const RealType originY = mesh.getOrigin().y();
   const RealType spaceStepY = mesh.getSpaceSteps().y();
   const RealType originZ = mesh.getOrigin().z();
   const RealType spaceStepZ = mesh.getSpaceSteps().z();
   const MeshIndex verticesCount = mesh.template getEntitiesCount< typename MeshType::Vertex >();
 
   str << "POINTS " << verticesCount << " float" << std::endl;
   for (int k = 0; k <= mesh.getDimensions().y(); k++)
   {
       for (int j = 0; j <= mesh.getDimensions().y(); j++)
       {
            for (int i = 0; i <= mesh.getDimensions().x(); i++)
            {
                 str << originX + i * spaceStepX << " " << originY + j * spaceStepY << " " <<
                        originZ + k * spaceStepZ << std::endl;
            }
       }
   }
 
   str << std::endl << "CELLS " << verticesCount << " " << verticesCount * 2 << std::endl;
   for (int k = 0; k < ( mesh.getDimensions().z() + 1 ); k++)
   {
        for (int j = 0; j < ( mesh.getDimensions().y() + 1 ); j++)
        {
            for (int i = 0; i < ( mesh.getDimensions().x() + 1 ); i++)
            {
                str << "1 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i  << std::endl;
            }
        }
   }
 
   str << std::endl << "CELL_TYPES " << verticesCount << std::endl;
   for (int i = 0; i < verticesCount; i++)
   {
       str << "1" << std::endl;
   }
 
   str << std::endl << "CELL_DATA " << verticesCount << std::endl;
   str << "SCALARS verticesFunctionValues float 1" << std::endl;
   str << "LOOKUP_TABLE default" << std::endl;

   for( MeshIndex i = 0; i < verticesCount; i++ )
   {
      typename MeshType::Vertex entity = mesh.template getEntity< typename MeshType::Vertex >( i );
      entity.refresh();
      str << scale * function.getData().getElement( entity.getIndex() ) << std::endl;
   }

   return true;
}

} // namespace Functions
} // namespace TNL

