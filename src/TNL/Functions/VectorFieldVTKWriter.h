/***************************************************************************
                          VectorFieldVTKWriter.h  -  description
                             -------------------
    begin                : Jan 10, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Functions {

template< int, typename > class VectorField;

template< typename VectorField >
class VectorFieldVTKWriter
{
   public:

      static bool write( const VectorField& vectorField,
                         std::ostream& str,
                         const double& scale );
      
      static void writeHeader( const VectorField& vectorField,
                               std::ostream& str ){}
      
};

/***
 * 1D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > > >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

/***
 * 1D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > > >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};


/***
 * 2D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 2, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );

      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

/***
 * 2D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

/***
 * 2D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};


/***
 * 3D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 3, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

/***
 * 3D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 2, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

/***
 * 3D grids edges
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

/***
 * 3D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldVTKWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > > VectorFieldType;
      using VectorType = typename VectorFieldType::VectorType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str,
                         const double& scale  );
      
      static void writeHeader( const VectorFieldType& vectorField,
                               std::ostream& str );
      
};

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/VectorFieldVTKWriter_impl.h>
