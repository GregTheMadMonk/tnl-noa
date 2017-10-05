/***************************************************************************
                          DummyMesh.h  -  description
                             -------------------
    begin                : Apr 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class DummyMesh
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef DummyMesh< Real, Device, Index > ThisType;
 
   constexpr static int getMeshDimension() { return 1; }
 
   const Real& getParametricStep(){ return 0.0; }
 
   String getSerializationType() const { return String( "DummyMesh" ); }

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const { return 0.0; }

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const { return 0.0; }

   bool save( File& file ) const { return true; }

   //! Method for restoring the object from a file
   bool load( File& file ) { return true; }

   bool save( const String& fileName ) const { return true; }

   bool load( const String& fileName ) { return true; }

   bool writeMesh( const String& fileName,
                   const String& format ) const { return true; }


   template< typename MeshFunction >
   bool write( const MeshFunction& function,
                const String& fileName,
                const String& format ) const { return true; }
};

} // namespace Meshes
} // namespace TNL
