/***************************************************************************
                          DummyMesh.h  -  description
                             -------------------
    begin                : Apr 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/File.h>
#include <TNL/Devices/Host.h>

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

   void save( File& file ) const {}

   void load( File& file ) {}

   void save( const String& fileName ) const {}

   void load( const String& fileName ) {}

   bool writeMesh( const String& fileName,
                   const String& format ) const { return true; }


   template< typename MeshFunction >
   bool write( const MeshFunction& function,
                const String& fileName,
                const String& format ) const { return true; }
};

} // namespace Meshes
} // namespace TNL
