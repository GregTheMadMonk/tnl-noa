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
 
   constexpr static int getMeshDimension() { return 1; }
 
   void save( File& file ) const {}

   void load( File& file ) {}

   void save( const String& fileName ) const {}

   void load( const String& fileName ) {}
};

} // namespace Meshes
} // namespace TNL
