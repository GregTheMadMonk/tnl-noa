/***************************************************************************
                          DummyProblem.h  -  description
                             -------------------
    begin                : Jul 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Problems/CommonData.h>

namespace TNL {
namespace Solvers {   

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class DummyProblem
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Containers::Vector< Real, Device, Index > DofVectorType;
      typedef Meshes::Grid< 1, Real, Device, Index > MeshType;
      using CommonDataType = Problems::CommonData;
      using CommonDataPointer = SharedPointer< CommonDataType, Device >;
      using CommunicatorType = Communicators::NoDistrCommunicator;
      
      static constexpr bool isTimeDependent(){ return true; };      
};

class DummySolver
{};

} // namespace Solvers
} // namespace TNL
