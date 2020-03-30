/***************************************************************************
                          AnyDevice.h  -  description
                             -------------------
    begin                : Mar 17, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>

namespace TNL {
namespace Devices {

class AnyDevice
{
};

template< typename Device >
struct PickDevice
{
   using DeviceType = Device;
};

template<>
struct PickDevice< Devices::AnyDevice >
{
   using DeviceType = Devices::Host;
};

} // namespace Devices
} // namespace TNL
