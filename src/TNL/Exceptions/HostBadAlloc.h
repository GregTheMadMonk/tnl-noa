/***************************************************************************
                          HostBadAlloc.h  -  description
                             -------------------
    begin                : Apr 17, 2019
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Lukas Cejka

#pragma once

#include <new>

namespace TNL {
namespace Exceptions {

struct HostBadAlloc
   : public std::bad_alloc
{
    HostBadAlloc()
    {
        // Assert that there is enough space to store the values.
//        TNL_ASSERT( Devices::SystemInfo::getFreeMemory() > Matrices::Matrix::getNumberOfMatrixElements() * sizeof( Matrices::Matrix::RealType ), );
        std::cerr << "terminate called after throwing an instance of 'TNL::Exceptions::HostBadAlloc'\n  what():  " << what() << std::endl;
        std::exit(1);
    }
    
   const char* what() const throw()
   {
      return "Failed to allocate memory on the Host device: "
             "most likely there is not enough space in the host memory.";
   }
};

} // namespace Exceptions
} // namespace TNL
