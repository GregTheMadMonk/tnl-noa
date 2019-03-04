/***************************************************************************
                          NotTNLFile.h  -  description
                             -------------------
    begin                : Mar 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <string>
#include <stdexcept>

namespace TNL {
namespace Exceptions {

class NotTNLFile
   : public std::runtime_error
{
public:
   NotTNLFile()
   : std::runtime_error( "Wring magic number found in a binary file. It is not TNL compatible file." )
   {}
};

} // namespace Exceptions
} // namespace TNL
