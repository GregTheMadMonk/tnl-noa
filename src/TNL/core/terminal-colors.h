/***************************************************************************
                          terminal-colors.h  -  description
                             -------------------
    begin                : Feb 7, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

const String red( "\033[0;31m" );
const String green( "\033[1;32m" );
const String yellow( "\033[1;33m" );
const String cyan( "\033[0;36m" );
const String magenta( "\033[0;35m" );
const String bold();
const String reset( "\033[0m" );

} // namespace TNL

