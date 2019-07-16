/***************************************************************************
                          StaticFor.h  -  description
                             -------------------
    begin                : Jul 16, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Cuda.h>

namespace TNL {

template< int Begin, int End >
struct StaticFor
{
    template< typename Function, typename... Args >
    __cuda_callable__
    static void exec( const Function& f, Args... args )
    {
        static_assert( Begin < End, "Wrong index interval for StaticFor. Being must be lower than end." );
        f( Begin, args... );
        StaticFor< Begin + 1, End >::exec( f, args... );
    };
};

template< int End >
struct StaticFor< End, End >
{
    template< typename Function, typename... Args >
    __cuda_callable__
    static void exec( const Function& f, Args... args ){};
};

} //namespace TNL
