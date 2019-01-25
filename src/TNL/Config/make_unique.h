#pragma once

// std::make_unique does not exist until C++14
// https://stackoverflow.com/a/9657991
#if __cplusplus < 201402L
#include <memory>

namespace std {
   template<typename T, typename ...Args>
   std::unique_ptr<T> make_unique( Args&& ...args )
   {
      return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
   }
}
#endif
