#include <iostream>
#include <TNL/TypeInfo.h>
#include <TNL/Object.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

using namespace TNL;
using namespace std;

template< typename Value,
          typename Device >
class MyArray : public Object
{
   public:

      static String getSerializationType()
      {
         return "MyArray< " + TNL::getType< Value >() + ", " + getType< Devices::Host >() + " >";
      }

      virtual String getSerializationTypeVirtual() const override
      {
         return getSerializationType();
      }
};

int main()
{
   using HostArray = MyArray< int, Devices::Host >;
   using CudaArray = MyArray< int, Devices::Cuda >;

   HostArray hostArray;
   CudaArray cudaArray;
   Object* hostArrayPtr = &hostArray;
   Object* cudaArrayPtr = &cudaArray;

   // Object types
   cout << "HostArray type is                  " << getType< HostArray >() << endl;
   cout << "hostArrayPtr type is               " << getType( *hostArrayPtr ) << endl;

   cout << "CudaArray type is                  " << getType< CudaArray >() << endl;
   cout << "cudaArrayPtr type is               " << getType( *cudaArrayPtr ) << endl;

   // Object serialization types
   cout << "HostArray serialization type is    " << HostArray::getSerializationType() << endl;
   cout << "hostArrayPtr serialization type is " << hostArrayPtr->getSerializationTypeVirtual() << endl;

   cout << "CudaArray serialization type is    " << CudaArray::getSerializationType() << endl;
   cout << "cudaArrayPtr serialization type is " << cudaArrayPtr->getSerializationTypeVirtual() << endl;
}
