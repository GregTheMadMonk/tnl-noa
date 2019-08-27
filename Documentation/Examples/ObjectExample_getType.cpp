#include <iostream>
#include <TNL/param-types.h>
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

      using HostType = MyArray< Value, Devices::Host >;
      
      static String getType()
      {
         return "MyArray< " + TNL::getType< Value >() + ", " + TNL::getType< Device >() + " >";
      }

      String getTypeVirtual() const
      {
         return getType();
      }

      static String getSerializationType()
      {
         return HostType::getType();
      }

      String getSerializationTypeVirtual() const
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
   cout << "HostArray type is                  " << HostArray::getType() << endl;
   cout << "hostArrayPtr type is               " << hostArrayPtr->getTypeVirtual() << endl;

   cout << "CudaArray type is                  " << CudaArray::getType() << endl;
   cout << "cudaArrayPtr type is               " << cudaArrayPtr->getTypeVirtual() << endl;

   // Object serialization types
   cout << "HostArray serialization type is    " << HostArray::getSerializationType() << endl;
   cout << "hostArrayPtr serialization type is " << hostArrayPtr->getSerializationTypeVirtual() << endl;

   cout << "CudaArray serialization type is    " << CudaArray::getSerializationType() << endl;
   cout << "cudaArrayPtr serialization type is " << cudaArrayPtr->getSerializationTypeVirtual() << endl;
}

