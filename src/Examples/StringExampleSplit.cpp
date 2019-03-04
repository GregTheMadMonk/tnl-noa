#include <iostream>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
   String dates("3/4/2005;8/7/2011;11/12/2019");
   vector< String > list = dates.split(';');
   cout << "list_dates = " << list[0] << ", " << list[1] << ", " << list[2] << endl;

   String cars("Subaru,Mazda,,Skoda," );
   vector< String > list3 = cars.split(',', String::SplitSkip::SkipEmpty );
   cout << "split with String::SkipEmpty = " << list3[0] << ", " << list3[1] << ", " << list3[2] << endl;
   std::vector<String> list5 = cars.split(',');
   cout << "split without  String::SkipEmpty = " << list5[0] << ", " << list5[1] << ", " << list5[2] << ", " << list5[3] << endl;
}
