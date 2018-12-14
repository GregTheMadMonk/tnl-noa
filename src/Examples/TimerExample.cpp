#include <iostream>
#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <unistd.h>

using namespace TNL;
using namespace std;
       
int main()
{
    unsigned int microseconds = 0.5;
    Timer time;
    time.start();
    usleep(microseconds);
    time.stop();
    cout << "before reset:" << time.getRealTime() << endl;
    time.reset();
    cout << "after reset:" << time.getRealTime() << endl;
    // writeLog example
    Logger log1(50,cout);
    time.writeLog( log1, 0 );
}

