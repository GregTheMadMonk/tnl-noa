#include <iostream>
#include <TNL/Timer.h>
#include <unistd.h>

using namespace TNL;
using namespace std;

int main()
{
    unsigned int microseconds = 0.5e6;
    Timer time;
    time.start();
    usleep(microseconds);
    time.stop();
    cout << "Elapsed real time: " << time.getRealTime() << endl;
    cout << "Elapsed CPU time: " << time.getCPUTime() << endl;
    cout << "Elapsed CPU cycles: " << time.getCPUCycles() << endl;
    time.reset();
    cout << "Real time after reset:" << time.getRealTime() << endl;
    cout << "CPU time after reset: " << time.getCPUTime() << endl;
    cout << "CPU cycles after reset: " << time.getCPUCycles() << endl;
}

