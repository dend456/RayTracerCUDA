#include "FEClock.h"

F32 FEClock::cyclesPerSecond = -1;

FEClock::FEClock(F32 startTime) :
	timeCycles(secondsToCycles(startTime)),
	timeScale(1.0f),
	paused(false),
	cyclesLastUpdate(0)
{
}

FEClock::~FEClock(void) {}