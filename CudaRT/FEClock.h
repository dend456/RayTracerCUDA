#ifndef FE_CLOCK_H
#define FE_CLOCK_H

#ifdef _WIN32
	#include <Windows.h>
#endif

	typedef unsigned long long U64;
	typedef long long I64;
	typedef float F32;
	typedef unsigned char byte;

	class FEClock
	{
	private:
		U64 timeCycles;
		F32 timeScale;
		bool paused;
		U64 cyclesLastUpdate;

	public:
		void setPaused(bool isPaused) { paused = isPaused; }
		bool getPaused() { return paused; }
		void setTimeScale(F32 scale) { timeScale = scale; }
		F32 getTimeScale() { return timeScale; }

		static F32 cyclesPerSecond;

		static U64 secondsToCycles(F32 seconds)
		{
			return (U64)(seconds * cyclesPerSecond);
		}

		static F32 cyclesToSeconds(U64 cycles)
		{
			return (F32) cycles / cyclesPerSecond;
		}

		static void init()
		{
			#ifdef _WIN32
				LARGE_INTEGER li;
				QueryPerformanceFrequency(&li);
				cyclesPerSecond = (F32) li.QuadPart;
			#else
				cyclesPerSecond = -1;
			#endif
		}

		void singleStep()
		{
			if(paused)
			{
				timeCycles += secondsToCycles((1.0f / 60) * timeScale);
			}
		}

		F32 update()
		{
			if(!paused)
			{
				#ifdef _WIN32
					LARGE_INTEGER li;
					QueryPerformanceCounter(&li);
					U64 cycles = li.QuadPart;
					if(cyclesLastUpdate == 0)
					{
						cyclesLastUpdate = cycles;
						return 0.0f;
					}
					U64 diff = (U64)(cycles - cyclesLastUpdate);
					cyclesLastUpdate = cycles;
					timeCycles += (U64)(diff * timeScale);

					return cyclesToSeconds(diff);
				#else

				#endif
			}
			return 0.0f;
		}

		F32 update(F32 dt)
		{
			if(!paused)
			{
				U64 cycles = secondsToCycles(dt * timeScale);
				timeCycles += cycles;
				return FEClock::cyclesToSeconds(cycles);
			}
			return 0.0f;
		}

		U64 getTimeCycles() const
		{
			return timeCycles;
		}

		F32 calcDeltaSeconds(const FEClock& o)
		{
			U64 dt = timeCycles - o.timeCycles;
			return cyclesToSeconds(dt);
		}

		FEClock(F32 startTime = 0.0f);

		~FEClock(void);
	};

#endif

