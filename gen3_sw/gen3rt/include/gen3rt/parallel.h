// Host threads of the runtime: G3RT_HOST_THREADS (1..4, default min(4, cores)) persistent workers, one caller at a
// time (src/llm.cpp).
#pragma once
#include <functional>

namespace g3rt {

// f(0) .. f(n - 1) on up to the host threads, this one included; f must not touch the device or nest a call
void parallel_for(int n, const std::function<void(int)>& f);

}  // namespace g3rt
