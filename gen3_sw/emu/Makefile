# Gen3 Node v3 emulator: library, CLI and kernel checker.
#
#   make                          # build/libg3emu.a, build/g3emu, build/g3emu_kcheck
#   make CXX=clang++              # any C++17 compiler (g++ 11+ on the board, llvm-mingw on Windows)
#   make OPT="-O3 -march=native"  # optimisation / target flags
#   make CXXFLAGS=...             # extra flags; the required ones below are always added
#
# The numerics need IEEE binary32 without FMA contraction: REQUIRED_FLAGS carries -ffp-contract=off on purpose
# (src/fp_strict.h repeats it as a pragma).

ifeq ($(OS),Windows_NT)
EXE := .exe
LINK_FLAGS := -static           # self-contained executables (no libc++ / libunwind DLLs of the toolchain)
else
EXE :=
LINK_FLAGS :=
endif

OPT ?= -O2
CXXFLAGS ?=
REQUIRED_FLAGS := -std=c++17 -ffp-contract=off -fno-fast-math -pthread -Iinclude
WARN := -Wall -Wextra
AR ?= ar
BUILD ?= build

ALL_FLAGS = $(OPT) $(REQUIRED_FLAGS) $(WARN) $(CXXFLAGS)
LIB_SRC := src/mxfp8.cpp src/gemm.cpp src/epilogue.cpp src/fold.cpp src/attention.cpp src/node.cpp src/program.cpp
LIB_OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRC))
HDRS := include/g3emu.h include/g3emu_kernels.h include/g3emu_layout.h src/tables.h src/bits.h src/fp_strict.h

all: $(BUILD)/libg3emu.a $(BUILD)/g3emu$(EXE) $(BUILD)/g3emu_kcheck$(EXE)

$(BUILD)/%.o: src/%.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(ALL_FLAGS) -c $< -o $@

$(BUILD)/libg3emu.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

$(BUILD)/g3emu$(EXE): $(BUILD)/g3emu_main.o $(BUILD)/libg3emu.a
	$(CXX) $(ALL_FLAGS) $(LINK_FLAGS) -o $@ $^

$(BUILD)/g3emu_kcheck$(EXE): $(BUILD)/kcheck.o $(BUILD)/libg3emu.a
	$(CXX) $(ALL_FLAGS) $(LINK_FLAGS) -o $@ $^

clean:
	rm -rf $(BUILD)

.PHONY: all clean
