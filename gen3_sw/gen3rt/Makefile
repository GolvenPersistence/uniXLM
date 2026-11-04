# gen3rt: the runtime of the Gen3 Node v3 (command encoders, layouts, devices, the SmolVLM2 programs) and its tools.
#
#   make                       # build/libgen3rt.a (with the emulator's sources), build/gen3rt_llm
#   make LLAMA=../llama.cpp LLAMA_BUILD=../llama.cpp-build-x86
#
# The numerics of the emulator need -ffp-contract=off (always added).

ifeq ($(OS),Windows_NT)
EXE := .exe
LINK_FLAGS := -static
SYS_LIBS := -lws2_32 -lmswsock
else
EXE :=
LINK_FLAGS :=
SYS_LIBS := -ldl
endif

OPT ?= -O2
CXXFLAGS ?=
EMU := ../emu
LLAMA ?= ../llama.cpp
LLAMA_BUILD ?= ../llama.cpp-build-x86
BUILD ?= build
REQUIRED := -std=c++17 -ffp-contract=off -fno-fast-math -pthread -Iinclude -I$(EMU)/include -I$(EMU)/src
ALL_FLAGS = $(OPT) $(REQUIRED) -Wall -Wextra $(CXXFLAGS)

EMU_SRC := mxfp8.cpp gemm.cpp epilogue.cpp fold.cpp attention.cpp node.cpp program.cpp
RT_SRC := cmd.cpp layout.cpp device_emu.cpp llm.cpp weights.cpp vit.cpp profile.cpp
ifdef BOARD
RT_SRC += device_board.cpp
SYS_LIBS += -lxrt_coreutil -lcrypto -fopenmp
CXXFLAGS += -DG3RT_BOARD -I/usr/include/xrt
endif
OBJ := $(patsubst %.cpp,$(BUILD)/emu_%.o,$(EMU_SRC)) $(patsubst %.cpp,$(BUILD)/%.o,$(RT_SRC))
HDRS := $(wildcard include/gen3rt/*.h) $(wildcard $(EMU)/include/*.h)

all: $(BUILD)/libgen3rt.a $(BUILD)/gen3rt_llm$(EXE)

$(BUILD)/emu_%.o: $(EMU)/src/%.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(ALL_FLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(ALL_FLAGS) -c $< -o $@

$(BUILD)/libgen3rt.a: $(OBJ)
	$(AR) rcs $@ $^

LLAMA_INC := -I$(LLAMA)/include -I$(LLAMA)/ggml/include
ifeq ($(OS),Windows_NT)
LLAMA_LIBS ?= $(LLAMA_BUILD)/src/libllama.a $(LLAMA_BUILD)/ggml/src/ggml.a $(LLAMA_BUILD)/ggml/src/ggml-cpu.a \
              $(LLAMA_BUILD)/ggml/src/ggml-base.a
else
LLAMA_LIBS ?= $(LLAMA_BUILD)/src/libllama.a $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-cpu.a $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-gen3/libggml-gen3.a) \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a
endif

$(BUILD)/llm_run.o: tools/llm_run.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(ALL_FLAGS) $(LLAMA_INC) -c $< -o $@

$(BUILD)/gen3rt_llm$(EXE): $(BUILD)/llm_run.o $(BUILD)/libgen3rt.a
	$(CXX) $(OPT) -pthread $(LINK_FLAGS) -o $@ $^ $(LLAMA_LIBS) $(SYS_LIBS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean

# Differential command encoding and weight-cache integrity tests (no board access).
REF_OPS := bind gemm attn move_out move_in fence wait unbind
$(BUILD)/cmd_reference.o: tests/cmd_reference.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(ALL_FLAGS) $(foreach op,$(REF_OPS),-Dcmd_$(op)=reference_$(op)) -c $< -o $@

$(BUILD)/cmd_test$(EXE): tests/cmd_test.cpp $(BUILD)/cmd_reference.o $(BUILD)/cmd.o
	$(CXX) $(ALL_FLAGS) $(LINK_FLAGS) $^ -o $@

$(BUILD)/weights_test$(EXE): tests/weights_test.cpp $(BUILD)/libgen3rt.a
	$(CXX) $(ALL_FLAGS) $(LINK_FLAGS) $^ $(SYS_LIBS) -o $@

check: $(BUILD)/cmd_test$(EXE) $(BUILD)/weights_test$(EXE)
	$(BUILD)/cmd_test$(EXE)
	$(BUILD)/weights_test$(EXE) $(BUILD)/weights_test_data

.PHONY: check
