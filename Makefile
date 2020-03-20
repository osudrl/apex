# Build for linux by default

# Compilation settings
INC     := -I$(HOME)/.mujoco/mujoco200_linux/include
CFLAGS  := -Wall -Wextra -O3 #-march=sandybridge -flto
# LDFLAGS := -L
LIBS    := -Wl, -lGL, -lglew, $(HOME)/.mujoco/mujoco200_linux/include -L$(HOME)/.mujoco/mujoco200_linux/bin

CC      := g++
LDFLAGS := -Wl,-rpath,'$$ORIGIN'
COMMON=-O2 -I$(HOME)/.mujoco/mujoco200_linux/include -L$(HOME)/.mujoco/mujoco200_linux/bin -std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN'


# Default target
all: $(TESTOUT) $(SIMOUT) $(CTRLOUT)

# Normal targets
clean:
	rm -f vis

vis: 
	# $(INC)
	# $(CC) vis_orig_recon.cpp $(INC) $(CFLAGS) $(LDFLAGS) $(LIBS) -o vis
	g++ $(COMMON) vis_orig_recon.cpp      -lmujoco200 -lGL -lglew $(HOME)/.mujoco/mujoco200_linux/bin/libglfw.so.3 -o vis


# Virtual targets
.PHONY: all clean
