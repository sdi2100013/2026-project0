CXX := g++
CPPFLAGS := -Iincludes
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS :=
LDLIBS :=
TARGET := search
BUILD_DIR := build
SOURCES := main/search.cpp $(wildcard src/*.cpp)
OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)
.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
-include $(DEPS)
