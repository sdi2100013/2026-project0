CXX := g++
CPPFLAGS := -Iinclude
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS :=
LDLIBS :=
TARGET := search
TEST_TARGET := build/test/test
#OpenCV 4.2.0
#OPENCV_CXXFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
#OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)
#OpenCV 4.10.0
OPENCV_CXXFLAGS := -I/opt/opencv-4.10/include/opencv4
OPENCV_LIBS := -L/opt/opencv-4.10/lib -lopencv_features2d -lopencv_imgcodecs -lopencv_imgproc -lopencv_flann -lopencv_core -Wl,-rpath,/opt/opencv-4.10/lib
BUILD_DIR := build
SOURCES := main/search.cpp $(wildcard src/*.cpp)
OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)
.PHONY: all test clean
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) $(OPENCV_LIBS) -o $@
test: $(TEST_TARGET)
$(TEST_TARGET): test/test.cpp
	@pkg-config --exists opencv4 || (echo "OpenCV development files are required. Install libopencv-dev."; exit 1)
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCV_CXXFLAGS) $< $(OPENCV_LIBS) -o $@
$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCV_CXXFLAGS) -MMD -MP -c $< -o $@
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
-include $(DEPS)