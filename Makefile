CXX := g++
CPPFLAGS := -Iincludes
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS :=
LDLIBS :=
TARGET := search
TEST_TARGET := build/test/test
OPENCV_PREFIX ?= /opt/opencv-4.10
OPENCV_PKG_CONFIG := $(shell pkg-config --exists opencv4 >/dev/null 2>&1 && echo yes)
ifeq ($(OPENCV_PKG_CONFIG),yes)
OPENCV_CXXFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)
else ifneq ($(wildcard $(OPENCV_PREFIX)/include/opencv4/opencv2/opencv.hpp),)
OPENCV_CXXFLAGS := -I$(OPENCV_PREFIX)/include/opencv4
OPENCV_LIBS := -L$(OPENCV_PREFIX)/lib -lopencv_features2d -lopencv_imgcodecs -lopencv_imgproc -lopencv_flann -lopencv_core -Wl,-rpath,$(OPENCV_PREFIX)/lib
else
$(error OpenCV 4 was not found. Install libopencv-dev, or set OPENCV_PREFIX to its installation directory)
endif
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
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCV_CXXFLAGS) $< $(OPENCV_LIBS) -o $@
$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCV_CXXFLAGS) -MMD -MP -c $< -o $@
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
-include $(DEPS)

.PHONY: test-vocabulary
test-vocabulary: $(BUILD_DIR)/test/vocabulary_test
	./$(BUILD_DIR)/test/vocabulary_test

$(BUILD_DIR)/test/vocabulary_test: test/vocabulary_test.cpp src/vocabulary.cpp src/dataset.cpp includes/vocabulary.hpp includes/dataset.hpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCV_CXXFLAGS) test/vocabulary_test.cpp src/vocabulary.cpp src/dataset.cpp $(LDFLAGS) $(LDLIBS) $(OPENCV_LIBS) -o $@
