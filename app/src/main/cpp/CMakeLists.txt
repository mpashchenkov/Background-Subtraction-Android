# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.22.1)
set (CMAKE_CXX_STANDARD 17)
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
#set(CMAKE_EXE_LINKER_FLAGS "-Wl,--allow-shlib-undefined")


find_package( OpenCV REQUIRED )

find_path(TENSORFLOW_SRC_DIR "tensorflow/lite/interpreter.h")
if (TENSORFLOW_SRC_DIR)
    set(LIB_NAME "libtensorflowlite.so")
    set(GPU_LIB_NAME "libtensorflowlite_gpu_delegate.so")
    #find_path(TFLITE_BUILD_DIR ${LIB_NAME})

    if(TFLITE_BUILD_DIR)
        get_filename_component(FB_DIR "${TFLITE_BUILD_DIR}/../.." ABSOLUTE)
        set(FB_INCLUDE "${FB_DIR}/external/flatbuffers/src/_virtual_includes/flatbuffers")
        set(TFLITE_INCLUDE_DIR "${TENSORFLOW_SRC_DIR};${FB_INCLUDE}")
        find_library(TFLITE_LIB_PATH "${LIB_NAME}"
                ${TFLITE_BUILD_DIR}
                CMAKE_FIND_ROOT_PATH_BOTH)
        message(${TFLITE_BUILD_DIR})
        find_library(TFLITE_GPU_LIB_PATH libtensorflowlite_gpu_delegate.so
                ${TFLITE_BUILD_DIR}/delegates/gpu/
                CMAKE_FIND_ROOT_PATH_BOTH)
        set(TFLITE_LIBRARY "tensorflow-lite")
        add_library(${TFLITE_LIBRARY} SHARED IMPORTED)
        set_target_properties(${TFLITE_LIBRARY} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_INCLUDE_DIR}"
                IMPORTED_LOCATION ${TFLITE_LIB_PATH}
                IMPORTED_IMPLIB ${TFLITE_LIB_PATH})

        #target_link_libraries(ex_tf_seg tensorflow-lite ${OpenCV_LIBRARIES} ${TFLITE_GPU_LIB_PATH})
    endif()
endif()

project("background_subtraction_sample")
add_library( # Sets the name of the library.
        background_subtraction_sample

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp )
target_link_libraries(background_subtraction_sample ${OpenCV_LIBRARIES} ${TFLITE_LIBRARY} ${TFLITE_GPU_LIB_PATH})