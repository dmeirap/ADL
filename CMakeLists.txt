cmake_minimum_required(VERSION 3.16.3)
project(qr_src)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Python3 COMPONENTS Intrepreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} include)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBar REQUIRED IMPORTED_TARGET zbar)

add_executable(qr_src qr_code.cpp)
target_link_libraries( qr_src ${OpenCV_LIBS} ${Python3_LIBRARIES})

target_link_libraries(qr_src PkgConfig::ZBar)
