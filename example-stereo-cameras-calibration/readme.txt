# Checkerboard:
# - landscape orientation
# - 10 x 7 squares
# - 9 x 6 inner corners (the values OpenCV uses)
# - 25 mm x 25 mm per square

# Build (sudo is not required):
$ cmake -S . -B build
$ cmake --build build -j

# Run from the build directory. The parameters below are now also defaults:
$ cd build
$ ./stereo_calibration -w=9 -h=6 -s=25 ../images/stereo_calib.xml

# Results:
# See the intrinsic/extrinsic results in the "cal_results" folder and copy
# them into your own project folder.

# Important: calib.io_checker_265x370_10x7_35.pdf is the old 35 mm target.
# Do not use it for this 25 mm calibration configuration.



