**Note** The presets feature is experimental and work in progress

# How to list all available CMake-Presets?
```
cmake --list-presets
```
# How to configure using CMake-Presets?
```
cmake --preset <preset-config-from-list>
eg.,
    cmake --preset linux-make-st-lp64-static
```
# How to build the configured preset?
## Using CMake
```
cmake --build build/linux-make-st-lp64-static --target install --verbose -j 128
```
## Using Presets
```
cmake --build --preset <preset-config-from-list>
eg.,
    cmake --build --preset linux-make-st-lp64-static
```
# How to run unit-tests for the configured preset?
## Using CMake
```
ctest --test-dir <build-directory> -j 128
where,
    <build-directory> = build/linux-make-st-lp64-static/
```
## Using Presets
```
ctest --preset <preset-config-from-list>
eg.,
    ctest --preset linux-make-st-lp64-static
```
# How to run a workflow preset that runs all 3 steps such as configure, build and test?
```
cmake --workflow --preset <preset-config-from-list>
eg.,
    cmake --workflow --preset linux-make-st-lp64-static
```

**Note**
1. CMake variable "USE_EXTERNAL_OMP_LIB" is only used in windows for non-MSVC builds and is disabled by default. Thus in default case, openmp targets are fetched using standard find_package(OpenMP)
2. To use an external or a vendor-provided OpenMP library using windows presets, define the following environment variables
```
    set "Enable_External_OpenMP_lib=ON"
    set "External_OpenMP_lib=C:/Path/to/external/OpenMP/library/libxxx.lib"
```
3. If Boost is installed in a local directory or expect CMake configuration to choose a local Boost install instead of system install, then add the following two CMake variables during configuration:
"-DCMAKE_PREFIX_PATH=<path to local boost installation> -DBoost_NO_SYSTEM_PATHS=TRUE". Note that "workflow preset" seem to have an issue finding non-system installation of boost at persent.
