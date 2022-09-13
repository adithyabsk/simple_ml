# Final Project

Implementation of OpenCL for the needle framework

## Installation

Python: 3.6.13

### macOS

```bash
brew install gcc@7 opencl-headers
sudo curl -o /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/OpenCL.framework/Headers/cl.hpp https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/main/include/CL/opencl.hpp
pip install -r requirements.txt
```

## Set up pre-commit

pre-commit runs some tooling before files are committed.

## Developers

### OpenCL

OpenCL C++ headers do not ship with macOS by default. This project is set up to
run using a legacy version of the OpenCL API (1.1). It is running using
compatibility mode to for example make kernels (`cl::compatibility::make_kernel`).

## TODOs

- [ ] Check if the project still passes all tests on Google Colab
  - Because of the CMakeLists.txt changes
- [ ] Update Makefile to only add brew specific paths if on macOS
- [ ] Find a way to only pass the DCMAKE flags in the Makefile when on MacOS
  - Add instructions for MacOS
    - Notes on finding pyenv using cmake (Python3_FIND_VIRTUALENV)
    - OpenCL on MacOS (does not ship with C++ headers)
      - https://stackoverflow.com/a/62310665/3262054
- [ ] Two main bugs right now
  - cannot find apps folder
  - some Nonetype array error (I've seen this before, I just need to break out
  the debugger)
- [ ] Add fixture to tests that downloads PTB and CIFAR to a data folder to run
in unit tests
