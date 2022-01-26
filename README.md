# Final Project

Implementation of OpenCL for the needle framework

## Installation

- TODO

```bash
pip install -r requirements.txt
```

## Set up pre-commit

pre-commit runs some tooling before files are committed.


## TODOs

- [ ] Check if the project still passes all tests on Google Colab
  - Because of the CMakeLists.txt changes
- [ ] Update Makefile to only add brew specific paths if on macOS
- [ ] Find a way to only pass the DCMAKE flags in the Makefile when on MacOS
  - Add instructions for MacOS
    - brew gcc
    - pyenv
    - Notes on finding pyenv using cmake (Python3_FIND_VIRTUALENV)
    - OpenCL on MacOS (does not ship with C++ headers)
      - https://stackoverflow.com/a/62310665/3262054
- [ ] Two main bugs right now
  - cannot find apps folder
  - some Nonetype array error (I've seen this before, I just need to break out
  the debugger)
- [ ] Add fixture to tests that downloads PTB and CIFAR to a data folder to run
in unit tests
