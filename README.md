# Render Score

## Description

An implementation of the renderscore.
It enables the comparison of MC rendering algorithm (using Mitsuba).

## Getting Started

### Dependencies

Python 2.7:
- numpy
- scipy

Additional:
- Mitsuba Renderer

### Installing

1. Setup your Python installation.
(for Anaconda users see .conda.yml)

2. Download Mitsuba from: https://www.mitsuba-renderer.org/index_old.html (see documentation to ensure that Python will be able to find the Mitsuba core libraries)

3. Copy or install the this module.

### Executing program

1. Setup Mistuba XML files for the reference, e.g. "./data/veach_ajar/bdpt.xml".
2. Setup Mistuba XML files for the test / comparison, e.g. "./data/veach_ajar/path.xml" and "./data/veach_ajar/erpt.xml"
3. Make sure sampler parameters match for all of the used scenes.
4. Choose iteration count (e.g. 1024 for the reference and 32 for the test runs) and execute:
```
python -m renderscore 1024 ./data/veach_ajar/bdpt.xml 32 ./data/veach_ajar/path.xml ./data/veach_
ajar/erpt.xml 
```

For detailed command line parameters see help:
```
python -m renderscore -h
``` 

## Authors

Christian Freude, freude (at) cg.tuwien.ac.at

## Version History

* 1.0.0
    * Initial Release

## License

This project is licensed under the GNU GPL LICENSE - see the LICENSE.md file for details

## Acknowledgments

Funded by Austrian Science Fund (FWF): ORD 61