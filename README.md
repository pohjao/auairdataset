# Fork of AU-AIR Dataset API with TFRecord converter
## Changes made from the original repo
* tfrecord exporting function
* invalid bounding box filtering

For TFRecord exporting see

```
python auairtools/to_tfrecord.py --helpshort
```
```
auairtools/to_tfrecord.py:
  --annotations: Path to annotations json-file
    (default: './data/annotations.json')
  --data_dir: Path to auair images directory
    (default: './data/images/')
  --output_dir: Output location.
    (default: './data/tfrecords')
  --splits: List of split percentages: train,val. Rest of the data is split into the test set.
    (default: '80,10')
    (a comma separated list)

```

The output consists of train, val and test datasets. Class namefile is also exported.


## Dataset
The AU-AIR is a multi-modal aerial dataset captured by a UAV. Having visual data, object annotations, and flight data (time, GPS, altitude, IMU sensor data, velocities), AU-AIR meets vision and robotics for UAVs.

https://bozcani.github.io/auairdataset

![alt text](https://raw.githubusercontent.com/bozcani/auairdataset/master/intro.jpg) 

### Specifications
- 8 raw RGB videos (more than 2 hours in total)
- 32,283 extracted and labelled frames
- Bounding box annotations for eight objects related to traffic:
	- human, car, van, truck, bike, motorbike, bus, trailar
- Time, GPS, altitude, IMU sensor data and linear velocities of the drone are avaliable for each extracted frame.

### Download
Please download both the AU-AIR images and annotations to run the demo and use the API:

Images: https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ (2.2 GB)

Annotations: https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA (3.9 MB)

## Dependencies
You will need common dependencies like `numpy` and `opencv`.

## Installation
To install the package from source, simply clone or download the repository to your machine
`git clone https://github.com/bozcani/auairdataset`

`cd auairdataset`

`python setup.py install`

## References

[1] I. Bozcan and E. Kayacan, "AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance", submitted to IEEE International Conference on Robotics and Automation 2020.
