Object Detection and Tracking Using YOLOv5 and motpy

![Hnet com-image](https://user-images.githubusercontent.com/188394/111580817-8736f900-87fb-11eb-88bd-fc5e24ecbae9.gif)

# Installation

```
git clone --recursive https://github.com/toyoshi/yolov5-motpy
pip install -r requirements.txt
```

# example

use webcam
```
python tracker.py --souce 0
```

track only people 
```
python3 tracker.py --source 0 --classes 0
```

## Count and send data to ambient

Send data to ambient(https://ambidata.io/) example

```
python track_and_send_ambient.py --source 0 --classes 0 --conf-thres 0.6 --ambient-channel {channnel} --ambient-wkey {key}
```

# See Also
- https://github.com/ultralytics/yolov5
- https://github.com/wmuron/motpy
