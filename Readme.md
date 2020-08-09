**Guide to run**
---
**Requirement:**  
Install mxnet if your computer not have:
```
Example: mxnet gpu for cuda-9.0
pip install mxnet-cu90
```
```
git clone https://github.com/phamvandan/Document-id-to-selfie-face-matching.git
git clone https://github.com/deepinsight/insightface.git
```
Pretrained _model_ can be download from here:  
https://drive.google.com/open?id=1qSwX7hDmww-A2Zwo5EUP9nZaOpc3RLJw  
```
cd Document-id-to-selfie-face-matching
python3 main.py --model="path_to_model,0" --det 0 -f path_to_image_folder
Example:
python3 main.py --model="/media/dan/Storage/Code/r100-triplet-chiya_v1_selfie_card_new/model,0" --det 0 -f test/
```
_Note:_  
-det 0 : face detection and face matching mode
-det 1 : face matching only
