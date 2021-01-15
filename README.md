# split-coco
Python tool to split COCO *(like)* json annotation. <br/>
Split was done using image_ids.

## Dependencies
- python3
- pycocotools
- sklearn
```
pip install -r requirements.txt
```

### Usage
```
python split_coco.py -h
```
#### Example
```
#train-test-split
python split_coco.py --annotation path/to/coco/annotation split --train-ratio 0.8

#kfold-split
python split_coco.py --annotation path/to/coco/annotation kfold --num-folds 5
```
