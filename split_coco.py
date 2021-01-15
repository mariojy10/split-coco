import json
import argparse
from pathlib import Path

from pycocotools.coco import COCO 
from sklearn.model_selection import train_test_split, KFold


def save_coco(save_path,info:dict,licenses:list,images:list,anns:list,cats:list):
    ds = {'info':info,'licenses':licenses,'images':images,'annotations':anns,'categories':cats}
    json_object = json.dumps(ds,indent=4)
    with open(save_path, 'w') as f:
        f.write(json_object)


def filter_image(images:list,train_ids,val_ids):
    train_data = list(filter(lambda x: x['id'] in train_ids, images))
    val_data = list(filter(lambda x: x['id'] in val_ids, images))
    return train_data,val_data


def filter_annotation(anns:list,train_ids,val_ids):
    train_anns = list(filter(lambda x: x['image_id'] in train_ids, anns))
    val_anns = list(filter(lambda x: x['image_id'] in val_ids, anns))
    return train_anns,val_anns


def filter_id(ids:list,train_idx,val_idx):
    train_ids = list(filter(lambda x: x in train_idx, ids))
    val_ids = list(filter(lambda x: x in val_idx, ids))
    return train_ids,val_ids


def main(args):
    coco = COCO(args.annotation)

    images = list(coco.imgs.values())
    anns = list(coco.anns.values())
    cats = list(coco.cats.values())
   
    dataset = coco.dataset
    try:
        info = dataset['info']
    except:
        info = dict() 
    try:
        licenses = dataset['licenses']
    except:
        licenses = list()

    #split using image_ids
    image_ids = coco.getImgIds()
    
    if args.command == 'split':
        train_ids, val_ids = train_test_split(image_ids,train_size=args.train_ratio,random_state=args.seed)
        train_images, val_images = filter_image(images,train_ids,val_ids)
        train_anns, val_anns = filter_annotation(anns,train_ids,val_ids)
        save_coco(str(Path(args.output_dir).joinpath('instances_train.json')),info,licenses,train_images,train_anns,cats)
        save_coco(str(Path(args.output_dir).joinpath('instances_val.json')),info,licenses,val_images,val_anns,cats)
        print('train-test split done!')

    elif args.command == 'kfold':
        kf = KFold(n_splits=args.num_folds,shuffle=True,random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(image_ids),1):
            train_ids, val_ids = filter_id(image_ids,train_idx,val_idx)
            train_images, val_images = filter_image(images,train_ids,val_ids)
            train_anns, val_anns = filter_annotation(anns,train_ids,val_ids)
            save_coco(str(Path(args.output_dir).joinpath(f'instances_train{fold}.json')),info,licenses,train_images,train_anns,cats)
            save_coco(str(Path(args.output_dir).joinpath(f'instances_val{fold}.json')),info,licenses,val_images,val_anns,cats)
        print(f'{args.num_folds}-fold split done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split COCO annotations')
    parser.add_argument('--annotation',type=str,help='path to COCO annotation file')
    parser.add_argument('--seed',type=int,default=1,help='seed value')
    parser.add_argument('--output-dir',type=str,default='.',help='path to save train json')

    subparser = parser.add_subparsers(dest='command')
    split = subparser.add_parser('split')
    split.add_argument('--train-ratio',type=float,help='split ratio for trainset',required=True)
    kfold = subparser.add_parser('kfold')
    kfold.add_argument('--num-folds',type=int,help='total fold split',required=True)

    args = parser.parse_args()
    main(args)