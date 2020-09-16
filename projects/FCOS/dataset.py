import os
import logging
import pickle
from datetime import datetime
import numpy as np

import xml.etree.ElementTree as ET
from multiprocessing import Manager
from multiprocessing import Pool

from fvcore.common.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger("dataset.py")

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)


def parse_batch(args):
    dicts = args[0]
    absolute = args[2]
    
    if not absolute:
        fileid = args[1]
        images_dir = args[3]
        annotation_dir = args[4]
        class_names = args[5]
    else:
        image_path = args[1]
        class_names = args[3]
    """
    Args:
        dicts (list from Manager): 多线程共享的list类型
        fileid (str): 数据集一个图片文件名去掉后缀
        image_path (str): 数据集一个图片文件路径
        images_dir (str): 数据集图片文件根目录
        annotation_dir (str): 数据集标签文件根目录
        class_names (tuple or list): 数据集类别名
    """
    if not absolute:
        fileid = fileid.replace(".jpg", "").replace(".jpeg", "")
        anno_file = os.path.join(annotation_dir, fileid + ".xml")
        jpeg_file = os.path.join(images_dir, fileid + ".jpg")
    else:
        image_path = image_path.replace("\\", "/")
        images_dir = os.path.dirname(image_path)
        # 去后缀的文件名
        fileid = os.path.basename(image_path)
        fileid = fileid.replace(".jpg", "").replace(".jpeg", "")
    
        anno_dir = os.path.join(os.path.dirname(images_dir), "Annotations")
        anno_file = os.path.join(anno_dir, fileid + ".xml")
        
        jpeg_file = image_path
        
    try:
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)
    except Exception as e:
        print(e)
        print(anno_file)
        print(images_dir)
        print(image_path)
        raise ValueError
        
    r = {
        "file_name": jpeg_file,
        "image_id": fileid,
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
    }
    instances = []
    
    for obj in tree.findall("object"):
        cls = obj.find("name").text
        # We include "difficult" samples in training.
        # Based on limited experiments, they don't hurt accuracy.
        # difficult = int(obj.find("difficult").text)
        # if difficult == 1:
        # continue
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        bbox[0] -= 1.0
        bbox[1] -= 1.0
        if cls not in class_names: # hb测试集类别名字比较乱("001"?), 但是只有一个
            cls = class_names[0]
            logger.info("xml file: {}, class {} not in class_names, change to class_names[0]: {}".format(anno_file, cls, class_names[0]))
        instances.append(
            {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
        )
    r["annotations"] = instances
    dicts.append(r)


def load_instances(dirname, settxt, class_names, absolute=False, threads=10, pool_batch_size=1000):
    """
    要求数据集格式(注意图片和标签文件名去掉后缀是一一对应的):
        {dirname}/
            Annotations/
                eg_001.xml
                ...
            JPEGImages/
                eg_001.jpg
                ...
            {settxt}:
                eg_001.jpg
                ...
            
    解析后的数据会生成一个instances.pkl文件保存在dirname下
    如果settxt=="train_xxx.txt", 那么pkl文件名为instances_xxx.pkl
    
    Args:
        dirname (str): 数据集根目录
        settxt (str): 数据集根目录下, 所有图片路径, 一个一个图片
        absolute (bool): settxt中的路径是!!绝对路径!!还是!!文件名!!
        class_names (tuple or list): 数据集类别名
        threads (int): 多线程个数
        pool_batch_size (int): 每个线程每次处理图片个数
    """
    postfix = settxt.replace(".txt", "").split("train_")
    poststr = ""
    if len(postfix) == 1:
        poststr = ""
    elif len(postfix) == 2:
        poststr = postfix[-1]
    else:
        raise ValueError("incorrect settxt name: {}".format(settxt))
    
    if len(poststr)>0:
        instances_pkl = os.path.join(dirname, "instances_{}.pkl".format(poststr))
    else:
        instances_pkl = os.path.join(dirname, "instances.pkl")
        
    if os.path.exists(instances_pkl):
        print("instances_pkl exists, loading directly.")
        dicts = load_obj(instances_pkl)["instances"]
        return dicts
    
    begin = datetime.now()
    print("building and loading instances... please wait")
#    with PathManager.open(os.path.join(dirname, settxt)) as f:
#        filepaths = np.loadtxt(f, dtype=np.str)
    with open(os.path.join(dirname, settxt)) as f:
        filepaths = [line.strip() for line in f.readlines() if line!="\n"]
    
    if not absolute:
        # 标签文件目录
        annotation_dir = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
        # 图片文件目录
        images_dir = PathManager.get_local_path(os.path.join(dirname, "JPEGImages/"))
    
    dicts = Manager().list()
    
    args_list = []
    pool = Pool(threads)
    length = len(filepaths)
    progress_unit = int(0.01 * length) if length>10000 else int(0.1 * length)
    progress_unit = progress_unit if progress_unit >= 1 else 1
    for i, filepath in enumerate(filepaths):
        progress = i+1
        if progress % progress_unit ==0:
            print("{:.1f}%...".format(progress/length*100))
            
        if not absolute: # settxt里给的是图片文件名
            args_list.append([dicts, filepath, absolute, images_dir, annotation_dir, class_names])
        else: # settxt里给的是图片文件绝对路径
            image_path = filepath
            args_list.append([dicts, image_path, absolute, class_names])
        
        if len(args_list) == pool_batch_size: # 每次达到batch_size后就使用pool多线程处理, 该batch处理完毕后才继续
            _ = pool.map(parse_batch, args_list)
            args_list = []
            
    if len(args_list) > 0:
        _ = pool.map(parse_batch, args_list)
        
    pool.close()
    pool.join()
        
    time_cost = datetime.now() - begin
    print("loading instances done. cost time: %s" % (time_cost))
    print("converting Manager().list() to list...")
    dicts = [item for item in dicts]
    
    print("saving instances...")
    save_obj({"instances": dicts}, instances_pkl)
    
    return dicts


def register_dataset(name, dirname, settxt, class_names, absolute, evaluator_type, threads=10, pool_batch_size=1000):
    """
    Args:
        name (str): 数据集名称
        dirname (str): 数据集所在目录
        settxt (str): 数据集目录下的文件名集合, 一行一个文件名
        threads (int): 多线程个数
        pool_batch_size (int): 每个线程每次处理图片个数
    """
    DatasetCatalog.register(name, lambda: load_instances(dirname, settxt, class_names, absolute, threads, pool_batch_size))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names),
        dirname=dirname,
        evaluator_type=evaluator_type
    )