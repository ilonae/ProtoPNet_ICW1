# python libraties
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import makedir
from tqdm import tqdm
import pydicom as dicom
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from typing import AnyStr, BinaryIO, Dict, List, NamedTuple, Optional, Union

image_obj ={}

def crop_box(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
):
    """Draw bounding box on the image"""
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)
    return image[y : y + height, x : x + width]

def read_boxes(
    boxes_fp: pd._typing.FilePathOrBuffer, filepaths_fp: pd._typing.FilePathOrBuffer
) -> pd.DataFrame:
    """Read pandas DataFrame with bounding boxes joined with file paths"""
    df_boxes = pd.read_csv(boxes_fp)
    df_filepaths = pd.read_csv(filepaths_fp)
    primary_key = ("PatientID", "StudyUID", "View")
    if not all([key in df_boxes.columns for key in primary_key]):
        raise AssertionError(
            f"Not all primary key columns {primary_key} are present in bounding boxes columns {df_boxes.columns}"
        )
    if not all([key in df_boxes.columns for key in primary_key]):
        raise AssertionError(
            f"Not all primary key columns {primary_key} are present in file paths columns {df_filepaths.columns}"
        )
    return pd.merge(df_boxes, df_filepaths, on=primary_key)


def _get_image_laterality(pixel_array: np.ndarray) -> str:
    left_edge = np.sum(pixel_array[:, 0])  # sum of left edge pixels
    right_edge = np.sum(pixel_array[:, -1])  # sum of right edge pixels
    return "R" if left_edge < right_edge else "L"


def _get_window_center(ds: dicom.dataset.FileDataset) -> np.float32:
    return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)


def _get_window_width(ds: dicom.dataset.FileDataset) -> np.float32:
    return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)


def dcmread_image(
    fp: Union[str, "os.PathLike[AnyStr]", BinaryIO],
    view: str,
    index: Optional[np.uint] = None,
) -> np.ndarray:
    """Read pixel array from DBT DICOM file"""
    ds = dicom.dcmread(fp)
    ds.decompress(handler_name="pylibjpeg")
    pixel_array = ds.pixel_array
    view_laterality = view[0].upper()
    image_laterality = _get_image_laterality(pixel_array[index or 0])
    if index is not None:
        pixel_array = pixel_array[index]
    if not image_laterality == view_laterality:
        pixel_array = np.flip(pixel_array, axis=(-1, -2))
    window_center = _get_window_center(ds)
    window_width = _get_window_width(ds)
    low = (2 * window_center - window_width) / 2
    high = (2 * window_center + window_width) / 2
    pixel_array = rescale_intensity(
        pixel_array, in_range=(low, high), out_range="dtype"
    )
    return pixel_array

df = read_boxes(boxes_fp="../pruning/input/BCS-DBT boxes-train-v2.csv", filepaths_fp="../pruning/input/BCS-DBT file-paths-train-v2.csv")
print('Size = ', df.size)
print('Dimension = ', df.ndim)
print('Shape = ', df.shape)

for i in range(df.shape[0]):
    box_series = df.iloc[i]
    view = box_series["View"]
    slice_index = box_series["Slice"]
    image_path = os.path.join('../pruning/input/manifest-1617905855234', box_series["descriptive_path"]).replace("\\","/")
    image = dcmread_image(fp=image_path, view=view, index=slice_index)
    x, y, width, height = box_series[["X", "Y", "Width", "Height"]]
    image = crop_box(image=image, x=x, y=y, width=width, height=height)
    folder_name=box_series["descriptive_path"].split("/")[1]
    img_name= ' '.join(box_series["descriptive_path"].split("/")[-2:])
    image_obj[i]={}
    print(i)
    image_obj[i]['img']=image
    image_obj[i]['img_name']=img_name
    image_obj[i]['folder_name']=folder_name
    
print(image_obj.__len__)
    #print(folder_name)
    #makedir('./datasets/BCS_DBT_cropped/train_cropped/'+folder_name)
    #Breast-Cancer-Screening-DBT/DBT-P00060/01-01-2000-DBT-S00787-MAMMO diagnostic digital bilateral-48574/10132.000000-NA-67888/1-1.dcm
    #plt.imsave('./datasets/BCS_DBT_cropped/train_cropped/'+folder_name+"/"+img_name+'.png', image, cmap=plt.cm.gray)
