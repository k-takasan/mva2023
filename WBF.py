#!/usr/bin/env python
# coding: utf-8

import csv, json
import pandas as pd
from ensemble_boxes import *
import subprocess
import warnings
warnings.simplefilter('ignore')

IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

def json_to_csv(input_file, output_file):
    # JSONファイルの読み込み
    with open(input_file, 'r') as f:
        data = json.load(f)

    # アノテーション情報の取得
    annotations = []
    for annotation in data: #['annotations']
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        score = annotation['score']
        x, y, w, h = annotation['bbox']
        annotations.append({
            'image_id': image_id,
            'score': score,
            'category_id': category_id,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })

    # CSVファイルに書き込み
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'category_id','score', 'x', 'y', 'w', 'h'])
        for annotation in annotations:
            writer.writerow([
                annotation['image_id'],
                annotation['category_id'],
                annotation['score'],
                annotation['x'],
                annotation['y'],
                annotation['w'],
                annotation['h']
            ])

def convert_df(df):
    df['x+w'] = (df['x'] + df['w'])/IMAGE_WIDTH
    df['y+h'] = (df['y'] + df['h'])/IMAGE_HEIGHT
    df['x'] /= IMAGE_WIDTH
    df['y'] /= IMAGE_HEIGHT
    return df


def WBF(file_path1, file_path2, file_path3, file_path4, file_path5, output_file_path, weights, score_thres, iou_thres):
    # csvファイルの読み込み
    df1 = pd.read_csv(file_path1)
    df1 = convert_df(df1)
    df2 = pd.read_csv(file_path2)
    df2 = convert_df(df2)
    df3 = pd.read_csv(file_path3)
    df3 = convert_df(df3)
    df4 = pd.read_csv(file_path4)
    df4 = convert_df(df4)
    df5 = pd.read_csv(file_path5)
    df5 = convert_df(df5)

    # 画像ごとにループ
    for img_id in df1['image_id'].unique():
        # ファイル1とファイル2の該当する画像の行を取得
        df1_img = df1[df1['image_id'] == img_id]
        df2_img = df2[df2['image_id'] == img_id]
        df3_img = df3[df3['image_id'] == img_id]
        df4_img = df4[df4['image_id'] == img_id]
        df5_img = df5[df5['image_id'] == img_id]
        # ボックス座標を[x1,y1,x2,y2]のフォーマットに変換
        boxes_list = [df[['x', 'y', 'x+w', 'y+h']].values.tolist() for df in [df1_img, df2_img, df3_img, df4_img, df5_img]]
        # スコアとラベルを取得
        scores_list = [df['score'].tolist() for df in [df1_img, df2_img, df3_img, df4_img, df5_img]]
        labels_list = [df['category_id'].tolist() for df in [df1_img, df2_img, df3_img, df4_img, df5_img]]
        
        # NMW
        #boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thres, skip_box_thr=score_thres)
        # WBFアンサンブルを実行
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thres, skip_box_thr=score_thres) #
        # 結果を新しいデータフレームに追加
        df_result = pd.DataFrame({
            'image_id': [img_id] * len(boxes),
            'category_id': labels,
            'score': scores,
            'x': [box[0]*IMAGE_WIDTH for box in boxes],
            'y': [box[1]*IMAGE_HEIGHT for box in boxes],
            'w': [(box[2] - box[0])*IMAGE_WIDTH for box in boxes],
            'h': [(box[3] - box[1])*IMAGE_HEIGHT for box in boxes]
        })
        # filtering
        #df_result = df_result.query("score>0.1")
        # 結果をファイルに書き込み（初回のみヘッダーを書き込む）
        with open(output_file_path, 'a') as f: # appendに注意
            df_result.to_csv(f, header=f.tell()==0, index=False)

            
def csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # ヘッダー行をスキップする
        for i,row in enumerate(reader):
            image_id = int(row[0])
            bbox = [float(row[3]), float(row[4]), float(row[5]), float(row[6])]
            score = float(row[2])
            category_id = int(float(row[1]))
            #if image_id not in data:
             #   data[image_id] = []
            data.append({"image_id": image_id, "bbox": bbox, "score": score, "category_id": category_id})

    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    
    # jsonをcsvに変換
    input1 = "../sahi/centernet_resnext101/result.json" #"centernet_resnext101_strongfit_large/results.json"
    output1 = "./WBF/centernet_resnext101_sahi.csv" #"./WBF/centernet_resnext101_strongfit_large.csv"
    input2 = "../sahi/detr_resnet50/result.json" #"detr_strongfit/results.json"
    output2 = "./WBF/d-detr_sahi.csv" #"./WBF/d-detr_strongfit.csv"
    input3 = "../sahi/centernet_resnet101/result.json" #centernet_resnet101/results.bbox.json
    output3 = "./WBF/centernet_resnet101_sahi.csv" #./WBF/centernet_resnet101.csv
    input4 = "centripetalnet_hourglass104/results.bbox.json" #"../sahi/centripetalnet/result.json"
    output4 = "./WBF/centripetalnet_hourglass104.csv" #"./WBF/centripetalnet_sahi.csv"
    input5 = "../sahi/detr_resnet101/result.json" #"detr_resnet101/results.bbox.json"
    output5 = "./WBF/d-detr_resnet101_sahi.csv" #"./WBF/d-detr_resnet101.csv"
    
    json_to_csv(input1, output1)
    json_to_csv(input2, output2)
    json_to_csv(input3, output3)
    json_to_csv(input4, output4)
    json_to_csv(input5, output5)
    
    weights = [3, 2, 1, 1, 1]
    score_thres = 0.2 #
    iou_thres = 0.4 #
    output_file_path = f"./WBF/WBF_5_sahi_th{score_thres}-{iou_thres}_{weights[0]}vs{weights[1]}vs{weights[2]}vs{weights[3]}vs{weights[4]}.csv" #
    WBF(output1, output2, output3, output4, output5, output_file_path, weights, score_thres, iou_thres)
    
    csv_file_path = output_file_path #"output.csv"
    json_file_path = csv_file_path + ".json" #"output.json"
    csv_to_json(csv_file_path, json_file_path)
    
    # 一時ファイルを削除
    cmd = 'rm {}'.format(csv_file_path)
    subprocess.run(cmd, shell=True)
    
    # 提出用ファイルを保存
    cmd = 'zip -j {} {}'.format(json_file_path[:-9]+".zip", json_file_path)
    subprocess.run(cmd, shell=True)