# created by chatGPT
import json
import argparse

def filter_coco_annotations(input_file, output_file, threshold):
    # COCOフォーマットのJSONファイルを読み込み
    with open(input_file, 'r') as f:
        annotations = json.load(f)

    # 抽出対象の画像IDを格納するリストを初期化
    image_ids = []

    # COCOフォーマットのJSONファイルから、面積が閾値以上のアノテーションが含まれる画像IDを抽出
    for annotation in annotations['annotations']:
        if annotation['area'] >= threshold:
            image_id = annotation['image_id']
            if image_id not in image_ids:
                image_ids.append(image_id)

    # 抽出対象の画像情報を格納する辞書を初期化
    new_images = []
    new_annotations = []

    # COCOフォーマットのJSONファイルから、抽出対象の画像情報とアノテーション情報を取得
    for image in annotations['images']:
        if image['id'] in image_ids:
            new_images.append(image)
    for annotation in annotations['annotations']:
        if annotation['image_id'] in image_ids:
            new_annotations.append(annotation)

    # 新しいJSONファイルに必要な情報を格納する辞書を作成
    new_data = {}
    new_data['info'] = annotations['info']
    new_data['licenses'] = annotations['licenses']
    new_data['images'] = new_images
    new_data['annotations'] = new_annotations
    new_data['categories'] = annotations['categories']

    # 新しいJSONファイルを保存
    with open(output_file, 'w') as f:
        json.dump(new_data, f)

if __name__ == '__main__':
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Filter COCO annotations by area')
    parser.add_argument('input_file', type=str, help='input COCO format JSON file')
    parser.add_argument('output_file', type=str, help='output COCO format JSON file')
    parser.add_argument('threshold', type=int, help='area threshold')
    args = parser.parse_args()

    # COCOフォーマットのJSONファイルを指定されたファイル名でフィルタリングして保存
    filter_coco_annotations(args.input_file, args.output_file, args.threshold)
