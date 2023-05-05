import argparse
import csv
import json

def json_to_csv(input_file, output_file):
    # JSONファイルの読み込み
    with open(input_file, 'r') as f:
        data = json.load(f)

    # カテゴリー情報の取得
    # categories = {}
    # for category in data['categories']:
    #     categories[category['id']] = category['name']

    # 画像情報の取得
    # images = {}
    # for image in data['images']:
    #     images[image['id']] = {
    #         'width': image['width'],
    #         'height': image['height'],
    #         'file_name': image['file_name']
    #     }

    # アノテーション情報の取得
    annotations = []
    for annotation in data: #['annotations']
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        score = annotation['score']
        x, y, w, h = annotation['bbox']
        annotations.append({
            'image_id': image_id,
            #'width': images[image_id]['width'],
            #'height': images[image_id]['height'],
            #'file_name': images[image_id]['file_name'],
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
                #annotation['width'],
                #annotation['height'],
                #annotation['file_name'],
                annotation['category_id'],
                annotation['score'],
                annotation['x'],
                annotation['y'],
                annotation['w'],
                annotation['h']
            ])
            
def csv_to_json(input_file, output_file):
    # CSVファイルの読み込み
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # 画像情報の作成
    images = {}
    for row in rows:
        image_id = int(row[0])
        images[image_id] = {
            'id': image_id,
            'width': int(row[1]),
            'height': int(row[2]),
            'file_name': row[3]
        }

    # アノテーション情報の作成
    annotations = []
    for row in rows:
        image_id = int(row[0])
        category = row[4]
        x, y, w, h = map(float, row[5:])
        annotations.append({
            'image_id': image_id,
            'category': category,
            'bbox': [x, y, w, h]
        })

    # カテゴリー情報の作成
    categories = []
    for annotation in annotations:
        category = annotation['category']
        if category not in categories:
            categories.append(category)

    # JSONファイルに書き込み
    data = {
        'images': list(images.values()),
        'annotations': annotations,
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(categories)]
    }
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
def main():
    parser = argparse.ArgumentParser(description='Convert between COCO JSON and CSV format.')
    parser.add_argument('input_file', help='input file name')
    parser.add_argument('output_file', help='output file name')
    parser.add_argument('--from-csv', dest='from_csv', action='store_true', help='convert from CSV to JSON')
    parser.add_argument('--from-json', dest='from_csv', action='store_false', help='convert from JSON to CSV')
    parser.set_defaults(from_csv=True)
    args = parser.parse_args()

    if args.from_csv:
        csv_to_json(args.input_file, args.output_file)
    else:
        json_to_csv(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
