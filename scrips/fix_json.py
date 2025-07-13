import json

input_json = "../data/nerf_data_mixed/transforms.json"
output_json = "../data/nerf_data_mixed/transforms.json"

W, H = 128, 128
fl_x = fl_y = 0.5 * W
cx = W / 2
cy = H / 2

with open(input_json, 'r') as f:
    data = json.load(f)

for frame in data['frames']:
    frame['fl_x'] = fl_x
    frame['fl_y'] = fl_y
    frame['cx'] = cx
    frame['cy'] = cy

with open(output_json, 'w') as f:
    json.dump(data, f, indent=4)

print(f"✅ save to  {output_json}")
