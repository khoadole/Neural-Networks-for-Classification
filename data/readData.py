import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_file = 'cifar-10-batches-py/data_batch_1'  # Hoặc data_batch_2, 3, 4, 5
batch = unpickle(batch_file)

images = batch[b'data']
labels = batch[b'labels']

print("Hình dạng mảng ảnh:", images.shape)
print("Hình dạng mảng nhãn:", len(labels))

print("\nThông số ảnh:")
print("Giá trị pixel nhỏ nhất:", images.min())
print("Giá trị pixel lớn nhất:", images.max())
print("Giá trị pixel trung bình:", images.mean())
print("Độ lệch chuẩn pixel:", images.std())

print("\nThông tin nhãn:")
unique_labels, counts = np.unique(labels, return_counts=True)
print("Các nhãn có trong batch:", unique_labels)
print("Số lượng mỗi nhãn:", dict(zip(unique_labels, counts)))