import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # chỉ số của thuộc tính được chọn để phân chia, thuộc tính phù hợp giúp việc phân chia cây chinh xác hơn
        self.threshold = threshold  # giá trị ngưỡng để phân chia nhánh của cây
        self.left = left  # Node con bên trái
        self.right = right  # Node con bên phải
        self.value = value  # giá trị dự đoán cho lá của cây

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        #Số hàng, số cột của tập dữ liệu đặc trưng
        n_samples, n_features = X.shape
        # print(X.shape)

        #Xác định số giá trị target khác nhau của tập target, dựa vào số lượng này để phân chia sô nhánh cây quyết định
        n_labels = len(np.unique(y))

        # Điều kiện dừng khi xây dựng cây
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < 2:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Lấy điểm phân chia tốt nhất
        best_feature, best_threshold = self._best_split(X, y)

        # Tiến hành phân chiaz
        left_indices = X.iloc[:, best_feature] < best_threshold #Tập giá trị đặc trưng nhỏ hơn best_threshold
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth=depth + 1)
        right_indices = X.iloc[:, best_feature] >= best_threshold    #Tập còn lại
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth=depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            # print(X[:, feature])
            #Lấy ra các giá trị duy nhất của mỗi cột
            col = X.iloc[:, feature]
            thresholds = np.unique(col)
            for threshold in thresholds:
                gain = self._information_gain(y, X.iloc[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    #Tính entropy
    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)  # tìm tất cả các giá trị riêng biệt và đếm số lần xuất hiện
        p = counts / len(y)  # tính tần số xuất hiện của từng giá trị trong mảng y
        return -np.sum(p * np.log2(p))  # tính entropy dựa trên tần số xuất hiện của từng giá trị

    #Tính giá trị của lợi ích thông tin (Entropy tb trước - Entropy tb sau)
    def _information_gain(self, y, feature, threshold):
        parent_entropy = self.entropy(y)
        # Tập con có thuộc tính feature nhỏ hơn threshold
        left_indices = feature < threshold
        left_entropy = self.entropy(y[left_indices])
        # Tập con có thuộc tính feature lớn hơn threshold
        right_indices = feature >= threshold
        right_entropy = self.entropy(y[right_indices])

        # Tính độ tinh khiết trung bình
        n_left, n_right = len(y[left_indices]), len(y[right_indices]) #Kích thước tập con
        child_entropy = (n_left / len(y)) * left_entropy + (n_right / len(y)) * right_entropy

        return parent_entropy - child_entropy

    def _leaf_value(self, y):
        value = np.mean(y)
        prob = 1 / (1 + np.exp(-value))
        # Giá trị node lá
        # ds, counts = np.unique(y, return_counts=True)
        # return ds[np.argmax(counts)]
        return prob
    #Dự đoán cho mẫu đưa vào
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x = X.iloc[i, self.tree.feature_index]
            prediction = self._traverse_tree(x, self.tree)
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)
        return np.array(predictions)

    #Duyệt cây
    def _traverse_tree(self, x, node):
        #node là lá
        if node.left is None and node.right is None:
            return node.value
        #node không phải là lá và có giá trị của feature < threshold thì đi sang trái
        if x < node.threshold:
            return self._traverse_tree(x, node.left)
        #đi sang phải
        else:
            return self._traverse_tree(x, node.right)

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.2, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        # Khởi tạo giá trị dự đoán ban đầu là trung bình của các giá trị y
        y_pred = np.full(y.shape, np.mean(y))
        y_pred = y_pred.astype(np.float64)
        for i in range(self.n_estimators):
            # Tính gradient và loss
            gradient = self._gradient(y, y_pred)
            # loss = self.loss(y, y_pred)
            # Huấn luyện một cây quyết định với gradient của dữ liệu hiện tại
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, gradient)
            # Cập nhật dự đoán
            update = self.learning_rate * tree.predict(X)
            y_pred = y_pred + update
            # print(y_pred)
            # Lưu tree vào danh sách các models
            self.models.append(tree)

    # def predict(self, X):
    #     # Tính dự đoán bằng cách lấy trung bình dự đoán của các cây quyết định
    #     y_pred = np.zeros(X.shape[0])
    #     for model in self.models:
    #         y_pred += self.learning_rate * model.predict(X)
    #     return y_pred
    #
    # def loss(self, y, y_pred):
    #     # Tính hàm mất mát (MSE)
    #     return np.mean((y - y_pred) ** 2)
    #
    # def loss_gradient(self, y, y_pred):
    #     # Tính đạo hàm của hàm mất mát MSE theo y_pred
    #     return -2 * (y - y_pred)

    def predict(self, X):
        # Tính giá trị dự đoán bằng cách lấy giá trị trung bình của tất cả các cây
        y_pred = np.zeros(X.shape[0])
        y_pred = y_pred + sum(self.learning_rate * model.predict(X) for model in self.models)
        # Áp dụng hàm sigmoid để chuyển đổi giá trị dự đoán thành xác suất
        y_pred = np.round(y_pred).astype(int)
        return y_pred

    def _gradient(self, y, y_pred):
        # Tính gradient của hàm mất mát (cross-entropy)
        return y - self._sigmoid(y_pred)

    # def _gradient(self, y, y_pred, x, lambda_):
    #     # Tính gradient của hàm mất mát (cross-entropy) với L1 regularization
    #     return -(y - y_pred) * x + lambda_ * np.sign(self._weights)
    #Sai số
    def _sigmoid(self, x):
        # Hàm sigmoid để chuyển đổi giá trị dự đoán thành xác suất
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    # Load dữ liệu
    # data = pd.read_csv('Hotel Reservations.csv')
    data = pd.read_csv('Hotel Reservations.csv')
    # Bỏ hàng thiếu dữ liệu
    data = data.dropna()
    # Thay thế phần "INN" bằng một chuỗi rỗng
    data['Booking_ID'] = data['Booking_ID'].str.replace('INN', '')

    # Chuyển đổi kiểu dữ liệu của các cột có kiểu chữ sang kiểu số
    data['Booking_ID'] = data['Booking_ID'].astype(int)
    data['type_of_meal_plan'] = data['type_of_meal_plan'].replace(
        {'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3,'Not Selected': 4})
    data['room_type_reserved'] = data['room_type_reserved'].replace(
        {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7})
    data['market_segment_type'] = data['market_segment_type'].replace({'Online': 1, 'Offline': 2, 'Corporate': 3, 'Aviation': 4, 'Complementary': 5})
    data['booking_status'] = data['booking_status'].replace({'Not_Canceled': 1, 'Canceled': 0})

    # Chia dữ liệu thành train và test set
    X = data.drop(['booking_status'], axis=1)
    y = data['booking_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # dt = DecisionTree()
    # dt.fit(X_train, y_train)
    # dt.predict(X_test)
    # best_feature, best_threshold = dt._best_split(X_train, y_train)
    # gb = GradientBoosting()
    # gb.fit(X_train, y_train)
    # print(best_feature)
    # print(best_threshold)
    # Khởi tạo mô hình
    gb = GradientBoosting(n_estimators=50, learning_rate=0.2, max_depth=3)
    # print(gb._gradient(0, 1))
    # Huấn luyện mô hình
    gb.fit(X_train, y_train)

    # #Dự đoán kết quả trên tập test
    y_pred = gb.predict(X_test)
    # #Tính toán độ chính xác của mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
