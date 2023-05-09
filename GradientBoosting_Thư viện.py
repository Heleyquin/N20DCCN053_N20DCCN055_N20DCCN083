from tkinter import ttk

import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
# from joblib import dump
from joblib import load

class Gradient_Boosting_GUI:

    def __init__(self, master):
        self.master = master
        master.title("Gradient Boosting Demo")
        ##################################
        # Đọc dữ liệu từ file lưu vào dataframe
        self.data = pd.read_csv('Hotel Reservations.csv')
        #Xử lý dữ liệu
        #Bỏ hàng không có dữ liệu
        self.data = self.data.dropna()
        # Bỏ cột Booking_ID
        self.data = self.data.drop(['Booking_ID'], axis=1)

        #Lấy ds dữ liệu đăc trưng
        self.X = self.data.drop(['booking_status'], axis=1)
        #Lấy cột dữ liệu mục tiêu
        self.Y = self.data['booking_status']
        #Chia làm tập huấn luyện và kiểm tra
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=42)

        # Mô hình gradient boosting
        self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.X_train = pre_processing(self.X_train)
        self.Y_train = self.Y_train.replace({'Not_Canceled': 1, 'Canceled': 0})
        self.clf.fit(self.X_train, self.Y_train)
        #Lưu lại mô hình đã train
        # dump(self.clf, 'model.joblib')
        #Lấy mô hình đã train ra để dùng
        # self.clf = load('model.joblib')


        # Tạo combobox cho mỗi thuộc tính
        self.no_of_adults_box = self.create_combobox("Number of adults", get_unique_values("no_of_adults", self.data))
        self.no_of_children_box = self.create_combobox("Number of children", get_unique_values("no_of_children", self.data))
        self.no_of_weekend_nights_box = self.create_combobox("Number of weekend nights", get_unique_values("no_of_weekend_nights", self.data))
        self.no_of_week_nights_box = self.create_combobox("Number of week nights", get_unique_values("no_of_week_nights", self.data))
        self.type_of_meal_plan_box = self.create_combobox("Type of meal plan", get_unique_values("type_of_meal_plan", self.data))
        self.required_car_parking_space_box = self.create_combobox("Required car parking space", get_unique_values("required_car_parking_space", self.data))
        self.room_type_reserved_box = self.create_combobox("Room type reserved", get_unique_values("room_type_reserved", self.data))
        self.lead_time_box = self.create_combobox("Lead time", get_unique_values("lead_time", self.data))
        self.arrival_year_box = self.create_combobox("Arrival year", get_unique_values("arrival_year", self.data))
        self.arrival_month_box = self.create_combobox("Arrival month", get_unique_values("arrival_month", self.data))
        self.arrival_date_box = self.create_combobox("Arrival date", get_unique_values("arrival_date", self.data))
        self.market_segment_type_box = self.create_combobox("Market segment type", get_unique_values("market_segment_type", self.data))
        self.repeated_guest_box = self.create_combobox("Repeated guest", get_unique_values("repeated_guest", self.data))
        self.no_of_previous_cancellations_box = self.create_combobox("Number of previous Cancellations", get_unique_values("no_of_previous_cancellations", self.data))
        self.no_of_previous_bookings_not_canceled_box = self.create_combobox("Number of previous booking not canceled", get_unique_values("no_of_previous_bookings_not_canceled", self.data))
        self.avg_price_per_room_box = self.create_combobox("Average price per room", get_unique_values("avg_price_per_room", self.data))
        self.no_of_special_requests_box = self.create_combobox("Number of special requests", get_unique_values("no_of_special_requests", self.data))
        # Tạo nút "Dự đoán"
        self.predict_button = tk.Button(master, text="Predict", command=self.predict_class)
        self.predict_button.pack()

        # Tạo kết quả dự đoán
        self.prediction_label = tk.Label(master, text="",padx=200, pady=10)
        self.prediction_label.pack()

        self.accuracy_label = tk.Label(master, text ="Accuracy: " + str(self.accuracy()),padx= 0, pady=0 )
        self.accuracy_label.pack()

    def create_combobox(self, label_text, values):
        frame = tk.Frame(self.master, padx=10, pady=5)
        frame.pack(side=tk.TOP)
        label = tk.Label(frame, text=label_text, padx=10, pady=3, width=35, anchor='w')
        label.pack(side=tk.LEFT,padx=(0, 5))
        combobox = ttk.Combobox(frame, values=values)
        combobox.pack(side=tk.RIGHT, padx=5, pady=5)
        return combobox
    def predict_class(self):
        #ẩn label đã dự đoán trước đó
        self.prediction_label.config(text="Prediction: ")
        # Lấy giá trị của các combobox

        no_of_adults = self.no_of_adults_box.get()
        no_of_children = self.no_of_children_box.get()
        no_of_weekend_nights = self.no_of_weekend_nights_box.get()
        no_of_week_nights = self.no_of_week_nights_box.get()
        type_of_meal_plan = self.type_of_meal_plan_box.get()
        required_car_parking_space = self.required_car_parking_space_box.get()
        room_type_reserved = self.room_type_reserved_box.get()
        lead_time = self.lead_time_box.get()
        arrival_year = self.arrival_year_box.get()
        arrival_month = self.arrival_month_box.get()
        arrival_date = self.arrival_date_box.get()
        market_segment_type = self.market_segment_type_box.get()
        repeated_guest = self.repeated_guest_box.get()
        no_of_previous_cancellations = self.no_of_previous_cancellations_box.get()
        no_of_previous_bookings_not_canceled = self.no_of_previous_bookings_not_canceled_box.get()
        avg_price_per_room = self.avg_price_per_room_box.get()
        no_of_special_requests = self.no_of_special_requests_box.get()

        # Tạo dataframe chứa dữ liệu người dùng nhập
        input_data = pd.DataFrame({'no_of_adults': [no_of_adults],
                                   'no_of_children': [no_of_children],
                                   'no_of_weekend_nights': [no_of_weekend_nights],
                                   'no_of_week_nights': [no_of_week_nights],
                                   'type_of_meal_plan': [type_of_meal_plan],
                                   'required_car_parking_space': [required_car_parking_space],
                                   'room_type_reserved': [room_type_reserved],
                                   'lead_time': [lead_time],
                                   'arrival_year': [arrival_year],
                                   'arrival_month': [arrival_month],
                                   'arrival_date': [arrival_date],
                                   'market_segment_type': [market_segment_type],
                                   'repeated_guest': [repeated_guest],
                                   'no_of_previous_cancellations': [no_of_previous_cancellations],
                                   'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
                                   'avg_price_per_room': [avg_price_per_room],
                                   'no_of_special_requests': [no_of_special_requests]})

        # Chuyển đổi dữ liệu từ kiểu chuỗi sang kiểu số
        input_data = pre_processing(input_data)

        # Dự đoán lớp của điểm dữ liệu test_point
        self.Y_pred = self.clf.predict(input_data)
        if (self.Y_pred == 0):
            prediction = "Canceled"
        if self.Y_pred == 1:
            prediction = "Not canceled"
        # Hiển thị kết quả dự đoán lên giao diện
        self.prediction_label.config(text="Prediction: " + str(prediction))

    def accuracy(self):
        self.X_test = pre_processing(self.X_test)
        self.Y_test = self.Y_test.replace({'Not_Canceled': 1, 'Canceled': 0})
        y_pred = self.clf.predict(self.X_test)
        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test.iloc[i] == y_pred[i]:
                count += 1
        return count/len(self.Y_test)

def pre_processing(data):
    data['type_of_meal_plan'] = data['type_of_meal_plan'].replace(
        {'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3, 'Not Selected': 4})
    data['room_type_reserved'] = data['room_type_reserved'].replace(
        {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6,
         'Room_Type 7': 7})
    data['market_segment_type'] = data['market_segment_type'].replace(
        {'Online': 1, 'Offline': 2, 'Corporate': 3, 'Aviation': 4, 'Complementary': 5})
    return data

def get_unique_values(column_name, data):
    dt = data[column_name].unique()
    values = [value for value in dt.tolist()]
    values = sorted(values)
    return values

if __name__=="__main__":
    root = tk.Tk()
    # Lấy kích thước màn hình
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Thiết lập kích thước của cửa sổ
    root.geometry(f"{screen_width}x{screen_height}")

    gradient_boosting_gui = Gradient_Boosting_GUI(root)
    root.mainloop()

