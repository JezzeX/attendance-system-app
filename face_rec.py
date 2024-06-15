import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import time
from datetime import datetime
import os

r = redis.StrictRedis(
    host="redis-18012.c44.us-east-1-2.ec2.redns.redis-cloud.com",
    port=18012,
    password="kTzXqvPvJTyZCaqu2GY5ZGObnjhTJHJr",
)


def retrive_data(name):
    try:
        retrive_dict = r.hgetall(name)
        retrive_series = pd.Series(retrive_dict)
        retrive_series = retrive_series.apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        index = retrive_series.index
        index = list(map(lambda x: x.decode(), index))
        retrive_series.index = index
        retrive_df = retrive_series.to_frame().reset_index()
        retrive_df.columns = ["name", "facial_features"]
        retrive_df["name"] = retrive_df["name"].apply(lambda x: x.split("@")[0])
        return retrive_df[["name", "facial_features"]]
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame(columns=["name", "facial_features"])


faceapp = FaceAnalysis(
    name="buffalo_sc", root="insightface_model", providers=["CPUExecutionProvider"]
)
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


def ml_search_algorithm(
    dataframe, feature_column, test_vector, name_column="name", thresh=0.5
):
    dataframe = dataframe.copy()
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe["cosine"] = similar_arr
    data_filter = dataframe.query(f"cosine >= {thresh}")
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter["cosine"].argmax()
        person_name = data_filter.loc[argmax][name_column]
    else:
        person_name = "Unknown"
    return person_name


class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], current_time=[])

    def saveLogs_redis(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates("name", inplace=True)
        name_list = dataframe["name"].tolist()
        ctime_list = dataframe["current_time"].tolist()
        encoded_data = []
        for name, ctime in zip(name_list, ctime_list):
            if name != "Unknown":
                concat_string = f"{name}@{ctime}"
                encoded_data.append(concat_string)
        if len(encoded_data) > 0:
            r.lpush("attendance:logs", *encoded_data)
        self.reset_dict()

    def face_prediction(
        self, test_image, dataframe, feature_column, name_column="name", thresh=0.5
    ):
        current_time = str(datetime.now())
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        for res in results:
            x1, y1, x2, y2 = res["bbox"].astype(int)
            embeddings = res["embedding"]
            person_name = ml_search_algorithm(
                dataframe,
                feature_column,
                test_vector=embeddings,
                name_column=name_column,
                thresh=thresh,
            )
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
            cv2.putText(
                test_copy, person_name, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2
            )
            cv2.putText(
                test_copy,
                current_time,
                (x1, y2 + 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                2,
            )
            self.logs["name"].append(person_name)
            self.logs["current_time"].append(current_time)
            self.saveLogs_redis()  # Save logs immediately after recognizing a face
        return test_copy

    def process_uploaded_image(
        self, img_array, dataframe, feature_column, name_column="name", thresh=0.5
    ):
        current_time = str(datetime.now())
        results = faceapp.get(img_array)
        processed_img = img_array.copy()
        attendance_data = []
        for res in results:
            x1, y1, x2, y2 = res["bbox"].astype(int)
            embeddings = res["embedding"]
            person_name = ml_search_algorithm(
                dataframe,
                feature_column,
                test_vector=embeddings,
                name_column=name_column,
                thresh=thresh,
            )
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), color)
            cv2.putText(
                processed_img,
                person_name,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                processed_img,
                current_time,
                (x1, y2 + 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                2,
            )
            self.logs["name"].append(person_name)
            self.logs["current_time"].append(current_time)
            attendance_data.append({"name": person_name, "current_time": current_time})
        self.saveLogs_redis()  # Save logs immediately after processing an uploaded image
        return processed_img, attendance_data


class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res["bbox"].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            text = f"samples = {self.sample}"
            cv2.putText(
                frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2
            )
            embeddings = res["embedding"]
        return frame, embeddings

    def save_data_in_redis_db(self, name):
        if name and name.strip():
            key = f"{name}"
        else:
            return "name_false"

        if "face_embedding.txt" not in os.listdir():
            return "file_false"

        x_array = np.loadtxt("face_embedding.txt", dtype=np.float32)
        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        r.hset(name="academy:register", key=key, value=x_mean_bytes)
        os.remove("face_embedding.txt")
        self.reset()
        return True

    def delete_data_in_redis_db(self, name):
        if r.hdel("academy:register", name):
            return True
        return False
