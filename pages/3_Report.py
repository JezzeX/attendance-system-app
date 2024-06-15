import pandas as pd
import streamlit as st
from Home import face_rec
import datetime

st.subheader("Reporting")

name = "attendance:logs"


def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    return logs_list


def clear_logs(name):
    face_rec.r.delete(name)


tab1, tab2, tab3 = st.tabs(["Registered Data", "Logs", "Attendance Report"])

with tab1:
    if st.button("Refresh Data"):
        with st.spinner("Retrieving Data from Redis DB ..."):
            redis_face_db = face_rec.retrive_data(name="academy:register")
            if redis_face_db is not None:
                st.dataframe(redis_face_db[["name"]])
            else:
                st.write("No registered data found.")

with tab2:
    if st.button("Refresh Logs"):
        logs = load_logs(name=name)
        if logs:
            st.write(logs)
        else:
            st.write("No logs found.")

with tab3:
    st.subheader("Attendance Report")

    logs_list = load_logs(name=name)
    if logs_list:
        convert_byte_to_string = lambda x: x.decode("utf-8")
        logs_list_string = list(map(convert_byte_to_string, logs_list))
        split_string = lambda x: x.split("@")
        logs_nested_list = list(map(split_string, logs_list_string))

        # Adjusted the DataFrame creation to handle cases where logs don't match the expected structure
        logs_df = pd.DataFrame(logs_nested_list, columns=["Name", "Timestamp"])
        if "Role" in logs_df.columns:
            logs_df = logs_df[["Name", "Role", "Timestamp"]]
        else:
            logs_df["Role"] = None

        # Ensure that Timestamp column has no None values before applying split
        logs_df = logs_df.dropna(subset=["Timestamp"])
        logs_df["Timestamp"] = logs_df["Timestamp"].apply(
            lambda x: x.split(".")[0] if pd.notnull(x) else None
        )
        logs_df["Timestamp"] = pd.to_datetime(logs_df["Timestamp"])
        logs_df["Date"] = logs_df["Timestamp"].dt.date

        report_df = (
            logs_df.groupby(by=["Date", "Name"])
            .agg(
                In_time=pd.NamedAgg("Timestamp", "min"),
                Out_time=pd.NamedAgg("Timestamp", "max"),
            )
            .reset_index()
        )

        all_dates = report_df["Date"].unique()
        name_list = face_rec.retrive_data(name="academy:register")["name"].tolist()

        attendance_df = (
            pd.DataFrame(all_dates, columns=["Date"])
            .assign(key=1)
            .merge(pd.DataFrame(name_list, columns=["Name"]).assign(key=1), on="key")
            .drop("key", axis=1)
        )

        report_df = report_df.merge(
            attendance_df,
            on=["Date", "Name"],
            how="right",
            indicator=True,
        )

        report_df["Status"] = report_df["_merge"].apply(
            lambda x: 1 if x == "both" else 0
        )
        report_df = report_df.drop("_merge", axis=1)

        t1, t2 = st.tabs(["Complete Report", "Filter Report"])

        with t1:
            st.subheader("Complete Report")
            st.dataframe(report_df[["Name", "Date", "Status"]])

        with t2:
            st.subheader("Search Records")
            date_in = str(st.date_input("Filter Date", datetime.datetime.now().date()))
            name_in = st.selectbox("Select Name", ["ALL"] + name_list)
            if st.button("Submit"):
                filter_df = report_df.query(f'Date == "{date_in}"')
                if name_in != "ALL":
                    filter_df = filter_df.query(f'Name == "{name_in}"')
                st.dataframe(filter_df[["Name", "Date", "Status"]])
    else:
        st.write("No attendance logs found.")

if st.button("Clear Logs"):
    clear_logs(name=name)
    st.success("Logs cleared successfully.")
