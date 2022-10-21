import pandas as pd
from multiprocessing import Pool, Queue, cpu_count
from uuid import uuid4
import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor


def make_link(id):
    return f"https://www.youtube.com/watch?v={id}"


def get_dataframe():
    file_paths = []
    class_labels = pd.read_csv("data/class_labels_indices.csv")
    bl_df = pd.read_csv("data/balanced_train_segments.csv", sep=", ")
    class_label_dict = {}

    def get_labels_string(ids):
        ids = ids.replace('"', "")
        ids = ids.split(",")
        names = [class_label_dict[x] for x in ids]
        return ",".join(names)

    def make_file_path(rownum):
        return f"./data/audios/{rownum}.wav"

    bl_df["link"] = bl_df.apply(lambda r: make_link(r["YTID"]), axis=1)
    for _, row in class_labels.iterrows():
        class_label_dict[row["mid"]] = row["display_name"]
    bl_df["class_names"] = bl_df.apply(
        lambda r: get_labels_string(r["positive_labels"]), axis=1
    )
    for i, _ in bl_df.iterrows():
        file_paths.append(make_file_path(i))
    bl_df["file_path"] = file_paths
    return bl_df


def download_audio_file(link, filename):
    cmd = f"./yt-dlp -q -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 -o ./tmp/{filename} {link}"
    os.system(cmd)

def process_file(*args):
    print("Running ffmpeg", flush=True)

    start, end, fname, tmpname = args

    s_hours = start // 360
    s_minutes = (start % 360) // 60
    s_seconds = start % 60

    command = f"ffmpeg -hide_banner -loglevel error -i ./tmp/{tmpname} -ac 1 -ar 44100 -ss {s_hours:0>2.0f}:{s_minutes:0>2.0f}:{s_seconds:0>2.0f} -t 00:00:10 {fname}"

    counter = 0
    while not os.path.exists(f"./tmp/{tmpname}"):
        counter += 1
        if counter > 20:
            print("????")
            return
        sleep(0.1)

    os.system(command)
    os.remove(f"./tmp/{tmpname}")

def inner_dl(link, tmp_file, start, end, file_path):
    try:
        download_audio_file(link, tmp_file)
        process_file(start, end, file_path, tmp_file)
    except Exception as e:
        print(f"Failed to download {link}", flush=True)

def download(dq, pq):
    print("Starting download thread", flush=True)
    with Pool(4) as p:
        while not dq.empty():
            try:
                link, start, end, file_path = dq.get()
            except:
                sleep(0.1)
                continue

            tmp_file = f"{uuid4()}.mp3"

            print("Downloading " + link)

            p.apply(inner_dl, args=(link, tmp_file, start, end, file_path))
    global done
    done = True





if __name__ == "__main__":
    done = False
    process_q = Queue()
    download_q = Queue()
    bl_df = get_dataframe()

    for _, row in bl_df.iterrows():
        if os.path.exists(row["file_path"]):
            continue

        q_element = (
            row["link"],
            row["start_seconds"],
            row["end_seconds"],
            row["file_path"],
        )

        download_q.put(q_element)

    n_download_threads = cpu_count() * 2 # Running processes will be n_download_threads * 4

    download_threads = []

    with ThreadPoolExecutor(max_workers=n_download_threads) as executor:
        for _ in range(n_download_threads):
            executor.submit(download, download_q, process_q)

        executor.shutdown(wait=True)