import pandas as pd
import json

bl_df = pd.read_csv('data/balanced_train_segments.csv', sep=', ')


bl_df.head()


def make_link(id):
    return f"https://www.youtube.com/watch?v={id}"



bl_df['link'] = bl_df.apply(lambda r: make_link(r['YTID']), axis=1)
bl_df.head()


class_labels = pd.read_csv('data/class_labels_indices.csv')
class_labels.head()



class_label_dict = {}
for _, row in class_labels.iterrows():
    class_label_dict[row['mid']] = row['display_name']



def get_labels_string(ids):
    ids = ids.replace("\"", '')
    ids = ids.split(',')
    names = [class_label_dict[x] for x in ids]
    return ','.join(names)


bl_df['class_names'] = bl_df.apply(lambda r: get_labels_string(r['positive_labels']), axis=1)
bl_df.head()


def make_file_path(rownum):
    return f'./data/audios/{rownum}.wav'

# bl_df['file_path'] = 

file_paths = []
for i, _ in bl_df.iterrows():
    file_paths.append(make_file_path(i))

bl_df['file_path'] = file_paths


bl_df.head()


from pytube import YouTube, Stream
from threading import Thread
# from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import Pool


def download_audio_file(link, filename):
    # thing: Stream = YouTube(link).streams.filter(only_audio=True, mime_type="audio/mp4").order_by('abr')[0]
    # thing.download('./tmp/', filename=filename)
    cmd = f"yt-dlp -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 -o ./tmp/{filename} {link}"
    os.system(cmd)


from multiprocessing import Queue, Process, Pool
# from queue import Queue
from uuid import uuid4
import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

done = False
process_q = Queue()
download_q = Queue()

for _, row in bl_df.iterrows():
    q_element = (row['link'], row['start_seconds'], row['end_seconds'], row['file_path'])

    download_q.put(q_element)

def process_file(*args):
    print("Running ffmpeg", flush=True)

    start, end, fname, tmpname = args

    s_hours = start // 360
    s_minutes = (start % 360) // 60
    s_seconds = start % 60

    command = f"ffmpeg -hide_banner -loglevel error -i ./tmp/{tmpname} -ac 1 -ar 44100 -ss {s_hours:0>2.0f}:{s_minutes:0>2.0f}:{s_seconds:0>2.0f} -t 00:00:10 {fname}"
    while not os.path.exists(f'./tmp/{tmpname}'):
        sleep(0.5)

    os.system(command)
    os.remove(f'./tmp/{tmpname}')

def download(dq, pq):
    print("Starting download thread", flush=True)
    while not dq.empty():
        try:
            link, start, end, file_path = dq.get()
        except Exception as e:
            continue

        tmp_file = f"{uuid4()}.mp3"

        print('Downloading ' + link, flush=True)

        try:
            download_audio_file(link, tmp_file)
            print(f'Downloaded {link} as {tmp_file}', flush=True)
            # pq.put((start, end, file_path, tmp_file))
            process_file(start, end, file_path, tmp_file)
        except:
            print(f'Failed to download {link}', flush=True)
            continue
    global done
    done = True

n_download_threads = 100

download_threads = []

with ThreadPoolExecutor(max_workers=n_download_threads) as executor:
    for _ in range(n_download_threads):
        executor.submit(download, download_q, process_q)

    executor.shutdown(wait=True)

for t in download_threads:
    t.join()
