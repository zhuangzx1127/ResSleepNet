import importlib
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
import os
import tensorflow as tf
from scipy.io import loadmat
from matplotlib import pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"

def sleeplabel_transform(sleep_label):
    sleep_stage = []
    for i in range(len(sleep_label)):
        if sleep_label[i] == 0:
            sleep_stage.append("Wake")
        elif sleep_label[i] == 1:
            sleep_stage.append("Light")
        elif sleep_label[i] == 2:
            sleep_stage.append("Deep")
        elif sleep_label[i] == 3:
            sleep_stage.append("REM")
        else:
            return False
    return sleep_stage

tf.keras.mixed_precision.set_global_policy('mixed_float16')

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    data_path = 'data/20241119-3_respiratory.mat'

    # Load Respiratory Signal
    tmp_data = loadmat(data_path)
    is_data = tmp_data['is_data']
    signal_respiratory = tmp_data['signal_respiratory']
    
    # Load Pretrained Model
    model_path = 'pretrained_model'
    module_path = os.path.join(model_path, 'model_backup.py')
    spec = importlib.util.spec_from_file_location("model_backup", module_path)
    model_backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_backup)
    Sleep_NET = model_backup.Sleep_NET
    model = Sleep_NET(n_kernel=9)
    model.load_weights(model_path + '/best_model/best_model')
    signal_output = signal_respiratory[np.newaxis, :, :]
    predictions_sleep, predictions_ahi, _ = model(signal_output, training=False)
    sleep_score = np.squeeze(predictions_sleep)
    sleep_stage_perd = np.argmax(sleep_score, axis=-1)
    print(f"predicted AHI: : {predictions_ahi.numpy()[0][0]:.2f}")

    plt.rc('font', family='Times New Roman')
    n = np.arange(len(sleep_stage_perd))
    plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(2, 1, 1)
    plt.plot(n, sleep_stage_perd, linewidth=2.0)
    plt.ylabel('Pred Label', fontsize=20)
    plt.axis([0, max(n), -1, 4])
    plt.grid(axis='both')
    plt.yticks([0, 1, 2, 3], ['Wake', 'Light', 'Deep', 'REM'])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax2 = plt.subplot(2, 1, 2)
    colors = ['steelblue', 'c', 'peachpuff', 'firebrick']
    possibility_data = np.sum(sleep_score, axis=1)
    for iii in range(4):
        plt.fill_between(n, possibility_data, color=colors[iii])
        possibility_data = possibility_data - sleep_score[:, 3 - iii]
    plt.axis([0, max(n), 0, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['REM', 'Deep', 'Light', 'Wake'], loc='best', ncol=4, fontsize=15)
    plt.xlabel('Epoch num', fontsize=20)
    plt.ylabel('probability',fontsize=20)
    plt.tight_layout()
    plt.savefig('figure/20241119.png', dpi=300)
    plt.show()