import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from model_respi import Sleep_NET
import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from utils import PrintScore, ConfusionMatrix, GetClassWeight, PrintAHIResults
from sklearn import metrics
import warnings
from datetime import datetime
import shutil

os.environ["KERAS_BACKEND"] = "tensorflow"

# Command settings
parser = argparse.ArgumentParser(description='Respiratory SleepStaging')
parser.add_argument('--model', type=str, default='model_respi')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='mixed_data')
parser.add_argument('--n_class', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--train_rule', default='None', type=str, help='ClassBalance ReWeight SqrtReWeight None')
parser.add_argument('--n_kernel', type=int, default=9)
parser.add_argument('--ahi_scale', type=int, default=500)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--domain_scaler', type=int, default=200)
args = parser.parse_args()

# Mixed Precision Calculation
tf.keras.mixed_precision.set_global_policy('mixed_float16')

if __name__ == '__main__':
    # Setting Global Seeds
    seed = args.seed
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    print(f'DataSet: {args.dataset}')
    data_path = '../dataloader/' + args.dataset + '_' + str(args.n_class)
    
    """ 0. Creat Distribute Strategy """
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = args.batch_size

    """ 1. Setup input pipeline """
    # load batch dataset and distribute to GPUS
    # load train-data
    train_ds_split1 = tf.data.experimental.load(os.path.join(data_path, "train_ds_1"))
    train_ds_split1 = train_ds_split1.unbatch()
    train_ds_split1 = train_ds_split1.batch(args.batch_size, drop_remainder=False)
    train_ds_split1 = strategy.experimental_distribute_dataset(train_ds_split1)

    train_ds_split2 = tf.data.experimental.load(os.path.join(data_path, "train_ds_2"))
    train_ds_split2 = train_ds_split2.unbatch()
    train_ds_split2 = train_ds_split2.batch(args.batch_size, drop_remainder=False)
    train_ds_split2 = strategy.experimental_distribute_dataset(train_ds_split2)

    # load val-data
    val_ds = tf.data.experimental.load(os.path.join(data_path, "val_ds"))
    val_ds = val_ds.unbatch()
    val_ds = val_ds.batch(args.batch_size, drop_remainder=False)
    val_ds = strategy.experimental_distribute_dataset(val_ds)
    
    cls_num_train = np.load(os.path.join(data_path, "num_list.npy"))
    
     # set folder path
    time_start = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    folder_path = './model_checkpoints/' + time_start + '_' + args.dataset + '_' + args.model + '_' + str(args.n_class)
    
    # save settings 
    os.makedirs(folder_path)
    argsDict = args.__dict__
    with open(folder_path + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    # save model raw file
    shutil.copy('./' + args.model + '.py', os.path.join(folder_path, 'model_backup.py'))

    # save training raw file
    shutil.copy('./main_training_ahi.py', os.path.join(folder_path, 'main_training_ahi_backup.py'))


    # Get sleep class weight
    per_cls_weights = GetClassWeight(args.train_rule, args.n_class, cls_num_train)
    print('Per Class Weight: {}'.format(per_cls_weights))
    per_cls_weights = tf.constant(per_cls_weights, dtype=tf.float32)
    
    """ 2. Model Configuration """
    # Definition of loss function
    with strategy.scope():
        loss_object_sleep = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss_object_apnea = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss_object_domain = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_apnea_loss(labels, predictions):
            per_example_loss = loss_object_apnea(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        
        def compute_domain_loss(labels, predictions):
            per_example_loss = loss_object_domain(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def compute_sleep_loss(labels, predictions):
            per_example_loss = loss_object_sleep(labels, predictions)
            label_indices = tf.argmax(labels, axis=1)
            weights = tf.gather(per_cls_weights, label_indices)
            weighted_loss = per_example_loss * weights
            return tf.nn.compute_average_loss(weighted_loss, global_batch_size=labels.shape[0])

    # Definition of metrics
    with strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_loss_sleep = tf.keras.metrics.Mean(name='train_loss_sleep')
        train_accuracy_sleep = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_sleep')
        test_loss_sleep = tf.keras.metrics.Mean(name='test_loss_sleep')
        test_accuracy_sleep = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_sleep')

        train_loss_ahi = tf.keras.metrics.Mean(name='train_loss_ahi')
        test_loss_ahi = tf.keras.metrics.Mean(name='test_loss_ahi')
        
        train_loss_domain = tf.keras.metrics.Mean(name='train_loss_domain')
        test_loss_domain = tf.keras.metrics.Mean(name='test_loss_domain')

    # Definition of model
    with strategy.scope():
        # Create model
        model = Sleep_NET(n_kernel=args.n_kernel)
        # Optimizer and Loss Function Selection for Training
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


    """ 3. Training Process"""
    def train_step(inputs):
        datas, sleep_labels, is_data, ahi_labels, domain_labels = inputs
        # ahi_labels = replace_nan_with_constant(ahi_labels)
        with tf.GradientTape() as tape:
            # model prediction
            sleep_predictions, ahi_predictions, domain_predictions = model(datas, training=True)
            # data reshape
            sleep_predictions = tf.reshape(sleep_predictions, (datas.shape[0] * 1200, args.n_class))  #
            ahi_predictions = tf.reshape(ahi_predictions, (datas.shape[0], 1))  #
            sleep_labels = tf.reshape(sleep_labels, (datas.shape[0] * 1200, args.n_class))
            ahi_labels = ahi_labels[..., tf.newaxis]
            is_data = tf.reshape(is_data, (datas.shape[0] * 1200,))
            # delete padding data
            active_seg = tf.where(is_data)[:, 0]
            sleep_labels = tf.gather(sleep_labels, indices=active_seg)
            sleep_predictions = tf.gather(sleep_predictions, indices=active_seg)
            # compute loss
            loss_sleep = compute_sleep_loss(sleep_labels, sleep_predictions)
            loss_ahi = compute_apnea_loss(ahi_labels, ahi_predictions)
            loss_domain = compute_domain_loss(domain_labels, domain_predictions)
            regularization_losses = tf.add_n(model.losses)
            
            loss = (
                loss_sleep 
                + loss_ahi / args.ahi_scale 
                - loss_domain / args.domain_scaler
                # + regularization_losses
            )
            # loss = loss_sleep + loss_ahi / args.ahi_loss_scale_factor
            
            scaled_loss = optimizer.get_scaled_loss(loss)

        # Mixed Precision Calculation
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_sleep(loss_sleep)
        train_loss_ahi(loss_ahi)
        train_loss_domain(loss_domain)
        train_accuracy_sleep(sleep_labels, sleep_predictions)
        return loss

    def test_step(inputs):
        datas, sleep_labels, is_data, ahi_labels, domain_labels = inputs
        # model prediction
        sleep_predictions, ahi_predictions, domain_predictions = model(datas, training=False)
        # data reshape
        sleep_predictions = tf.reshape(sleep_predictions, (datas.shape[0] * 1200, args.n_class))
        ahi_predictions = tf.reshape(ahi_predictions, (datas.shape[0], 1))
        sleep_labels = tf.reshape(sleep_labels, (datas.shape[0] * 1200, args.n_class))
        ahi_labels = ahi_labels[..., tf.newaxis]
        is_data = tf.reshape(is_data, (datas.shape[0] * 1200,))
        # delete padding data
        active_seg = tf.where(is_data)[:, 0]
        sleep_labels = tf.gather(sleep_labels, indices=active_seg)
        sleep_predictions = tf.gather(sleep_predictions, indices=active_seg)
        # compute loss
        t_loss_sleep = compute_sleep_loss(sleep_labels, sleep_predictions)
        t_loss_ahi = compute_apnea_loss(ahi_labels, ahi_predictions)
        t_loss_domain = compute_domain_loss(domain_labels, domain_predictions)
        t_regularization_losses = tf.add_n(model.losses)
        
        t_loss = (
            t_loss_sleep 
            + t_loss_ahi / args.ahi_scale
        )
        # t_loss = t_loss_sleep + t_loss_ahi / args.ahi_loss_scale_factor

        test_loss(t_loss)
        test_loss_sleep(t_loss_sleep)
        test_loss_ahi(t_loss_ahi)
        test_loss_domain(t_loss_domain)
        test_accuracy_sleep(sleep_labels, sleep_predictions)
        
    # `run` replicates the provided computation and runs it with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs,))


    EPOCHS = args.n_epoch

    log = []
    best_acc = 0
    best_loss = 1000
    early_stop = 0

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = folder_path + '/logs/gradient_tape/' + current_time + '/train'
    test_log_dir = folder_path + '/logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir) 
    test_summary_writer = tf.summary.create_file_writer(test_log_dir) 

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        test_loss.reset_states()

        train_loss_sleep.reset_states()
        train_accuracy_sleep.reset_states()
        test_loss_sleep.reset_states()
        test_accuracy_sleep.reset_states()

        train_loss_ahi.reset_states()
        test_loss_ahi.reset_states()
        
        train_loss_domain.reset_states()
        test_loss_domain.reset_states()

        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for inputs_split1 in train_ds_split1:
            total_loss += distributed_train_step(inputs_split1)
            num_batches += 1
            
        for inputs_split2 in train_ds_split2:
            total_loss += distributed_train_step(inputs_split2)
            num_batches += 1
        train_loss = total_loss / num_batches
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('sleep_loss', train_loss_sleep.result(), step=epoch)
            tf.summary.scalar('ahi_loss', train_loss_ahi.result(), step=epoch)
            tf.summary.scalar('sleep_accuracy', train_accuracy_sleep.result(), step=epoch)       

        # TEST LOOP
        for test_inputs in val_ds:
            distributed_test_step(test_inputs)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('sleep_loss', test_loss_sleep.result(), step=epoch)
            tf.summary.scalar('ahi_loss', test_loss_ahi.result(), step=epoch)
            tf.summary.scalar('sleep_accuracy', test_accuracy_sleep.result(), step=epoch)

        log.append([
            epoch, train_loss.numpy(), train_loss_ahi.result().numpy(), train_loss_domain.result().numpy(),
            train_loss_sleep.result().numpy(), train_accuracy_sleep.result().numpy(),
            test_loss.result().numpy(), test_loss_ahi.result().numpy(), test_loss_domain.result().numpy(),
            test_loss_sleep.result().numpy(), test_accuracy_sleep.result().numpy()])
        pd.DataFrame.from_dict(log).to_csv(folder_path + '/train_log.csv', header=[
            'Epoch', 'Train_loss', 'Train_loss_ahi', 'Train_loss_domain',
            'Train_loss_sleep', 'Train_acc_sleep',
            'Val_loss', 'Val_loss_ahi', 'Val_loss_domain',
            'Val_loss_sleep', 'Val_acc_sleep'])
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(
            f'[{current_time}] '
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss:.4f}, '
            f'LossAHI: {train_loss_ahi.result():.4f}, '
            f'LossDomain: {train_loss_domain.result():.4f}, '
            f'LossSleep: {train_loss_sleep.result():.4f}, '
            f'AccuracySleep: {train_accuracy_sleep.result() * 100:.4f}, '
            f'Test Loss: {test_loss.result():.4f}, '
            f'Test LossAHI: {test_loss_ahi.result():.4f}, '
            f'Test LossDomain: {test_loss_domain.result():.4f}, '
            f'Test LossSleep: {test_loss_sleep.result():.4f}, '
            f'Test AccuracySleep: {test_accuracy_sleep.result() * 100:.4f}, '
        )
        early_stop += 1

        # # save best loss
        if best_loss > test_loss.result():
            early_stop = 0
            best_loss = test_loss.result()
            model.save_weights(folder_path + '/best_model/best_model')

        if early_stop >= args.early_stop:
            break
    del model, train_ds_split1, train_ds_split2, val_ds

    """ 4. Testing Process"""
    test_ds = tf.data.experimental.load(os.path.join(data_path, "test_ds"))
    test_model = Sleep_NET(n_kernel=args.n_kernel)
    test_model.load_weights(folder_path + '/best_model/best_model')
    fold = 0
    for test_inputs in test_ds:
        test_datas, test_labels, is_test_datas, ahi_labels, domain_labels = test_inputs
        # model prediction
        predictions, ahi_predictions, _ = test_model(test_datas, training=False)
        # data reshape
        predictions = tf.reshape(predictions, (test_datas.shape[0] * 1200, args.n_class))  
        ahi_predictions = tf.reshape(ahi_predictions, (test_datas.shape[0], ))
        test_labels = tf.reshape(test_labels, (test_datas.shape[0] * 1200, args.n_class))
        is_test_datas = tf.reshape(is_test_datas, (test_datas.shape[0] * 1200,))
        # delete padding data
        active_seg = tf.where(is_test_datas)[:, 0]
        test_labels = tf.gather(test_labels, indices=active_seg)
        predictions = tf.gather(predictions, indices=active_seg)
        # predictions collection
        AllTrue_temp = np.argmax(test_labels, axis=-1)
        AllPred_temp = np.argmax(predictions, axis=-1)
        AHITrue_temp = np.copy(ahi_labels)
        AHIPred_temp = np.copy(ahi_predictions)

        if fold == 0:
            AllPred = AllPred_temp
            AllTrue = AllTrue_temp
            AHIPred = AHIPred_temp
            AHITrue = AHITrue_temp
        else:
            AllPred = np.concatenate((AllPred, AllPred_temp))
            AllTrue = np.concatenate((AllTrue, AllTrue_temp))
            AHIPred = np.concatenate((AHIPred, AHIPred_temp))
            AHITrue = np.concatenate((AHITrue, AHITrue_temp))
        fold += 1
    del test_model

    # Print score to console
    print(128 * '=')
    PrintScore(AllTrue, AllPred, num_classes=args.n_class)

    # Print score to Result.txt file
    PrintScore(AllTrue, AllPred, num_classes=args.n_class, savePath='./' + folder_path + '/')

    # Save Confusion Matrix
    classes = ['Awake', 'Light', 'Deep', 'REM']
    cm_numpy = metrics.confusion_matrix(AllTrue, AllPred, labels=list(range(args.n_class)))
    ConfusionMatrix(cm_numpy, classes=classes, title="Confusion_Matrix", savePath=folder_path)

    # Save AHI Results
    AHIPred = AHIPred.astype(np.float32)
    AHITrue = AHITrue.astype(np.float32)
    PrintAHIResults(AHITrue, AHIPred, './' + folder_path + '/')