# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
import random
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneOut

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
with open('../data/skempi_all_result_dict.pkl', 'rb') as f:
    skempi_all_result_dict = pickle.load(f)

with open('../data/ab_all_result_dict.pkl', 'rb') as f:
    ab_all_result_dict = pickle.load(f)

ab_name = list(ab_all_result_dict.keys())
skempi_name = list(skempi_all_result_dict.keys())

aa_string = 'ARNDCEQGHILKMFPSTWYV'
def encode_seq_int(data= 'HELLOWORLD'):
    # seq_letter = string.ascii_uppercase
    seq_letter = aa_string
    char_to_int = dict((c, i+1) for i, c in enumerate(seq_letter))
    integer_encoded = [char_to_int[char] for char in data]
    return integer_encoded

#original model####################################
def build_model(activation, latent_dim = 64, seq_num = 27, out_len = 5,f_num_1 = 64,
                        f_num_2 = 128,f_num_3 = 256,f_num_4 = 512,k_size = 5,drop_ratio = 0.2,
                        dense_1_num = 128,dense_2_num = 64,dense_3_num = 8,dense_4_num = 1):
    # build
    # input and embedding
    Input_pre = keras.Input(shape=(None,20), name='Input_pre') #@@@
    Input_aft = keras.Input(shape=(None,20), name='Input_aft') #@@@

    Input_rbd = keras.Input(shape=(None, 1), name='Input_rbd')
    Input_same = keras.Input(shape=(None, 1), name='Input_same')


    Input_x = keras.Input(shape=(None, 1), name='Input_x')
    Input_y = keras.Input(shape=(None, 1), name='Input_y')
    Input_z = keras.Input(shape=(None, 1), name='Input_z')

    diff_layer = layers.subtract([Input_pre, Input_aft])
    # lstm
    x_pre = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_pre')(Input_pre) #@@@

    # print('Input_aft.shape')
    # print(Input_aft.shape)
    # x_aft = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_aft')(Input_aft) #@@@

    x_diff = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_diff')(diff_layer)

    x_rbd = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_rbd')(Input_rbd)
    x_same = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_same')(Input_same)
    x_x = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_x')(Input_x)
    x_y = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_y')(Input_y)
    x_z = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_z')(Input_z)


    x_pre = layers.Reshape((-1, x_pre.shape[2], 1))(x_pre)
    # x_aft = layers.Reshape((-1, x_aft.shape[2], 1))(x_aft)
    ########################################################################
    x_diff = layers.Reshape((-1, x_diff.shape[2], 1))(x_diff)
    ########################################################################
    x_rbd = layers.Reshape((-1, x_rbd.shape[2], 1))(x_rbd)
    x_same = layers.Reshape((-1, x_same.shape[2], 1))(x_same)
    x_x = layers.Reshape((-1, x_x.shape[2], 1))(x_x)
    x_y = layers.Reshape((-1, x_y.shape[2], 1))(x_y)
    x_z = layers.Reshape((-1, x_z.shape[2], 1))(x_z)

    ########################################################################
    Concat_aux = layers.Concatenate(axis=3, name='Concat_aux')([x_rbd, x_same, x_x, x_y, x_z])  # (none, none, 128, 5)


    ###############################################################################
    # cnn
    def create_cnn_block(input, filter_num, kernel_size, drop_out=True):
        x = layers.Conv2D(filter_num, kernel_size, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPool2D(padding='same')(x)
        if drop_out:
            x = layers.Dropout(drop_ratio)(x)
        return x
    #################################################################################
    Concat_aux_conv = create_cnn_block(Concat_aux, f_num_1, k_size)
    Concat_aux_conv = create_cnn_block(Concat_aux_conv, f_num_2, k_size)
    Concat_x_pre_conv = create_cnn_block(x_pre, f_num_1, k_size)
    Concat_x_diff_conv = create_cnn_block(x_diff, f_num_1, k_size)

    Concat_x_pre_conv = create_cnn_block(Concat_x_pre_conv, f_num_2, k_size)
    Concat_x_diff_conv = create_cnn_block(Concat_x_diff_conv, f_num_2, k_size)
    Concat = layers.Concatenate(axis=3, name='Concat')([Concat_aux_conv, Concat_x_pre_conv, Concat_x_diff_conv])


    ##################################################################################
    Cnn_1 = create_cnn_block(Concat, f_num_1, k_size)
    Cnn_2 = create_cnn_block(Cnn_1, f_num_2, k_size)
    Cnn_3 = create_cnn_block(Cnn_2, f_num_3, k_size)
    Cnn_4 = create_cnn_block(Cnn_3, f_num_4, k_size, drop_out=False)
    MaxPooling = layers.GlobalMaxPooling2D()(Cnn_4)

    # dense
    Dense = layers.Dense(dense_1_num, activation='relu')(MaxPooling)
    Dense = layers.Dense(dense_2_num, activation='relu')(Dense)
    Dense = layers.Dense(dense_3_num)(Dense)
    # Dense = layers.BatchNormalization()(Dense)
    Dense = layers.LeakyReLU(alpha=0.2)(Dense)
    Dense = layers.Dense(dense_4_num)(Dense)

    if activation == 'sigmoid':
        # Dense = layers.BatchNormalization()(Dense)
        Dense = layers.Activation(keras.activations.sigmoid)(Dense)
    if activation == 'linear':
        # Dense = layers.BatchNormalization()(Dense)
        Dense = layers.Activation(keras.activations.linear)(Dense)


    model = keras.models.Model([Input_pre, Input_aft, Input_rbd, Input_same, Input_x, Input_y, Input_z], Dense)
    return model

model = build_model(activation='linear',f_num_1 = 32,f_num_2 = 64,f_num_3 = 128,f_num_4 = 256)
model.summary()

# %%
old_weights = model.get_weights()
def build_and_compile_binding_affinity_predictor(lr):
    model = build_model(activation='linear', f_num_1=32, f_num_2=64, f_num_3=128, f_num_4=256)
    model.set_weights(old_weights)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mae', metrics=['mae'])
    return model

ab_bind_name = ab_name.copy()
for i in ab_name:
    if ab_all_result_dict[i]['result']['subset_ddg'].size == 0:
        ab_bind_name.remove(i)

skempi_bind_name = skempi_name.copy()
for i in skempi_name:
    if skempi_all_result_dict[i]['result']['subset_ddg'].size == 0 or \
            '?' in skempi_all_result_dict[i]['result']['subset_pre_mutated_seq'] or \
            len(skempi_all_result_dict[i]['result']['subset_after_mutated_seq']) != len(skempi_all_result_dict[i]['result']['subset_ddg']):
        skempi_bind_name.remove(i)

# %%
a_list = []
b_list = []
c_list = []
d_list = []
e_list = []
x_list = []
y_list = []
z_list = []
for i in ab_bind_name:
    data = ab_all_result_dict[i]
    input_label = data['result']['subset_ddg']
    input_rbd_index = [data['rbd_index']]*len(input_label)
    input_pre_mutated_seq = [data['result']['subset_pre_mutated_seq']]*len(input_label)
    input_aft_mutated_seq = data['result']['subset_after_mutated_seq']
    input_chain_index = [data['result']['subset_chain_index']]*len(input_label)
    input_same_index = [data['result']['subset_same_index']]*len(input_label)
    input_coordinate = data['subset_alpha_carbon_coordinate']

    a = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_pre_mutated_seq])
    b = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_aft_mutated_seq])
    a = tf.one_hot(a, 20)
    b = tf.one_hot(b, 20)
    c = np.array([np.array(list(x)).reshape(len(x)) for x in input_rbd_index])
    d = np.array(list(map(lambda x: [int(same) for same in x], input_same_index)))
    x = np.array([input_coordinate[:, 0]]*len(input_label))
    y = np.array([input_coordinate[:, 1]]*len(input_label))
    z = np.array([input_coordinate[:, 2]]*len(input_label))
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    z = np.reshape(z, (z.shape[0], z.shape[1], 1))
    c = np.reshape(c, (c.shape[0], c.shape[1], 1))
    d = np.reshape(d, (d.shape[0], d.shape[1], 1))
    # normalize coordinate
    x = keras.utils.normalize(x, axis=1)
    y = keras.utils.normalize(y, axis=1)
    z = keras.utils.normalize(z, axis=1)
    e = np.array(list(np.array(input_label).reshape(len(input_label), 1)))
    a_list.append(a)
    b_list.append(b)
    c_list.append(c)
    d_list.append(d)
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
    e_list.append(e)

# skempi data
skempi_a_list = []
skempi_b_list = []
skempi_c_list = []
skempi_d_list = []
skempi_e_list = []
skempi_x_list = []
skempi_y_list = []
skempi_z_list = []
# i = skempi_bind_name[1]
for i in skempi_bind_name:
    data = skempi_all_result_dict[i]
    input_label = data['result']['subset_ddg']
    input_rbd_index = [data['rbd_index']]*len(input_label)
    input_pre_mutated_seq = [data['result']['subset_pre_mutated_seq']]*len(input_label)
    input_aft_mutated_seq = data['result']['subset_after_mutated_seq']
    input_chain_index = [data['result']['subset_chain_index']]*len(input_label)
    input_same_index = [data['result']['subset_same_index']]*len(input_label)
    input_coordinate = data['subset_alpha_carbon_coordinate']

    a = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_pre_mutated_seq])
    b = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_aft_mutated_seq])
    a = tf.one_hot(a, 20)
    b = tf.one_hot(b, 20)
    c = np.array([np.array(list(x)).reshape(len(x)) for x in input_rbd_index])
    d = np.array(list(map(lambda x: [int(same) for same in x], input_same_index)))
    x = np.array([input_coordinate[:, 0]]*len(input_label))
    y = np.array([input_coordinate[:, 1]]*len(input_label))
    z = np.array([input_coordinate[:, 2]]*len(input_label))
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    z = np.reshape(z, (z.shape[0], z.shape[1], 1))
    c = np.reshape(c, (c.shape[0], c.shape[1], 1))
    d = np.reshape(d, (d.shape[0], d.shape[1], 1))
    # normalize coordinate
    x = keras.utils.normalize(x, axis=1)
    y = keras.utils.normalize(y, axis=1)
    z = keras.utils.normalize(z, axis=1)
    e = np.array(list(np.array(input_label).reshape(len(input_label), 1)))
    skempi_a_list.append(a)
    skempi_b_list.append(b)
    skempi_c_list.append(c)
    skempi_d_list.append(d)
    skempi_x_list.append(x)
    skempi_y_list.append(y)
    skempi_z_list.append(z)
    skempi_e_list.append(e)

a_list = a_list + skempi_a_list
b_list = b_list + skempi_b_list
c_list = c_list + skempi_c_list
d_list = d_list + skempi_d_list
e_list = e_list + skempi_e_list
x_list = x_list + skempi_x_list
y_list = y_list + skempi_y_list
z_list = z_list + skempi_z_list

#%%
# ten-fold cross validation for binding affinity predictor
def get_batch_draw_sample_1(k, a_lst, b_lst, c_lst, d_lst, e_lst, x_lst, y_lst, z_lst):
    for _ in range(k):
        batch_n = len(a_lst)
        i = random.randint(0, batch_n - 1)
        sample_num = a_lst[i].shape[0]
        j = random.randint(0, sample_num - 1)
        print(i)
        print(j)

        a = tf.reshape(a_lst[i][j], [-1, tf.shape(a_lst[i][j])[0], tf.shape(a_lst[i][j])[1]])
        b = tf.reshape(b_lst[i][j], [-1, tf.shape(b_lst[i][j])[0], tf.shape(b_lst[i][j])[1]])
        c = c_lst[i][j].reshape(-1, c_lst[i][j].shape[0])
        d = d_lst[i][j].reshape(-1, d_lst[i][j].shape[0])
        x = x_lst[i][j].reshape(-1, x_lst[i][j].shape[0])
        y = y_lst[i][j].reshape(-1, y_lst[i][j].shape[0])
        z = z_lst[i][j].reshape(-1, z_lst[i][j].shape[0])
        e = e_lst[i][j]

        yield a, b, c, d, x, y, z, e

# Set the number of folds
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)

loo = LeaveOneOut()

# Initialize lists to store metrics across folds
all_train_loss = []
all_train_acc = []
all_test_loss = []
all_test_acc = []

# Combine the lists into a single dataset
data = list(zip(a_list, b_list, c_list, d_list, e_list, x_list, y_list, z_list))
N_EPOCHS = 1
random_num_binding_aff = random.randint(1, 100000)

# for fold, (train_idx, test_idx) in enumerate(kf.split(data)):  
for fold, (train_idx, test_idx) in enumerate(loo.split(data)):  
    print("=" * 50)
    print(f"Fold {fold + 1}")
    
    # Split the data into training and testing sets using KFold indices
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]

    # Unpack the training and testing sets
    a_train, b_train, c_train, d_train, e_train, x_train, y_train, z_train = zip(*train_data)
    a_test, b_test, c_test, d_test, e_test, x_test, y_test, z_test = zip(*test_data)

    zero = tf.zeros(1)
    random_num = random.randint(1, 100000)
    print('random_num')
    print(random_num)

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model = build_and_compile_binding_affinity_predictor(lr = 0.00001)
    
    train_loss = []
    train_acc = []
    csv_logger = tf.keras.callbacks.CSVLogger(
                'metric/loocv/{}-{}-{}-binding_affinity_predictor_lr-{}.log'.format(fold+1,random_num_binding_aff, random_num, i), separator=',', append=True)
    # for epoch in range(N_EPOCHS):
    print("=" * 50)
    print("training model")
    # print(epoch, "/", N_EPOCHS)

    for a, b, c, d, x, y, z, e in get_batch_draw_sample_1(5000, a_train, b_train, c_train, d_train, e_train, x_train, y_train, z_train):
        # print('e.shape')
        # print(e.shape)
        # print('zero.shape')
        # print(zero.shape)
        # print('true value')
        # print(e)
        # print('predicted value')
        # print(model.predict([a, b, c, d, x, y, z]))
        loss = abs(e - model.predict([a, b, c, d, x, y, z]))
        train_loss.append(loss)
        # print(loss)
        # print(train_loss)
        # print('avg')
        # print(sum(train_loss)/len(train_loss))
        train_acc.append(np.sign(e[0]) == np.sign(model.predict([a, b, c, d, x, y, z])[0][0]))
        print("train")
        model.fit([a, b, c, d, x, y, z], e, batch_size=1, epochs=N_EPOCHS, validation_data=([a, b, c, d, x, y, z], e), callbacks=[early_stopping, csv_logger], verbose=0)
        print("train_zero")
        model.fit([a, a, c, d, x, y, z], zero, batch_size=1, epochs=N_EPOCHS, validation_data=([a, b, c, d, x, y, z], zero), verbose=0)
            
    all_train_loss.append(train_loss)
    all_train_acc.append(train_acc)   

    test_loss = []
    test_acc = []
    true_e = []
    pred_val = []
    for a, b, c, d, x, y, z, e in get_batch_draw_sample_1(len(a_test), a_test, b_test, c_test, d_test, e_test, x_test, y_test, z_test):
        # print('Test')
        score = model.evaluate([a, b, c, d, x, y, z], e, batch_size=1)
        print('True value')
        print(e)
        print('Predicted value')
        predictions = model.predict([a, b, c, d, x, y, z])
        print(predictions)
        # Update your loss calculation accordingly
        loss = abs(e - predictions)
        test_loss.append(loss)
        test_acc.append(np.sign(e[0]) == np.sign(predictions[0][0]))
        true_e.append(e[0])
        pred_val.append(predictions[0][0])
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)

    print('avg_train_loss')
    print(sum(train_loss)/len(train_loss))
    avg_train = sum(train_loss)/len(train_loss)
    avg_acc_train = sum(train_acc)/len(train_acc)
    print('sample_num')
    print(len(train_loss))
    with open('metric/loocv/{}-{}-{}-binding_affinity_predictor_train_avg_loss_lr-{}.pkl'.format(fold+1, random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(avg_train, f)
    with open('metric/loocv/{}-{}-{}-binding_affinity_predictor_train_loss_lr-{}.pkl'.format(fold+1, random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(train_loss, f)


    print('avg_test_loss')
    print(sum(test_loss)/len(test_loss))
    print('sample_num')
    print(len(test_loss))
    avg_test = sum(test_loss)/len(test_loss)
    avg_acc_test = sum(test_acc) / len(test_acc)

    with open('metric/loocv/{}-{}-{}-binding_affinity_predictor_test_avg_loss_lr-{}.pkl'.format(fold+1, random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(avg_test, f)
    with open('metric/loocv/{}-{}-{}-binding_affinity_predictor_test_loss_lr-{}.pkl'.format(fold+1, random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(test_loss, f)

    model.save_weights('models/loocv/{}-{}-{}-binding_affinity_predictor_weights_lr-{}-train-{}-test-{}-train_acc-{}--test_acc-{}.h5'.format(fold+1, random_num_binding_aff,random_num,i, avg_train, avg_test, avg_acc_train, avg_acc_test))

    # # Compute R2 score
    # r2 = r2_score(true_e, pred_val)

    # # Plot R2 curve
    # plt.figure(figsize=(8, 8))
    # plt.scatter(true_e, pred_val, alpha=0.5)
    # plt.plot(true_e, pred_val, linestyle='--', color='red', linewidth=2)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('R2 Curve (R2 Score = {:.2f})'.format(r2))
    # # plt.show()
    # plt.savefig(f"./{fold + 1}_R2Curve.png")

# import csv
# all_lists = [all_train_loss, all_train_acc, all_test_loss, all_test_acc]
# print(all_lists)
# Write the lists to a CSV file
# with open('./result.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
    
#     # Write header
#     csv_writer.writerow(['Train loss', 'Train acc', 'Test loss', 'Test acc'])
    
#     # Write data
#     csv_writer.writerows(zip(*all_lists))

# print(f'Successfully saved data to ./result.csv')
