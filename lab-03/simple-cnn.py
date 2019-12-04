import os
import random
import shutil
import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from sklearn import metrics
import numpy as np

random.seed(0)

base = '/Users/JuanFelipe/Documents/Universidad/10 Semestre - Intercambio/Applied Machine Learning/Labs/lab-03/'
original_dataset_dir = os.path.join(base, 'flowers')
dataset = os.path.join(base, 'flowers_split')

train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'validation')
test_dir = os.path.join(dataset, 'test')

categories = os.listdir(original_dataset_dir)
categories = [category for category in categories if not category.startswith('.')]
print('Image types:', categories)
data_folders = [os.path.join(original_dataset_dir, category) for category in categories]

pairs = []
for folder, category in zip(data_folders, categories):
    images = os.listdir(folder)
    images = [image for image in images if not image.startswith('.')]
    pairs.extend([(image, category) for image in images])

random.shuffle(pairs)
number_of_images = len(pairs)
train_imgs = pairs[:int(number_of_images*0.6)]
validation_imgs = pairs[int(number_of_images*0.6):int(number_of_images*0.8)]
test_imgs = pairs[int(number_of_images*0.8):]

num_of_train_samples = len(train_imgs)
num_of_val_samples = len(validation_imgs)
num_of_test_samples = len(test_imgs)

print(num_of_train_samples)
print(num_of_val_samples)
print(num_of_test_samples)

for img, label in train_imgs:
    src = os.path.join(original_dataset_dir, label, img)
    dst = os.path.join(train_dir, label, img)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

for img, label in validation_imgs:
    src = os.path.join(original_dataset_dir, label, img)
    dst = os.path.join(validation_dir, label, img)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

for img, label in test_imgs:
    src = os.path.join(original_dataset_dir, label, img)
    dst = os.path.join(test_dir, label, img)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

print('Number of training images: ', len(os.listdir(train_dir + '/daisy')) + len(os.listdir(train_dir + '/dandelion')) + len(os.listdir(train_dir + '/rose')) + len(os.listdir(train_dir + '/tulip')) + len(os.listdir(train_dir + '/sunflower')))
print('Number of val images: ', len(os.listdir(validation_dir + '/daisy')) + len(os.listdir(validation_dir + '/dandelion')) + len(os.listdir(validation_dir + '/rose')) + len(os.listdir(validation_dir + '/tulip')) + len(os.listdir(validation_dir + '/sunflower')))
print('Number of test images: ', len(os.listdir(test_dir + '/daisy')) + len(os.listdir(test_dir + '/dandelion')) + len(os.listdir(test_dir + '/rose')) + len(os.listdir(test_dir + '/tulip')) + len(os.listdir(test_dir + '/sunflower')))

# ---------------------------------------------------------------------------------------------------------------------

print('Building a simple convolutional Neural Network')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical', shuffle=False)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(train_generator, steps_per_epoch=num_of_train_samples//batch_size, epochs=30, validation_data=validation_generator, validation_steps=num_of_val_samples//batch_size)

model.save('simple-cnn-lab03.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

Y_pred = model.predict_generator(test_generator, num_of_test_samples//batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(metrics.confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
print(metrics.classification_report(test_generator.classes, y_pred, target_names=categories))

# --------------------------------------------------------------------------------------------------------------------

print('Using image augmentation')

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(5, activation='sigmoid'))

model2.summary()

model2.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen2 = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen2 = ImageDataGenerator(rescale=1./255)
test_datagen2 = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator2 = train_datagen2.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
validation_generator2 = validation_datagen2.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
test_generator2 = test_datagen2.flow_from_directory(test_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical', shuffle=False)

for data_batch, labels_batch in train_generator2:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history2 = model2.fit_generator(train_generator2, steps_per_epoch=num_of_train_samples//batch_size, epochs=100, validation_data=validation_generator2, validation_steps=num_of_val_samples//batch_size)

model2.save('simple-cnn-lab03-data-aug.h5')

acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

Y_pred = model2.predict_generator(test_generator2, num_of_test_samples//batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(metrics.confusion_matrix(test_generator2.classes, y_pred))
print('Classification Report')
print(metrics.classification_report(test_generator2.classes, y_pred, target_names=categories))

# --------------------------------------------------------------------------------------------------------------------

print('Using a pretrained convolutional base')

conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base.summary()

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count, 5))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, num_of_train_samples)
validation_features, validation_labels = extract_features(validation_dir, num_of_val_samples)
test_features, test_labels = extract_features(test_dir, num_of_test_samples)

train_features = np.reshape(train_features, (num_of_train_samples, 3 * 3 * 2048))
validation_features = np.reshape(validation_features, (num_of_val_samples, 3 * 3 * 2048))
test_features = np.reshape(test_features, (num_of_test_samples, 3 * 3 * 2048))

model3 = models.Sequential()

model3 = models.Sequential()
model3.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 2048))
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(5, activation='softmax'))

model3.summary()

model3.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['acc'])

history3 = model3.fit(train_features, train_labels, epochs=30, batch_size=32, validation_data=(validation_features, validation_labels))

model3.save('simple-cnn-lab03-pretrained.h5')

acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

Y_pred = model3.predict(test_features,  num_of_test_samples//32)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(metrics.confusion_matrix(np.argmax(test_labels, axis=1), y_pred))
print('Classification Report')
print(metrics.classification_report(np.argmax(test_labels, axis=1), y_pred, target_names=categories))

# --------------------------------------------------------------------------------------------------------------------

print('Using a pretrained convolutional base with image transformer')

conv_base2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


model4 = models.Sequential()
model4.add(conv_base2)
model4.add(layers.Flatten())
model4.add(layers.Dense(256, activation='relu'))
model4.add(layers.Dense(5, activation='softmax'))

conv_base2.trainable = False

model4.summary()

batch_size = 32

train_datagen4 = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
validation_datagen4 = ImageDataGenerator(rescale=1./255)
test_datagen4 = ImageDataGenerator(rescale=1./255)

train_generator4 = train_datagen4.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
validation_generator4 = validation_datagen4.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
test_generator4 = test_datagen4.flow_from_directory(test_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical', shuffle=False)

model4.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = model4.fit_generator(train_generator4, steps_per_epoch=num_of_train_samples//batch_size, epochs=30, validation_data=validation_generator4, validation_steps=num_of_val_samples//batch_size, verbose=2)

model4.save('simple-cnn-lab03-pretrained-transformer.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

Y_pred = model4.predict_generator(test_generator4, num_of_test_samples//batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(metrics.confusion_matrix(test_generator4.classes, y_pred))
print('Classification Report')
print(metrics.classification_report(test_generator4.classes, y_pred, target_names=categories))
