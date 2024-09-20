from tensorflow.keras.applications import EfficientNetB0, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Paths
data_dir = 'C:/Users/lenovo/OneDrive/Desktop/skinsense/data/'
save_model_path = 'model/stacking.keras'

# Model Parameters
batch_size = 32
epochs = 10
img_size = (224, 224)  # Use the common size for both models

# Prepare Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load EfficientNetB0 Model
base_model_eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x_eff = base_model_eff.output
x_eff = GlobalAveragePooling2D()(x_eff)

# Load InceptionV3 Model
base_model_inc = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x_inc = base_model_inc.output
x_inc = GlobalAveragePooling2D()(x_inc)

# Concatenate the outputs from both models
combined = concatenate([x_eff, x_inc])

# Add new layers
x = Dense(1024, activation='relu')(combined)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Define Model
model = Model(inputs=[base_model_eff.input, base_model_inc.input], outputs=predictions)

# Freeze the base model layers
for layer in base_model_eff.layers:
    layer.trainable = False
for layer in base_model_inc.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    save_model_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Train the model
model.fit(
    [train_generator, train_generator],  # Adjust for how you provide input
    epochs=epochs,
    validation_data=([validation_generator, validation_generator]),
    callbacks=[checkpoint]
)

# After training, the best model will be saved at save_model_path
