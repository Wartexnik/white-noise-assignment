"""White noise analysis module."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(images):
    """Normalize images."""
    if np.max(images)-np.min(images) != 0:
        images = (images-np.min(images))/(np.max(images)-np.min(images))
    return images


def rgb_to_gray(images):
    """Convert a 3-channel image array to a 1-channel image array."""
    images = np.array(0.2989 * images[:, :, :, 0]
                      + 0.5870 * images[:, :, :, 1]
                      + 0.1140 * images[:, :, :, 2])
    return images


def cnn1(shape):
    """First model architecture."""
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics='accuracy')
    return model


def cnn2(shape):
    """Second model architecture."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid',
                               input_shape=shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(128, (3, 3), activation='sigmoid'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    return model


def cnn_ab(shape):
    """Binary model architecture."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics='accuracy')
    return model


def cnn_alexnet(shape):
    """Implementation of AlexNet."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Resizing(227, 227, input_shape=shape),
        tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    return model


def plot_model_history(model_history):
    """Plot model training and validation accuracy."""
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validate'])
    plt.show()


def filter(images, labels, *args):
    """Choose images of classes listed in *args.

    Example:
    images: images of classes 0-9
    *args: 3, 5, 7
    output: images of new classes 0-2 (old classes 3, 5, 7)
    """
    keep_images = [label in args for label in labels]
    images, labels = images[keep_images], labels[keep_images]
    labels = np.array([args.index(label) for label in labels])
    return images, labels


def generate_noise(shape=(28, 28)):
    """Generate Gaussian noise."""
    return np.random.normal(0, 1, np.prod(shape)).reshape(shape)/6


def generate_noise_arr(size, shape=(28, 28)):
    """Generate noise array."""
    noise_arr = []
    for _ in range(size):
        noise_arr.append(generate_noise(shape))
    return np.array(noise_arr)


def generate_binary_classification_images(shape,
                                          training_images, training_labels,
                                          testing_images, testing_labels,
                                          epochs, class_a, class_b):
    """Train a model on classes 'a' and 'b',
       then generate classification images."""
    j = 0
    model_ab = cnn_ab(shape)
    # Keep two classes.
    training_images, training_labels = filter(training_images,
                                              training_labels,
                                              class_a, class_b)
    testing_images, testing_labels = filter(testing_images,
                                            testing_labels,
                                            class_a, class_b)

    model_ab.fit(training_images, training_labels,
                 validation_split=0.1, epochs=epochs)

    # Prepare a 4x8 figure for visualizing the results.
    _, ax = plt.subplots(4, 8, figsize=(20, 20))
    for axes in ax.ravel():
        axes.set_xticks([])
    for axes in ax.ravel():
        axes.set_yticks([])
    ax[0, 2].set_title(f'pred {class_a}, gt {class_a}')
    ax[0, 3].set_title(f'pred {class_a}, gt {class_b}')
    ax[0, 4].set_title(f'pred {class_b}, gt {class_a}')
    ax[0, 5].set_title(f'pred {class_b}, gt {class_b}')
    ax[0, 6].set_title(f'mean: {class_a}-{class_b}')
    ax[0, 7].set_title('noise only')

    # Calculate the results for different values of gamma.
    for gamma in [0.3, 0.2, 0.1, 0.0]:
        n_arr = np.zeros((4))
        image_arr = np.zeros((4, shape[0], shape[1]))
        noise_arr = np.zeros((4, shape[0], shape[1]))
        mean_ab = np.zeros((shape[0], shape[1]))
        noise_ab = np.zeros((shape[0], shape[1]))
        noise = (1-gamma)*generate_noise((shape[0], shape[1]))

        # Example images
        image_a = gamma*testing_images[testing_labels.tolist().index(0)]+noise
        image_b = gamma*testing_images[testing_labels.tolist().index(1)]+noise

        for i, _ in enumerate(testing_labels):
            noise = (1-gamma)*generate_noise((shape[0], shape[1]))
            stimulus = gamma*testing_images[i]+noise
            prediction = int(
                np.round(model_ab(np.expand_dims(stimulus, 0))[0]))
            index = int(str(testing_labels[i])+str(prediction), 2)
            image_arr[index] += stimulus
            noise_arr[index] += noise
            n_arr[index] += 1
        for i in range(4):
            if n_arr[i] != 0:
                image_arr[i] /= n_arr[i]
                noise_arr[i] /= n_arr[i]
            ax[j, i+2].imshow(image_arr[i])
            if i < 2:
                mean_ab += image_arr[i]
                noise_ab += noise_arr[i]
            else:
                mean_ab -= image_arr[i]
                noise_ab -= noise_arr[i]
        ax[j, 0].set_ylabel(f'g={gamma}')
        ax[j, 0].imshow(image_a)
        ax[j, 1].imshow(image_b)
        ax[j, 6].imshow(mean_ab)
        ax[j, 7].imshow(noise_ab)
        j += 1
    plt.show()
    return normalize(image_arr)


def generate_classification_images(model, color='gray', shape=(28, 28),
                                   mode='noise', testing_images=None,
                                   gamma=0.3, class_count=10):
    """Take a trained model and generate classification images."""
    if color == 'gray':
        image_arr = np.zeros((class_count, shape[0], shape[1]))
        noise_arr = np.zeros((class_count, shape[0], shape[1]))
    elif color == 'rgb':
        image_arr = np.zeros((class_count, shape[0], shape[1], shape[2]))
        noise_arr = np.zeros((class_count, shape[0], shape[1], shape[2]))
    n_arr = np.zeros((class_count))
    for i in range(10000):
        if color == 'gray':
            noise = generate_noise((shape[0], shape[1]))
        elif color == 'rgb':
            noise = generate_noise((shape[0], shape[1], shape[2]))

        # Classification images based on average noise or stimulus map
        if mode == 'noise':
            prediction = np.argmax(model(np.expand_dims(noise, 0)))
            noise_arr[prediction] += noise
        elif mode == 'stimulus':
            stimulus = gamma*testing_images[i] + (1-gamma)*noise
            prediction = np.argmax(model(np.expand_dims(stimulus, 0)))
            image_arr[prediction] += stimulus

        n_arr[prediction] += 1

    # Prepare a 2x5 figure.
    _, ax = plt.subplots(2, 5, figsize=(20, 20))
    for axes in ax.ravel():
        axes.set_xticks([])
    for axes in ax.ravel():
        axes.set_yticks([])
    for i in range(10):
        plot = ax[i//5, i % 5]
        plot.set_title(f'{i}')

        # Calculate and visualize the results.
        if mode == 'noise':
            if n_arr[i] != 0:
                noise_arr[i] /= n_arr[i]
            plot.imshow(noise_arr[i])
        elif mode == 'stimulus':
            if n_arr[i] != 0:
                image_arr[i] /= n_arr[i]
            plot.imshow(normalize(image_arr[i]))

    if mode == 'noise':
        return noise_arr
    elif mode == 'stimulus':
        return normalize(image_arr)


def test_classifier(classifier_images, testing_images, testing_labels):
    """Calculate the accuracy and confusion matrix of a classifier."""
    correct = 0
    predicted_labels = []
    for i, _ in enumerate(testing_images):
        # Test the classifier based on a smallest mean squared error
        # between the classifier and testing image pixel values.
        mse_arr = []
        for j, _ in enumerate(classifier_images):
            mse_arr.append(((classifier_images[j]
                            - testing_images[i])**2).mean(axis=None))
        predicted_labels.append(np.argmin(mse_arr))

    conf_matrix = tf.math.confusion_matrix(testing_labels,
                                           predicted_labels).numpy()
    for i in range(conf_matrix.shape[0]):
        correct += conf_matrix[i, i]
    accuracy = correct/len(testing_images)
    conf_matrix_weighted = np.round(conf_matrix.astype('float')
                                    / conf_matrix.sum(axis=1)[:, np.newaxis],
                                    2)
    return accuracy, conf_matrix_weighted


def sta(model, images, class_count):
    """Visualize kernel activations using spike-triggered average."""
    outputs = [layer.output for layer in model.layers]
    visualization_model = tf.keras.models.Model(inputs=model.input,
                                                outputs=outputs)
    kernel_layers_count = len([layer for layer
                               in visualization_model.layers[1:]
                               if len(layer.output_shape) == 4])
    display_grid_arr = [
        [0 for _ in range(kernel_layers_count)] for _ in range(class_count)]
    n_arr = [0 for _ in range(class_count)]
    layer_names = [layer.name for layer in model.layers]
    images = images[np.random.choice(len(images), 100)]
    for image in images:
        feature_maps = visualization_model(np.expand_dims(image, 0))
        prediction = np.argmax(model(np.expand_dims(image, 0)))
        k = 0
        for (_, feature_map) in zip(layer_names, feature_maps):
            # Choose only the neccessary layers (omit the dense layers).
            if len(feature_map.shape) == 4:
                n_features = feature_map.shape[-1]
                size = feature_map.shape[1]
                display_grid = np.zeros((size, size * n_features))
                for i in range(n_features):
                    # Prepare the kernel for visualization.
                    kernel = feature_map[0, :, :, i].numpy()
                    kernel -= kernel.mean()
                    if kernel.std() != 0:
                        kernel /= kernel.std()
                    kernel *= 64
                    kernel += 128
                    kernel = np.clip(kernel, 0, 255).astype('uint8')
                    display_grid[:, i * size: (i + 1) * size] = kernel
                display_grid_arr[prediction][k] += display_grid
                n_arr[prediction] += 1
                k += 1
    display_grid_arr = np.array(display_grid_arr)
    for i in range(class_count):
        for j in range(kernel_layers_count):
            try:
                if display_grid_arr[i, j].size != 0:
                    plt.figure(figsize=(20, 20/n_features))
                    plt.title(f'{i}, {layer_names[j]}')
                    plt.imshow(display_grid_arr[i, j],
                               aspect='auto', cmap='viridis')
            except AttributeError:
                pass


def transform_classifier(classifier_images, modifier):
    """Transforms classifier images to improve accuracy."""
    images = classifier_images ** modifier
    images[images < 0.07] = 0
    return normalize(images)


def transform_and_test(classifier_images, testing_images,
                       testing_labels, modifier):
    classifier_images = transform_classifier(classifier_images, modifier)
    acc, conf = test_classifier(
        classifier_images, testing_images, testing_labels)
    print(f'Accuracy: {acc}\nConfusion Matrix:\n{conf}')
