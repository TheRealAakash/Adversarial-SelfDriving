import load_datasets

config = {
    "models_dir": "models",
    "DATASET": "CIFAR10",
    "target_learning_rate": 0.001,
    "target_model_epochs": 50,
    "target_batch_size": 64,
    "model_name": "CIFAR10-KerasModel.model",

    "LOAD_TARGET_WEIGHTS": False,
    "AdvGAN_epochs": 60,
    "AdvGAN_batch_size": 128,
    "AdvGAN_Generator_LR": 0.0002,
    "AdvGAN_Discriminator_LR": 0.001,  # 0.01

    "generator_model_name": "CIFAR10-GeneratorModel.model",
    "discriminator_model_name": "CIFAR10-DiscriminatorModel.model",
    "load_generator": False,
    "load_discriminator": False,
    "images_name": "CIFAR10Generated",

    "c": 0.3,

    "targeted": True,
    "targets": {
        8: 4
    },
    "target": 8,

    "IMG_W": 32,
    "IMG_H": 32,
    "CHANNELS": 3,
    "IMG_SHAPE": (32, 32, 3),
    "N_CLASSES": 10,
    "DATA_LOADER": load_datasets.load_data_cifar10,
    "mode": "train",
}
