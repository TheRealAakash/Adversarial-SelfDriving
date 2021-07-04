import load_datasets

config = {
    "models_dir": "models",
    "DATASET": "HighRes",
    "target_learning_rate": 0.001,
    "target_model_epochs": 50,
    "target_batch_size": 64,
    "model_name": "HighRes-KerasModel.model",

    "LOAD_TARGET_WEIGHTS": False,
    "AdvGAN_epochs": 60,
    "AdvGAN_batch_size": 128,
    "AdvGAN_Generator_LR": 0.0002,
    "AdvGAN_Discriminator_LR": 0.001,  # 0.01

    "generator_model_name": "HighRes-GeneratorModel.model",
    "discriminator_model_name": "HighRes-DiscriminatorModel.model",
    "load_generator": False,
    "load_discriminator": False,
    "images_name": "HighResGenerated",

    "c": 0.3,

    "targeted": True,
    "targets": {
        8: 4
    },
    "target": 8,

    "IMG_W": 299,
    "IMG_H": 299,
    "CHANNELS": 3,
    "IMG_SHAPE": (299, 299, 3),
    "N_CLASSES": 1000,
    "DATA_LOADER": load_datasets.load_data_high_res,
    "mode": "train",
}
