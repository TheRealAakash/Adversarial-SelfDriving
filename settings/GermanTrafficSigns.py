import load_datasets

config = {
    "models_dir": "models",
    "DATASET": "TrafficSigns",
    "target_learning_rate": 0.001,
    "target_model_epochs": 50,
    "target_batch_size": 64,
    "model_name": "TrafficSignRecognition-KerasModel.model",

    "LOAD_TARGET_WEIGHTS": True,
    "AdvGAN_epochs": 60,
    "AdvGAN_batch_size": 128,
    "AdvGAN_Generator_LR": 0.0002,
    "AdvGAN_Discriminator_LR": 0.001,  # 0.01

    "generator_model_name": "TrafficSignRecognition-GeneratorModel.model",
    "discriminator_model_name": "TrafficSignRecognition-DiscriminatorModel.model",
    "load_generator": True,
    "load_discriminator": True,
    "images_name": "TrafficSignGenerated",

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
    "N_CLASSES": 43,
    "DATA_LOADER": load_datasets.load_data_traffic_signs,
    "mode": "train",
}
