import load_datasets

config = {
    "models_dir": "models",
    "DATASET": "CarEnv",
    "target_learning_rate": 0.001,
    "target_model_epochs": 50,
    "target_batch_size": 64,
    "model_name": "CarEnv_PPO_Actor_back.h5",

    "LOAD_TARGET_WEIGHTS": True,
    "AdvGAN_epochs": 250,
    "AdvGAN_batch_size": 128,
    "AdvGAN_Generator_LR": 0.0002,
    "AdvGAN_Discriminator_LR": 0.01,  # default: 0.01 , prev: 0.001

    "generator_model_name": "CarEnv-GeneratorModel.model",
    "discriminator_model_name": "CarEnv-DiscriminatorModel.model",
    "stacked_model_name": "CarEnv-StackedModel.model",
    "load": False,
    "images_name": "CIFAR10Generated",

    "c": 0.3,

    "targeted": False,
    "targets": {
        8: 4
    },
    "target": 8,

    "IMG_W": 64,
    "IMG_H": 64,
    "CHANNELS": 3,
    "IMG_SHAPE": (64, 64, 3),
    "N_CLASSES": 5,
    "mode": "train",
}
