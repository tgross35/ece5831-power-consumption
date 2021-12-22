"""Constants used across separate runs."""

RUN_CONSTANTS = {
    "run1": {
        "BATCH_TYPES": ("linear", "log"),
        "NUM_EPOCHS": [10],
        "NUM_LAYERS": [2, 3, 5],
        "EPOCH_MULTIPLIER": [0.01, 0.05, 0.1, 0.2],
        "MODELS_DIRECTORY": "data/run1_models",
        "EXPORT_DIRECTORY": "data/run1_models_export",
        "LOG_MODELS_DIRECTORY": "logs/run1_models",
        "LOG_FIT_DIRECTORY": "logs/run1_fit",
    },
    "run2": {
        "BATCH_TYPES": ("linear",),
        "NUM_EPOCHS": [20],
        "NUM_LAYERS": [3, 4],
        "EPOCH_MULTIPLIER": [0.01, 0.05],
        "MODELS_DIRECTORY": "data/run2_models",
        "EXPORT_DIRECTORY": "data/run2_models_export",
        "LOG_MODELS_DIRECTORY": "logs/run2_models",
        "LOG_FIT_DIRECTORY": "logs/run2_fit",
    },
}
