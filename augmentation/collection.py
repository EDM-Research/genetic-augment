import albumentations as A

from augmentation.custom import UniformNoise

blur = {
    "gaussian_blur": {
        "function": A.GaussianBlur,
        "parameters": {
            "blur_limit": (3, 15)
        }
    },
    "blur": {
        "function": A.Blur,
        "parameters": {
            "blur_limit": 7
        }
    },
    "defocus": {
        "function": A.Defocus,
        "parameters": {
            "radius": (3, 10)
        }
    },
    "glass_blur": {
        "function": A.GlassBlur,
        "parameters": {
            "sigma": 0.7
        }
    },
    "median_blur": {
        "function": A.MedianBlur,
        "parameters": {
            "blur_limit": 7
        }
    },
    "motion_blur": {
        "function": A.MotionBlur,
        "parameters": {
            "blur_limit": 7
        }
    },
    "zoom_blur": {
        "function": A.ZoomBlur,
        "parameters": {
            "max_factor": 1.09
        }
    }
}

color = {
    "channel_shuffle": {
        "function": A.ChannelShuffle,
        "parameters": {}
    },
    "clahe": {
        "function": A.CLAHE,
        "parameters": {
            "clip_limit": 4
        }
    },
    "brightness": {
        "function": A.ColorJitter,
        "parameters": {
            "brightness": 0.3,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0
        }
    },
    "contrast": {
        "function": A.ColorJitter,
        "parameters": {
            "brightness": 0.0,
            "contrast": 0.3,
            "saturation": 0.0,
            "hue": 0.0
        }
    },
    "saturation": {
        "function": A.ColorJitter,
        "parameters": {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.5,
            "hue": 0.0
        }
    },
    "hue": {
        "function": A.ColorJitter,
        "parameters": {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.3
        }
    },
    "equalize": {
        "function": A.Equalize,
        "parameters": {
        }
    },
    "fancy_pca": {
        "function": A.FancyPCA,
        "parameters": {
            "alpha": 0.3
        }
    },
    "invert": {
        "function": A.InvertImg,
        "parameters": {
        }
    },
    "posterize": {
        "function": A.Posterize,
        "parameters": {
            "num_bits": 4
        }
    },
    "gamma": {
        "function": A.RandomGamma,
        "parameters": {
            "gamma_limit":(80, 120)
        }
    },
    "tone_curve": {
        "function": A.RandomToneCurve,
        "parameters": {
            "scale": 0.1
        }
    }
}

sharpen = {
    "emboss": {
        "function": A.Emboss,
        "parameters": {
            "strength": (0.2, 0.5)
        }
    },
    "sharpen": {
        "function": A.Sharpen,
        "parameters": {
            "alpha": (0.2, 0.5)
        }
    },
    "unsharp_mask": {
        "function": A.UnsharpMask,
        "parameters": {
            "blur_limit": (3, 15),
            "alpha": 0.5
        }
    }
}

noise = {
    "gauss_noise": {
        "function": A.GaussNoise,
        "parameters": {
            "var_limit": 75.0
        }
    },
    "iso_noise": {
        "function": A.ISONoise,
        "parameters": {
            "color_shift": (0.01, 0.05),
            "intensity": (0.1, 0.5)
        }
    },
    "multiplicative_noise": {
        "function": A.MultiplicativeNoise,
        "parameters": {
            "multiplier": 0.9
        }
    },
    "dropout": {
        "function": A.PixelDropout,
        "parameters": {
            "dropout_prob": 0.01
        }
    },
    "uniform_noise": {
        "function": UniformNoise,
        "parameters": {
            "strength": 0.2
        }
    }
}

augmentations = {}
augmentations.update(blur)
augmentations.update(color)
augmentations.update(sharpen)
augmentations.update(noise)
