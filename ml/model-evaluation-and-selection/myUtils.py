import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data():
    X = np.array([[1651.  ],
                    [1691.82],
                    [1732.63],
                    [1773.45], 
                    [1814.27],
                    [1855.08],
                    [1895.9 ],
                    [1936.71],
                    [1977.53],
                    [2018.35],
                    [2059.16],
                    [2099.98],
                    [2140.8 ],
                    [2181.61],
                    [2222.43],
                    [2263.24],
                    [2304.06],
                    [2344.88],
                    [2385.69],
                    [2426.51],
                    [2467.33],
                    [2508.14],
                    [2548.96],
                    [2589.78],
                    [2630.59],
                    [2671.41],
                    [2712.22],
                    [2753.04],
                    [2793.86],
                    [2834.67],
                    [2875.49],
                    [2916.31],
                    [2957.12],
                    [2997.94],
                    [3038.76],
                    [3079.57],
                    [3120.39],
                    [3161.2 ],
                    [3202.02],
                    [3242.84],
                    [3283.65],
                    [3324.47],
                    [3365.29],
                    [3406.1 ],
                    [3446.92],
                    [3487.73],
                    [3528.55],
                    [3569.37],
                    [3610.18],
                    [3651.  ]])
    Y = np.array([[432.65],
                    [454.94],
                    [471.53],
                    [482.51],
                    [468.36],
                    [482.15],
                    [540.02],
                    [534.58],
                    [558.35],
                    [566.42],
                    [581.4 ],
                    [596.46],
                    [596.71],
                    [619.45],
                    [616.58],
                    [653.16],
                    [666.52],
                    [670.59],
                    [669.02],
                    [678.91],
                    [707.44],
                    [710.76],
                    [745.19],
                    [729.85],
                    [743.8 ],
                    [738.2 ],
                    [772.95],
                    [772.22],
                    [784.21],
                    [776.43],
                    [804.78],
                    [833.27],
                    [825.69],
                    [821.05],
                    [833.82],
                    [833.06],
                    [825.7 ],
                    [843.58],
                    [869.4 ],
                    [851.5 ],
                    [863.18],
                    [853.01],
                    [877.16],
                    [863.74],
                    [874.67],
                    [877.74],
                    [874.11],
                    [882.8 ],
                    [910.83],
                    [897.42]])
    return (X,Y)


def build_models():
    model1 = tf.keras.Sequential(
        [
            Dense(25, activation='relu', name = 'layer1'),
            Dense(15, activation='relu', name = 'layer2'),
            Dense(1, activation='linear', name = 'layer3')
        ]
    )

    model2 = tf.keras.Sequential(
        [
            Dense(20, activation='relu', name = 'layer1'),
            Dense(12, activation='relu', name = 'layer2'),
            Dense(12, activation='relu', name = 'layer3'),
            Dense(20, activation='relu', name = 'layer4'),
            Dense(1, activation='linear', name = 'layer5')
        ]
    )

    model3 = tf.keras.Sequential(
        [
            Dense(32, activation='relu', name = 'layer1'),
            Dense(16, activation='relu', name = 'layer2'),
            Dense(8, activation='relu', name = 'layer3'),
            Dense(4, activation='relu', name = 'layer4'),
            Dense(12, activation='relu', name = 'layer5'),
            Dense(1, activation='linear', name = 'layer6')
        ]
    )
    models = np.array([[model1], [model2], [model3]])
    return models