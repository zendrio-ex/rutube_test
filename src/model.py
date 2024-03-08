import torch
import torch.nn.functional as F
from torch import nn
from config import settings

labels = {
    0: 'apple_pie',
    1: 'baby_back_ribs',
    2: 'baklava',
    3: 'beef_carpaccio',
    4: 'beef_tartare',
    5: 'beet_salad',
    6: 'beignets',
    7: 'bibimbap',
    8: 'bread_pudding',
    9: 'breakfast_burrito',
    10: 'bruschetta',
    11: 'caesar_salad',
    12: 'cannoli',
    13: 'caprese_salad',
    14: 'carrot_cake',
    15: 'ceviche',
    16: 'cheese_plate',
    17: 'cheesecake',
    18: 'chicken_curry',
    19: 'chicken_quesadilla',
    20: 'chicken_wings',
    21: 'chocolate_cake',
    22: 'chocolate_mousse',
    23: 'churros',
    24: 'clam_chowder',
    25: 'club_sandwich',
    26: 'crab_cakes',
    27: 'creme_brulee',
    28: 'croque_madame',
    29: 'cup_cakes',
    30: 'deviled_eggs',
    31: 'donuts',
    32: 'dumplings',
    33: 'edamame',
    34: 'eggs_benedict',
    35: 'escargots',
    36: 'falafel',
    37: 'filet_mignon',
    38: 'fish_and_chips',
    39: 'foie_gras',
    40: 'french_fries',
    41: 'french_onion_soup',
    42: 'french_toast',
    43: 'fried_calamari',
    44: 'fried_rice',
    45: 'frozen_yogurt',
    46: 'garlic_bread',
    47: 'gnocchi',
    48: 'greek_salad',
    49: 'grilled_cheese_sandwich',
    50: 'grilled_salmon',
    51: 'guacamole',
    52: 'gyoza',
    53: 'hamburger',
    54: 'hot_and_sour_soup',
    55: 'hot_dog',
    56: 'huevos_rancheros',
    57: 'hummus',
    58: 'ice_cream',
    59: 'lasagna',
    60: 'lobster_bisque',
    61: 'lobster_roll_sandwich',
    62: 'macaroni_and_cheese',
    63: 'macarons',
    64: 'miso_soup',
    65: 'mussels',
    66: 'nachos',
    67: 'omelette',
    68: 'onion_rings',
    69: 'oysters',
    70: 'pad_thai',
    71: 'paella',
    72: 'pancakes',
    73: 'panna_cotta',
    74: 'peking_duck',
    75: 'pho',
    76: 'pizza',
    77: 'pork_chop',
    78: 'poutine',
    79: 'prime_rib',
    80: 'pulled_pork_sandwich',
    81: 'ramen',
    82: 'ravioli',
    83: 'red_velvet_cake',
    84: 'risotto',
    85: 'samosa',
    86: 'sashimi',
    87: 'scallops',
    88: 'seaweed_salad',
    89: 'shrimp_and_grits',
    90: 'spaghetti_bolognese',
    91: 'spaghetti_carbonara',
    92: 'spring_rolls',
    93: 'steak',
    94: 'strawberry_shortcake',
    95: 'sushi',
    96: 'tacos',
    97: 'takoyaki',
    98: 'tiramisu',
    99: 'tuna_tartare',
    100: 'waffles'
}


class MyImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 5, 10, stride=1, padding=0, dilation=1)
        self.pool = nn.MaxPool2d(5, 5)
        self.batchnorm_1 = nn.BatchNorm2d(5)  # по каналу
        self.conv_2 = nn.Conv2d(5, 10, 3, stride=1, padding=0, dilation=1)
        self.batchnorm_2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10 * 19 * 19, 150)
        self.fc2 = nn.Linear(150, 101)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv_1(x)))  # [32, 5, 100, 100]
        x = self.batchnorm_1(x)
        x = self.pool(F.relu(self.conv_2(x)))  # [32, 10, 19, 19]
        x = self.batchnorm_2(x)
        x = torch.flatten(x, 1)  # for fc
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def install_model():
    model_type = settings['model_type']
    if model_type == 'my_own_model':
        model = MyImageNet().to(device)
        model.load_state_dict(torch.load('./models/model.pt'))
        model.eval()

    return model