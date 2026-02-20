import logging

from demos.basic_hand_draw import basic_hand_draw_demo
from demos.make_dataset import make_dataset_demo
from demos.label_dataset import label_dataset_demo

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # basic_hand_draw_demo()
    # make_dataset_demo()
    label_dataset_demo()