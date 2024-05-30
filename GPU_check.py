# import tensorflow as tf
# tf.config.experimental.list_physical_devices('GPU')
#
# #
# # # TensorFlow가 GPU를 사용하도록 설정합니다.
# # physical_devices = tf.config.list_physical_devices('GPU')
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)
# #
# # 현재 사용 가능한 GPU 디바이스를 출력합니다.
# print("Available GPUs:", physical_devices)
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import tensorflow as tf

def main():
    # 현재 사용 가능한 GPU 디바이스를 출력합니다.
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    else:
        print("No GPU detected.")

if __name__ == "__main__":
    main()
