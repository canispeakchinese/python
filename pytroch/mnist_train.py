import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    
    all_values = data_list[0].split(',')
    print(all_values[0])
    image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28,28))

    # 创建子图
    #fig, (ax1) = plt.subplots(1, 1)
    #ax1.imshow(image_array, cmap='Greys', interpolation='None')
    #plt.show()

    scaled_input = (numpy.asarray(all_values[1:], dtype=float)/255.0*0.99)+0.01
    print(scaled_input)